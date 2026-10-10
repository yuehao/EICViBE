"""The virtual accelerator: machines that keep running, what people change on them, and the log (ADR 0008).

A machine is a ring tracked turn after turn or a route shot after shot (ADR
0007), on the lattice of the machine files, with an operating mode's settings
and a live layer of what people changed. Shared machines are the ones everyone
works on, as on the real accelerator; a sandbox is a user's own copy of one, to
try things without disturbing others. Every change goes through
:meth:`VirtualAccelerator.write`: checked against the user's role and the
channel's limits, appended to the write log with the old value and a reason,
applied to the live layer and then to the running engine. The machine files
are never changed; resetting a machine drops its live layer.
"""

import re
import threading
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from eicvibe.domain.lattice import Lattice, Route
from eicvibe.domain.machine import MachineRegistry
from eicvibe.engines.contracts import Fidelity

from .settings import MachineSettings, Values
from .store import ANONYMOUS, Role, Store, User


class Forbidden(Exception):
    """The user's role, or ownership, does not allow it."""


class Conflict(Exception):
    """The machine's state does not allow it now (e.g. removing a running machine)."""


class MachineSpec(BaseModel):
    """What a machine is: what it tracks, with which engine and mode, and whose it is."""

    id: str
    name: str
    kind: Literal["ring", "route"]
    target: str = Field(description="The ring's branch name, or the route's name")
    engine: str | None = Field(None, description="xsuite or trackpad; None: TrackPad when installed")
    fidelity: str = "standard"
    mode: str | None = Field(None, description="An operating mode of the machine files, e.g. bar_mode")
    shared: bool = True
    owner: str
    parent: str | None = Field(None, description="The machine a sandbox was copied from")
    particles: int = Field(50, ge=1, le=100_000)
    rate: float | None = Field(None, gt=0, description="Turns (ring) or shots (route) per second, at most")
    ramp_turns: int = Field(
        1024,
        ge=0,
        le=1_000_000,
        description="Turns a ring's setting change ramps over (a half cosine); 0 steps",
    )
    autostart: bool = Field(False, description="Started whenever the service starts")


@dataclass
class Source:
    """The machine files a virtual accelerator runs: lattice, registry, routes and operating modes."""

    name: str
    lattice: Lattice
    registry: MachineRegistry
    routes: dict[str, Route]
    folder: Path | None = None
    """A machine folder, whose settings/ holds the modes; None for a lattice file."""

    def modes(self) -> list[str]:
        if self.folder is None:
            return []
        return sorted(p.stem for p in (self.folder / "settings").glob("*.yaml"))

    def mode_values(self, mode: str | None) -> Values:
        if not mode:
            return {}
        if mode not in self.modes():
            raise ValueError(f"no mode {mode}; modes: {', '.join(self.modes()) or 'none'}")
        from eicvibe.store import read_settings

        return {(s.occurrence, s.parameter): s.value for s in read_settings(self.folder, mode, self.lattice)}


def load_source(path: str | Path) -> Source:
    """A machine folder, or a ``<stem>_lattice.json`` with its registry (and routes) files beside it."""
    from eicvibe.store import is_machine, load_machine
    from eicvibe.store.machine import read_routes_file

    path = Path(path)
    if is_machine(path):
        machine = load_machine(path)
        return Source(
            path.name, machine.lattice, machine.registry, {r.name: r for r in machine.routes}, folder=path
        )
    stem = path.name.removesuffix("_lattice.json")
    registry_path = path.with_name(f"{stem}_registry.json")
    if not registry_path.is_file():
        raise ValueError(
            f"a virtual accelerator needs a machine registry: no {registry_path.name} beside {path}"
        )
    lattice = Lattice.from_json(path.read_text())
    registry = MachineRegistry.model_validate_json(registry_path.read_text())
    routes = read_routes_file(path.with_name(f"{stem}_routes.yaml"), lattice)
    return Source(stem, lattice, registry, {r.name: r for r in routes})


@dataclass
class Instance:
    spec: MachineSpec
    state: str = "stopped"
    """stopped, starting, running, stopping or failed."""
    error: str | None = None
    run: Any = None
    started: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def run_id(self) -> str | None:
        return self.run.run_id if self.run is not None else None

    @property
    def position(self) -> int | None:
        """The run's turn (ring) or shot (route) now."""
        if self.run is None:
            return None
        return self.run.turn if self.spec.kind == "ring" else self.run.shot


def _slug(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    if not slug:
        raise ValueError("a machine needs a name with letters or digits")
    return slug[:40]


class VirtualAccelerator:
    """Machines on one set of machine files, their settings, users and logs, in a state folder."""

    def __init__(
        self,
        source: Source,
        state: str | Path,
        *,
        sandbox_quota: int = 2,
        ring_rate: float = 2000.0,
        route_rate: float = 2.0,
        retention_bytes: int = 256 << 20,
    ) -> None:
        self.source = source
        self.state = Path(state)
        self.runs = self.state / "runs"
        self.store = Store(self.state / "va.sqlite")
        self.sandbox_quota = sandbox_quota
        self.ring_rate = ring_rate
        self.route_rate = route_rate
        self.retention_bytes = retention_bytes
        self._lock = threading.RLock()
        self._settings: dict[str | None, MachineSettings] = {}
        self._instances = {m["id"]: Instance(MachineSpec(**_spec_fields(m))) for m in self.store.machines()}

    # -- life ---------------------------------------------------------------------------------

    def first_admin(self, password: str | None = None) -> str | None:
        """With no users yet, add ``admin`` and return its password (random unless given); else None."""
        if self.store.has_users():
            return None
        password = self.store.add_user("admin", Role.ADMIN, password)
        self.store.act("service", "add user", name="admin", role="admin")
        return password

    def autostart(self) -> None:
        for instance in list(self._instances.values()):
            if instance.spec.autostart:
                self._launch(instance, "service", "autostart")

    def close(self) -> None:
        """Stop every machine (the live settings stay; a shared machine with autostart comes back)."""
        for instance in list(self._instances.values()):
            run = instance.run
            if run is not None:
                run.stop(timeout=30)
                instance.run, instance.state = None, "stopped"
        self.store.close()

    # -- reading ------------------------------------------------------------------------------

    def options(self) -> dict[str, Any]:
        """What machines can be made of: rings, routes, modes and engines."""
        from eicvibe.engines import engine

        rings = [b.name for b in self.source.lattice.branches if self.source.lattice.layout(b.id).periodic]
        engines = []
        for name in ("xsuite", "trackpad"):
            try:
                engines.append({"name": name, "available": engine(name).available()})
            except Exception:  # noqa: BLE001 - an engine that cannot even load is not available
                engines.append({"name": name, "available": False})
        return {
            "source": self.source.name,
            "rings": rings,
            "routes": sorted(self.source.routes),
            "modes": self.source.modes(),
            "engines": engines,
            "fidelities": [f.value for f in Fidelity],
            "sandbox_quota": self.sandbox_quota,
        }

    def machines(self) -> list[dict[str, Any]]:
        return [self._entry(i) for i in self._instances.values()]

    def machine(self, machine_id: str) -> dict[str, Any]:
        return self._entry(self._instance(machine_id))

    def _entry(self, instance: Instance) -> dict[str, Any]:
        live = self.store.live(instance.spec.id)
        return instance.spec.model_dump() | {
            "state": instance.state,
            "error": instance.error,
            "run": instance.run_id,
            "position": instance.position,
            "started": instance.started,
            "changes": len(live),
            "warnings": list(getattr(instance.run, "warnings", []) or []),
        }

    def settings(self, machine_id: str, changed: bool = False) -> list[dict[str, Any]]:
        """Every setting channel: original (design and mode), live, the value now, who changed it."""
        instance = self._instance(machine_id)
        settings = self._machine_settings(instance.spec.mode)
        live = self.store.live(machine_id)
        rows = []
        for name, s in settings.settings.items():
            entry = live.get(name)
            if changed and entry is None:
                continue
            rows.append(
                {
                    "channel": name,
                    "device": s.device,
                    "unit": s.unit,
                    "kind": s.kind,
                    "limits": list(s.limits) if s.limits else None,
                    "original": s.original,
                    "value": entry["value"] if entry else s.original,
                    "changed": entry is not None,
                    "by": entry["user"] if entry else None,
                    "at": entry["time"] if entry else None,
                    "occurrences": sorted({o for o, _ in s.keys}),
                }
            )
        return rows

    def live(self, machine_id: str) -> dict[str, float]:
        return {c: e["value"] for c, e in self.store.live(machine_id).items()}

    # -- machines -----------------------------------------------------------------------------

    def create(
        self,
        user: User,
        *,
        name: str,
        kind: str,
        target: str,
        engine: str | None = None,
        fidelity: str = "standard",
        mode: str | None = None,
        shared: bool = True,
        particles: int = 50,
        rate: float | None = None,
        ramp_turns: int = 1024,
        autostart: bool = False,
        reason: str = "",
    ) -> dict[str, Any]:
        """A new machine: shared (administrators), or a sandbox of the user's own."""
        self._require(user, Role.ADMIN if shared else Role.PHYSICIST)
        if not shared:
            self._quota(user)
        if kind == "ring":
            branch = self.source.lattice.layout(target)
            if not branch.periodic:
                raise ValueError(f"{target} is a line, not a ring; a line runs as a route")
            target = branch.branch.name
        elif kind == "route":
            if target not in self.source.routes:
                raise ValueError(f"no route {target}; routes: {', '.join(self.source.routes) or 'none'}")
        else:
            raise ValueError("a machine is a ring or a route")
        Fidelity(fidelity)
        self.source.mode_values(mode)  # the mode exists
        with self._lock:
            used = self.store.machine_ids()
            base = _slug(name)
            machine_id, n = base, 2
            while machine_id in used:
                machine_id, n = f"{base}-{n}", n + 1
            spec = MachineSpec(
                id=machine_id,
                name=name,
                kind=kind,
                target=target,
                engine=engine,
                fidelity=fidelity,
                mode=mode,
                shared=shared,
                owner=user.name,
                particles=particles,
                rate=rate,
                ramp_turns=ramp_turns,
                autostart=autostart and shared,
            )
            self.store.add_machine(machine_id, spec.model_dump(), user.name)
            self._instances[machine_id] = Instance(spec)
        self.store.act(user.name, "create", machine_id, reason, **spec.model_dump(exclude={"id"}))
        return self._entry(self._instances[machine_id])

    def fork(self, user: User, machine_id: str, name: str | None = None, reason: str = "") -> dict[str, Any]:
        """A sandbox of the user's own, with the machine's settings as they are now."""
        source = self._instance(machine_id).spec
        entry = self.create(
            user,
            name=name or f"{source.name} ({user.name})",
            kind=source.kind,
            target=source.target,
            engine=source.engine,
            fidelity=source.fidelity,
            mode=source.mode,
            shared=False,
            particles=source.particles,
            rate=source.rate,
            ramp_turns=source.ramp_turns,
            reason=reason,
        )
        instance = self._instances[entry["id"]]
        instance.spec = instance.spec.model_copy(update={"parent": machine_id})
        self.store.update_machine(instance.spec.id, instance.spec.model_dump())
        copied = self.store.copy_live(machine_id, instance.spec.id, user.name)
        self.store.act(user.name, "fork", instance.spec.id, reason, source=machine_id, settings=copied)
        return self._entry(instance)

    def remove(self, user: User, machine_id: str, reason: str = "") -> None:
        instance = self._instance(machine_id)
        self._may(user, instance, Role.ADMIN)
        if instance.state not in ("stopped", "failed"):
            raise Conflict(f"stop {machine_id} first")
        with self._lock:
            self.store.remove_machine(machine_id)
            del self._instances[machine_id]
        self.store.act(user.name, "remove", machine_id, reason)

    def start(self, user: User, machine_id: str, reason: str = "") -> dict[str, Any]:
        instance = self._instance(machine_id)
        self._may(user, instance, Role.OPERATOR)
        self._launch(instance, user.name, reason)
        return self._entry(instance)

    def stop(self, user: User, machine_id: str, reason: str = "", wait: bool = False) -> dict[str, Any]:
        instance = self._instance(machine_id)
        self._may(user, instance, Role.OPERATOR)
        with instance.lock:
            if instance.state not in ("running", "starting"):
                raise Conflict(f"{machine_id} is {instance.state}")
            run, instance.state = instance.run, "stopping"
        self.store.act(user.name, "stop", machine_id, reason, run=instance.run_id)

        def stop() -> None:
            if run is not None:
                run.stop(timeout=60)
            with instance.lock:
                instance.run, instance.state = None, "stopped"

        thread = threading.Thread(target=stop, name=f"stop {machine_id}", daemon=True)
        thread.start()
        if wait:
            thread.join()
        return self._entry(instance)

    def _launch(self, instance: Instance, user: str, reason: str) -> None:
        with instance.lock:
            if instance.state in ("starting", "running", "stopping"):
                raise Conflict(f"{instance.spec.id} is {instance.state}")
            instance.state, instance.error = "starting", None
        self.store.act(user, "start", instance.spec.id, reason)

        def launch() -> None:
            try:
                run = self._run(instance.spec)
                run.start()
                with instance.lock:
                    instance.run, instance.state = run, "running"
                    instance.started = datetime.now(UTC).isoformat(timespec="seconds")
                self.store.act("service", "running", instance.spec.id, run=run.run_id, engine=run.engine)
            except Exception as error:  # noqa: BLE001 - reported on the machine and in the log
                with instance.lock:
                    instance.state, instance.error = "failed", f"{type(error).__name__}: {error}"
                self.store.act("service", "failed", instance.spec.id, error=instance.error)

        threading.Thread(target=launch, name=f"start {instance.spec.id}", daemon=True).start()

    def _run(self, spec: MachineSpec):
        """A ring or route run of the machine as it is now: its mode's settings and its live ones."""
        from eicvibe.virtual import Beam, RingRun, RouteRun

        settings = self._machine_settings(spec.mode)
        values = self._for(spec, settings.start_values(self.live(spec.id)))
        run_id = f"{spec.id}-{datetime.now(UTC):%Y%m%dT%H%M%S}"
        common = {
            "engine": spec.engine,
            "fidelity": Fidelity(spec.fidelity),
            "beam": Beam(particles=spec.particles),
            "values": values,
            "registry": self.source.registry,
            "retention_bytes": self.retention_bytes,
            "run_id": run_id,
            "machine": spec.id,
        }
        lattice = self.source.lattice
        if spec.kind == "ring":
            branch = lattice.layout(spec.target).id
            return RingRun(
                lattice,
                branch,
                self.runs / run_id,
                max_rate=spec.rate or self.ring_rate,
                ramp_turns=spec.ramp_turns,
                **common,
            )
        route = self.source.routes[spec.target]
        return RouteRun(lattice, route, self.runs / run_id, rate=spec.rate or self.route_rate, **common)

    def _for(self, spec: MachineSpec, values: Values) -> Values:
        """The values a machine's engine takes: those of the branches it tracks."""
        lattice = self.source.lattice
        if spec.kind == "ring":
            branches = {lattice.layout(spec.target).id}
        else:
            branches = {s.branch for s in self.source.routes[spec.target].segments}
        return {k: v for k, v in values.items() if lattice.placement(k[0]).branch_id in branches}

    # -- settings ---------------------------------------------------------------------------------

    def write(
        self,
        user: User,
        machine_id: str,
        changes: Mapping[str, float],
        reason: str = "",
        *,
        atomic: bool = True,
        correlation: str | None = None,
    ) -> dict[str, Any]:
        """Write setting channels of a machine: checked, logged with the old values, applied.

        With ``atomic`` (the default) one refused write refuses them all, as a correction
        applied in part is not the correction; every one is logged either way.
        """
        instance = self._instance(machine_id)
        self._may(user, instance, Role.PHYSICIST)
        if not changes:
            raise ValueError("no changes")
        settings = self._machine_settings(instance.spec.mode)
        correlation = correlation or uuid.uuid4().hex[:12]
        with self._lock:
            live = self.live(machine_id)
            entries = []
            for channel, value in changes.items():
                value = float(value)
                refusal = "not a finite number" if value != value or abs(value) == float("inf") else None
                refusal = refusal or settings.check(channel, value)
                setting = settings.settings.get(channel)
                original = setting.original if setting else None
                entries.append(
                    {
                        "channel": channel,
                        "unit": setting.unit if setting else "",
                        "old": settings.value(channel, live) if setting else None,
                        "new": value,
                        "applied": refusal is None,
                        "refusal": refusal,
                        "original": original is not None
                        and abs(value - original) <= 1e-12 * max(1.0, abs(original)),
                    }
                )
            if atomic and any(not e["applied"] for e in entries):
                for e in entries:
                    if e["applied"]:
                        e["applied"], e["refusal"] = False, "another write of this change was refused"
            applied = {e["channel"]: e["new"] for e in entries if e["applied"]}
            self.store.record_writes(
                machine_id, user.name, entries, correlation, reason, instance.run_id, instance.position
            )
            after = self.live(machine_id)
        if applied and instance.run is not None and instance.state == "running":
            values = self._for(instance.spec, settings.lattice_values(after, applied))
            if values:
                instance.run.update(values, by=user.name, correlation=correlation, reason=reason)
        return {
            "correlation": correlation,
            "applied": bool(applied),
            "writes": [
                {k: e[k] for k in ("channel", "unit", "old", "new", "applied", "refusal")} for e in entries
            ],
        }

    def reset(
        self, user: User, machine_id: str, channels: list[str] | None = None, reason: str = ""
    ) -> dict[str, Any]:
        """Put channels (all changed ones by default) back to their original values, logged as writes."""
        instance = self._instance(machine_id)
        self._may(user, instance, Role.OPERATOR if channels is None else Role.PHYSICIST)
        settings = self._machine_settings(instance.spec.mode)
        live = self.live(machine_id)
        names = [c for c in (channels if channels is not None else live) if c in live]
        if not names:
            return {"correlation": None, "applied": False, "writes": []}
        return self.write(
            user, machine_id, {c: settings.settings[c].original for c in names}, reason or "reset to original"
        )

    def snapshot(self, user: User, machine_id: str, name: str) -> dict[str, Any]:
        self._require(user, Role.PHYSICIST)
        self._instance(machine_id)
        live = self.live(machine_id)
        snapshot_id = self.store.add_snapshot(machine_id, name, user.name, live)
        self.store.act(user.name, "snapshot", machine_id, name, snapshot=snapshot_id, settings=len(live))
        return {"id": snapshot_id, "name": name, "changes": len(live)}

    def restore(self, user: User, machine_id: str, snapshot_id: int, reason: str = "") -> dict[str, Any]:
        """Bring a machine's settings to a snapshot's (of it or another machine), as one logged change."""
        instance = self._instance(machine_id)
        self._may(user, instance, Role.OPERATOR)
        snapshot = self.store.snapshot(snapshot_id)
        settings = self._machine_settings(instance.spec.mode)
        live = self.live(machine_id)
        wanted = {c: v for c, v in snapshot["live"].items() if c in settings.settings}
        changes = {c: v for c, v in wanted.items() if live.get(c) != v}
        changes |= {c: settings.settings[c].original for c in live if c not in wanted}
        if not changes:
            return {"correlation": None, "applied": False, "writes": []}
        return self.write(user, machine_id, changes, reason or f"restore snapshot {snapshot['name']}")

    def beam(
        self, user: User, machine_id: str, action: str, x: float = 0.0, y: float = 0.0, reason: str = ""
    ) -> None:
        """Kick (a pinger at the ring's start, rad) or inject a fresh bunch."""
        instance = self._instance(machine_id)
        self._may(user, instance, Role.OPERATOR)
        if instance.spec.kind != "ring":
            raise ValueError("kicks and injections are for rings; a route's every shot is a new bunch")
        if instance.state != "running":
            raise Conflict(f"{machine_id} is {instance.state}")
        if action == "kick":
            instance.run.kick(x, y)
        elif action == "inject":
            instance.run.inject()
        else:
            raise ValueError("the beam takes kick or inject")
        self.store.act(
            user.name, action, machine_id, reason, x=x, y=y, run=instance.run_id, turn=instance.position
        )

    def acquire(self, user: User, machine_id: str, reason: str = "") -> None:
        """Capture a ring's BPM buffer as it is now (an acquisition on request; it changes nothing)."""
        instance = self._instance(machine_id)
        self._require(user, Role.PHYSICIST)
        if instance.spec.kind != "ring":
            raise ValueError("a route records every shot; captures are a ring's")
        if instance.state != "running":
            raise Conflict(f"{machine_id} is {instance.state}")
        instance.run.acquire()
        self.store.act(user.name, "acquire", machine_id, reason, run=instance.run_id, turn=instance.position)

    # -- helpers --------------------------------------------------------------------------------

    def _instance(self, machine_id: str) -> Instance:
        try:
            return self._instances[machine_id]
        except KeyError:
            raise KeyError(f"no machine {machine_id}") from None

    def _machine_settings(self, mode: str | None) -> MachineSettings:
        with self._lock:
            if mode not in self._settings:
                self._settings[mode] = MachineSettings(
                    self.source.lattice, self.source.registry, self.source.mode_values(mode)
                )
            return self._settings[mode]

    @staticmethod
    def _require(user: User, role: Role) -> None:
        if user is ANONYMOUS:
            raise Forbidden("log in first")
        if not user.can(role):
            raise Forbidden(f"{user.name} is a {user.role.label}; this needs a {role.label}")

    def _may(self, user: User, instance: Instance, shared_role: Role) -> None:
        """Shared machines need ``shared_role``; a sandbox, its owner (or an administrator)."""
        if instance.spec.shared:
            self._require(user, shared_role)
            return
        self._require(user, Role.PHYSICIST if user.name == instance.spec.owner else Role.ADMIN)

    def _quota(self, user: User) -> None:
        own = [i for i in self._instances.values() if not i.spec.shared and i.spec.owner == user.name]
        if len(own) >= self.sandbox_quota and not user.can(Role.ADMIN):
            raise Forbidden(f"{user.name} has {len(own)} sandboxes, the most allowed; remove one first")


def _spec_fields(stored: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in stored.items() if k in MachineSpec.model_fields}
