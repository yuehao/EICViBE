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

import json
import logging
import math
import re
import shutil
import threading
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from eicvibe.ai import AiConfig, load_config, save_config
from eicvibe.domain.lattice import Lattice, Route
from eicvibe.domain.machine import MachineRegistry
from eicvibe.engines.contracts import Effect, Fidelity, Options
from eicvibe.virtual.errors import ErrorSpec

from .settings import MachineSettings, Values
from .store import ANONYMOUS, Role, Store, User, now

_log = logging.getLogger(__name__)


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
    engine: str | None = Field(
        None, description="xsuite or trackpad; None: TrackPad when installed, Xsuite for a ring with a ramp"
    )
    fidelity: str = "standard"
    mode: str | None = Field(None, description="An operating mode of the machine files: settings/<mode>.yaml")
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
    errors: ErrorSpec = Field(
        default_factory=ErrorSpec,
        description="Its magnets', supplies' and BPMs' errors, drawn from their seed as it starts (ADR 0011)",
    )


@dataclass
class Source:
    """The machine files a virtual accelerator runs: lattice, registry, routes and operating modes."""

    name: str
    lattice: Lattice
    registry: MachineRegistry
    routes: dict[str, Route]
    folder: Path | None = None
    """A machine folder, whose settings/ holds the modes; None for a lattice file."""
    beams: dict[str, Any] = field(default_factory=dict)
    """Beams delivered at branch starts (a machine folder's beams/), by branch name (ADR 0012)."""
    ramps: dict[str, Any] = field(default_factory=dict)
    """Rings' ramp profiles (a machine folder's ramps/), by the ring's branch name (ADR 0013)."""

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
            path.name,
            machine.lattice,
            machine.registry,
            {r.name: r for r in machine.routes},
            folder=path,
            beams=dict(machine.beams),
            ramps={r.branch: r for r in machine.ramps.values()},
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

    # -- the AI features (ADR 0009) -----------------------------------------------------------

    #: The server's own AI configuration (its command line), until an administrator saves one.
    ai_default: AiConfig = AiConfig()

    @property
    def _ai_file(self) -> Path:
        return self.state / "ai.json"

    def ai_config(self) -> AiConfig:
        """The AI configuration in force: the one saved here, or the server's default."""
        return load_config(self._ai_file) or self.ai_default

    def ai_saved(self) -> bool:
        return self._ai_file.is_file()

    def set_ai(
        self,
        user: User,
        *,
        enabled: bool,
        provider: str,
        model: str = "",
        url: str = "",
        api_key: str | None = None,
        clear_key: bool = False,
        reason: str = "",
    ) -> AiConfig:
        """Choose the AI features' model (administrators); a new key replaces the provider's, unlogged."""
        self._require(user, Role.ADMIN)
        current = self.ai_config()
        keys = dict(current.keys)
        if clear_key:
            keys.pop(provider, None)
            key = "cleared"
        elif api_key:
            keys[provider] = api_key.strip()
            key = "changed"
        else:
            key = "kept" if provider in keys else "none stored"
        config = AiConfig(enabled=enabled, provider=provider, model=model.strip(), url=url.strip(), keys=keys)
        save_config(self._ai_file, config)
        self.store.act(
            user.name,
            "ai settings",
            None,
            reason,
            enabled=enabled,
            provider=provider,
            model=model,
            url=url,
            key=key,
        )
        return config

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
            "ramp": self._ramp(instance.spec),
            "cycle": getattr(instance.run, "cycle", None) if instance.state == "running" else None,
        }

    def _ramp(self, spec: MachineSpec) -> str | None:
        """The ramp profile a ring machine runs on command (ADR 0013); None for a route or a ring without."""
        profile = self.source.ramps.get(spec.target) if spec.kind == "ring" else None
        return profile.name if profile is not None else None

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
                    "source": s.source,
                    "beyond": s.beyond,
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
        errors: ErrorSpec | Mapping[str, Any] | None = None,
        reason: str = "",
    ) -> dict[str, Any]:
        """A new machine: shared (administrators), or a sandbox of the user's own.

        ``errors`` (ADR 0011): by default none on magnets and supplies, the BPMs' usual ones; a spec
        without a seed gets one drawn now, so the machine has the same errors every time it starts.
        """
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
                errors=_seeded(ErrorSpec.model_validate(errors or {}), None),
            )
            self.store.add_machine(machine_id, spec.model_dump(), user.name)
            self._instances[machine_id] = Instance(spec)
        self.store.act(
            user.name,
            "create",
            machine_id,
            reason,
            **spec.model_dump(exclude={"id", "errors"}),
            errors=_described(spec.errors),
        )
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
            errors=source.errors,  # a copy of the same machine: the same errors
            reason=reason,
        )
        instance = self._instances[entry["id"]]
        instance.spec = instance.spec.model_copy(update={"parent": machine_id})
        self.store.update_machine(instance.spec.id, instance.spec.model_dump())
        copied = self.store.copy_live(machine_id, instance.spec.id, user.name)
        self.store.act(user.name, "fork", instance.spec.id, reason, source=machine_id, settings=copied)
        return self._entry(instance)

    def errors(self, user: User, machine_id: str) -> dict[str, Any]:
        """A machine's errors as specified, seed included, and what each row chooses: for whoever
        may configure it."""
        instance = self._instance(machine_id)
        self._may(user, instance, Role.ADMIN)
        return self._errors_entry(instance.spec, instance.spec.errors)

    def check_errors(
        self, user: User, machine_id: str, errors: ErrorSpec | Mapping[str, Any]
    ) -> dict[str, Any]:
        """What ``errors`` would choose on a machine: magnets and supplies by row, and the rows that
        choose nothing. Nothing is saved."""
        instance = self._instance(machine_id)
        self._may(user, instance, Role.ADMIN)
        return self._errors_entry(instance.spec, ErrorSpec.model_validate(errors))

    def _errors_entry(self, spec: MachineSpec, errors: ErrorSpec) -> dict[str, Any]:
        from eicvibe.virtual import MachineErrors

        drawn = MachineErrors(errors, self.source.lattice, self._branches(spec), self.source.registry)
        return {"errors": errors.model_dump(mode="json"), "chosen": drawn.chosen, "problems": drawn.problems}

    def set_errors(
        self, user: User, machine_id: str, errors: ErrorSpec | Mapping[str, Any], reason: str = ""
    ) -> dict[str, Any]:
        """Give a machine other errors. Its magnets' and supplies' (and the seed) only while it is
        stopped: it draws them when it starts. Its BPMs' also while it runs: they read the beam and
        leave it alone, so the run reads with them from its next turn. Without a seed, it keeps its own:
        the same magnets keep the same errors where their spec is the same."""
        instance = self._instance(machine_id)
        self._may(user, instance, Role.ADMIN)
        errors = _seeded(ErrorSpec.model_validate(errors), instance.spec.errors.seed)
        current = instance.spec.errors
        bpms_only = errors.model_dump(exclude={"bpms"}) == current.model_dump(exclude={"bpms"})
        if problems := self._errors_entry(instance.spec, errors)["problems"]:
            raise ValueError(
                "; ".join(problems)
                + ": nothing would be drawn for it; check its kinds and names (no names: all)"
            )
        with instance.lock:
            if instance.state not in ("stopped", "failed", "running"):
                raise Conflict(f"{machine_id} is {instance.state}: wait until it runs or has stopped")
            if instance.state == "running" and not bpms_only:
                raise Conflict(
                    f"stop {machine_id} to change its magnets' or supplies' errors or its seed: it draws "
                    "them as it starts (its BPMs' errors change while it runs)"
                )
            with self._lock:
                instance.spec = instance.spec.model_copy(update={"errors": errors})
                self.store.update_machine(machine_id, instance.spec.model_dump())
            live = instance.state == "running" and errors.bpms != current.bpms
            if live:
                instance.run.set_bpms(errors.bpm_model())
        self.store.act(
            user.name,
            "errors",
            machine_id,
            reason,
            errors=_described(errors),
            seed="new" if errors.seed != current.seed else "kept",
            **({"run": instance.run_id} if live else {}),
        )
        return self.errors(user, machine_id)

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
        from eicvibe.virtual import Beam, MachineErrors, RingRun, RouteRun

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
            "bpm": spec.errors.bpm_model(),
        }
        lattice = self.source.lattice
        common["errors"] = MachineErrors(spec.errors, lattice, self._branches(spec), self.source.registry)
        if spec.kind == "ring":
            branch = lattice.layout(spec.target).id
            return RingRun(
                lattice,
                branch,
                self.runs / run_id,
                max_rate=spec.rate or self.ring_rate,
                ramp_turns=spec.ramp_turns,
                ramp=self.source.ramps.get(spec.target),
                **common,
            )
        route = self.source.routes[spec.target]
        first = route.segments[0]
        delivered = self.source.beams.get(lattice.branch(first.branch).name)
        if delivered is not None and lattice.layout(first.branch).placements[0].id != first.start:
            delivered = None  # delivered at the branch's start; this route starts further on
        return RouteRun(
            lattice,
            route,
            self.runs / run_id,
            rate=spec.rate or self.route_rate,
            delivered=delivered,
            **common,
        )

    def _branches(self, spec: MachineSpec) -> list[str]:
        """The branches a machine tracks, by id: its ring, or its route's, in order."""
        if spec.kind == "ring":
            return [self.source.lattice.layout(spec.target).id]
        return list(dict.fromkeys(s.branch for s in self.source.routes[spec.target].segments))

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
        """Kick (a pinger at the ring's start, rad), inject a fresh bunch, or ramp (ADR 0013)."""
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
        elif action == "ramp":
            try:
                instance.run.ramp()
            except RuntimeError as error:  # ramping already
                raise Conflict(f"{machine_id}: {error}") from None
        else:
            raise ValueError("the beam takes kick, inject or ramp")
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

    # -- proposals (ADR 0010) -------------------------------------------------------------------

    def propose(self, user: User, machine_id: str, request: Mapping[str, Any]) -> dict[str, Any]:
        """An operation computed on the machine as it is set and as its BPMs read: a proposal, written to
        nothing. Agents may propose on shared machines too; a person applies there. The operation is the
        catalog's (ADR 0017): its parameters are validated, and what verifies it is measured beforehand."""
        from eicvibe.operations.catalog import operation
        from eicvibe.operations.model import MachineModel

        instance = self._instance(machine_id)
        self._may(user, instance, Role.PHYSICIST, agent=True)
        spec = instance.spec
        declared = operation(request.get("operation"))
        if spec.kind not in declared.machines:
            kinds = " and ".join(f"{kind}s" for kind in sorted(declared.machines))
            raise ValueError(f"{declared.label} is for {kinds} so far, not a {spec.kind}")
        parameters = declared.parameters.model_validate(request)
        request = {"operation": declared.name, **parameters.model_dump(), "reason": request.get("reason", "")}
        settings = self._machine_settings(spec.mode)
        context: dict[str, Any] = {"turn": instance.position}
        if declared.verified_by == "orbit" and instance.state == "running":
            try:
                context["orbit_before"] = self._measured_orbit(instance)  # to verify against
            except Conflict:
                pass
        # The model as the ring is tracked: 6D, with its RF (a horizontal kick that changes the path
        # changes the energy), unless the RF holds no bunch (the run's warnings).
        effects = frozenset({Effect.LONGITUDINAL})
        if instance.run is not None:
            effects = frozenset() if instance.run.warnings else instance.run.options.effects
        options = Options(fidelity=Fidelity(spec.fidelity), effects=effects)
        measured = _Measured(lambda: context.get("orbit_before") or self._measured_orbit(instance))
        with MachineModel(settings, spec.target, self.live(machine_id), options=options) as model:
            try:
                proposal = declared.propose(model, parameters, measured)
            except ValueError, KeyError, Conflict, Forbidden:
                raise
            except Exception as error:  # noqa: BLE001 - the engines' own errors, whichever engine
                _log.exception("%s on %s failed in the model", declared.name, machine_id)
                raise ValueError(
                    f"{declared.label} could not be computed in the model: {type(error).__name__}: {error}"
                ) from error
            if declared.verified_by == "tunes":
                context["tunes_before"] = self.tunes(machine_id)
        ident = uuid.uuid4().hex[:10]
        stored = proposal.model_dump(mode="json") | {"context": context}
        self.store.add_proposal(ident, machine_id, user, declared.name, request, stored, instance.position)
        self.store.act(
            user.name, "propose", machine_id, request["reason"], proposal=ident, operation=declared.name,
            agent=user.agent, summary=proposal.summary,
        )  # fmt: skip
        return self.store.proposal(ident)

    def proposals(self, machine_id: str, waiting: bool = False) -> list[dict[str, Any]]:
        self._instance(machine_id)
        return self.store.proposals(machine_id, waiting=waiting)

    def proposal(self, ident: str) -> dict[str, Any]:
        return self.store.proposal(ident)

    # -- what the service keeps besides its logs ------------------------------------------------

    def history(self, user: User) -> dict[str, Any]:
        """The runs' folders (their BPM histories, on disk), newest first, with their size and whether a
        running machine uses them; and how many proposals each state has. For administrators to clear."""
        self._require(user, Role.ADMIN)
        in_use = self._runs_in_use()
        runs = []
        for folder in self.runs.iterdir() if self.runs.is_dir() else []:
            if not folder.is_dir():
                continue
            try:
                info = json.loads((folder / "run.json").read_text(encoding="utf-8"))
            except OSError, ValueError:
                info = {}  # a run that failed before it wrote what it was
            runs.append(
                {
                    "id": folder.name,
                    "machine": info.get("machine"),
                    "engine": info.get("engine"),
                    "started": info.get("start"),
                    "bytes": _size(folder),
                    "in_use": folder.name in in_use,
                }
            )
        runs.sort(key=lambda r: r["started"] or "", reverse=True)
        return {"runs": runs, "proposals": self.store.proposal_counts()}

    def _runs_in_use(self) -> set[str]:
        return {i.run_id for i in self._instances.values() if i.run_id is not None}

    def delete_runs(self, user: User, runs: list[str] | None = None, reason: str = "") -> dict[str, Any]:
        """Delete runs' folders, their BPM histories: these, or (None) every run no running machine uses.

        A running machine's run is refused. The logs keep what the runs did, and this deletion.
        """
        self._require(user, Role.ADMIN)
        in_use = self._runs_in_use()
        known = {f.name for f in self.runs.iterdir() if f.is_dir()} if self.runs.is_dir() else set()
        if runs is None:
            chosen = sorted(known - in_use)
        else:
            chosen = list(dict.fromkeys(runs))
            if unknown := [r for r in chosen if r not in known]:
                raise KeyError(f"no run {unknown[0]}")
            if busy := [r for r in chosen if r in in_use]:
                raise Conflict(f"{busy[0]} is a running machine's run: stop the machine first")
        freed = 0
        for run in chosen:
            freed += _size(self.runs / run)
            shutil.rmtree(self.runs / run, ignore_errors=True)
        self.store.act(user.name, "delete runs", None, reason, runs=chosen, bytes=freed)
        return {"deleted": chosen, "bytes": freed}

    def delete_proposals(self, user: User, waiting: bool = False, reason: str = "") -> dict[str, Any]:
        """Delete every machine's finished proposals (verified, discarded, stale), and with ``waiting`` those
        still waiting for a person too (proposed, applied). The logs keep what was proposed and done."""
        self._require(user, Role.ADMIN)
        states = ["verified", "discarded", "stale"] + (["proposed", "applied"] if waiting else [])
        count = self.store.delete_proposals(states)
        self.store.act(user.name, "delete proposals", None, reason, states=states, count=count)
        return {"deleted": count}

    def apply(self, user: User, ident: str, reason: str = "") -> dict[str, Any]:
        """Write a proposal's changes, atomically, if the machine is still as it was proposed for.

        On a shared machine a person applies it (an agent's token cannot); in a sandbox its owner or
        the owner's agent.
        """
        from eicvibe.operations.proposal import Proposal

        row = self.store.proposal(ident)
        instance = self._instance(row["machine"])
        self._may(user, instance, Role.PHYSICIST)
        if row["state"] != "proposed":
            raise Conflict(f"proposal {ident} is {row['state']}")
        proposal = Proposal.model_validate(row["proposal"])
        settings = self._machine_settings(instance.spec.mode)
        live = self.live(row["machine"])
        moved = [
            c.channel
            for c in proposal.changes
            if abs(settings.value(c.channel, live) - c.before) > 1e-9 * max(1.0, abs(c.before))
        ]
        if moved:
            self.store.update_proposal(ident, state="stale", decided=now(), decided_by=user.name)
            self.store.act(
                user.name, "proposal stale", row["machine"], reason, proposal=ident, channels=moved
            )
            raise Conflict(
                f"the machine changed since proposal {ident} ({', '.join(moved[:3])}): propose again"
            )
        why = f"proposal {ident}: {proposal.summary}" + (f"; {reason}" if reason else "")
        result = self.write(user, row["machine"], proposal.values(), why, correlation=f"proposal-{ident}")
        if not result["applied"]:
            refused = [w for w in result["writes"] if w["refusal"]]
            raise Conflict(f"refused: {refused[0]['channel']}: {refused[0]['refusal']}")
        self.store.update_proposal(
            ident, state="applied", decided=now(), decided_by=user.name,
            applied_turn=instance.position, correlation=result["correlation"],
        )  # fmt: skip
        self.store.act(user.name, "apply proposal", row["machine"], reason, proposal=ident, agent=user.agent)
        return self.store.proposal(ident)

    def discard(self, user: User, ident: str, reason: str = "") -> dict[str, Any]:
        row = self.store.proposal(ident)
        self._may(user, self._instance(row["machine"]), Role.PHYSICIST, agent=True)
        if row["state"] != "proposed":
            raise Conflict(f"proposal {ident} is {row['state']}")
        self.store.update_proposal(ident, state="discarded", decided=now(), decided_by=user.name)
        self.store.act(user.name, "discard proposal", row["machine"], reason, proposal=ident)
        return self.store.proposal(ident)

    def verify(self, user: User, ident: str) -> dict[str, Any]:
        """What the machine shows since a proposal was applied, against what the proposal predicted.

        Once its settings have ramped and the BPMs (or the tune monitor) have measured after the
        ramp; until then the answer says when. Orbits are compared as changes (the BPMs' hidden
        offsets cancel), tunes as changes too (a tune monitor may see a synchrotron sideband,
        which moves with the tune).
        """
        row = self.store.proposal(ident)
        instance = self._instance(row["machine"])
        self._may(user, instance, Role.PHYSICIST, agent=True)
        if row["state"] not in ("applied", "verified"):
            raise Conflict(f"proposal {ident} is {row['state']}: only an applied one is verified")
        if instance.state != "running":
            raise Conflict(f"{row['machine']} is {instance.state}: start it to measure")
        from eicvibe.operations.catalog import operation

        settled = row["applied_turn"] + instance.spec.ramp_turns
        proposal, context = row["proposal"], row["proposal"].get("context", {})
        if operation(row["operation"]).verified_by == "tunes":
            reading = self.tunes(row["machine"])
            ready = (
                reading is not None and reading["settled"] and reading["turn"] - reading["turns"] >= settled
            )
            if not ready:
                return row | {"verification": {"ready": False, "until": settled + instance.run.tune_turns}}
            verification = _verify_tunes(proposal, context.get("tunes_before"), reading)
        else:
            samples = self._orbit_samples(instance)
            after = [i for i, first in enumerate(samples.turn) if first >= settled]
            if not after:
                return row | {"verification": {"ready": False, "until": settled}}
            verification = _verify_orbit(proposal, context.get("orbit_before"), samples, after[-1])
        self.store.update_proposal(ident, state="verified", verification=verification)
        self.store.act(
            user.name, "verify proposal", row["machine"], "", proposal=ident, ok=verification["ok"]
        )
        return self.store.proposal(ident)

    def ramp(self, machine_id: str, samples: int = 241) -> dict[str, Any]:
        """A ring's ramp (ADR 0013): its profile and its programs (sampled in time over the ramp and the
        flattop), where the ring is in it, what keeps it from ramping now, and the ramps of its run."""
        from eicvibe.domain.machine.ramps import ramp_period
        from eicvibe.engines.convert import reference

        instance = self._instance(machine_id)
        spec = instance.spec
        profile = self.source.ramps.get(spec.target) if spec.kind == "ring" else None
        if profile is None:
            raise KeyError(f"{machine_id} has no ramp profile")
        lattice = self.source.lattice
        period = ramp_period(profile, lattice)
        mass = reference("va", lattice, lattice.layout(spec.target).id)["mass"]
        flattop, extraction = profile.ramp_turns(period), profile.extraction_turn(period)
        rows = [profile.at(extraction * k / (samples - 1), period, mass) for k in range(samples)]
        run = instance.run if instance.state == "running" else None
        if run is None:
            refusal = f"{machine_id} is {instance.state}: start it to ramp"
        else:
            refusal = run.ramp_refusal
        return {
            "machine": machine_id,
            "ramp": profile.name,
            "branch": profile.branch,
            "energy": profile.energy.model_dump(),
            "porch": profile.porch,
            "flattop": profile.flattop,
            "cycle_time": profile.cycle,
            "period": period,
            "flattop_turn": flattop,
            "extraction_turn": extraction,
            "source": profile.source,
            "program": {
                "time": [extraction * k / (samples - 1) * period * 1e3 for k in range(samples)],
                "energy": [r["energy"] / 1e9 for r in rows],
                "voltage": [r["voltage"] / 1e6 for r in rows],
                "phase": [math.degrees(r["phase"]) for r in rows],
            },
            "cycle": run.cycle if run is not None else None,
            "engine": run.engine if run is not None else spec.engine,
            "refusal": refusal,
            "ramps": self._ramps_done(instance),
        }

    def _ramps_done(self, instance: Instance, keep: int = 20) -> list[dict[str, Any]]:
        """The ramps of a machine's latest run, newest first, from its events: when each began, reached
        the flattop and extracted, the bunch it extracted (one under way has no extraction yet), and the
        capture of its BPMs' readings, every turn, saved at extraction."""
        from eicvibe.virtual import BpmHistory

        if instance.run_id is None or not (self.runs / instance.run_id).is_dir():
            return []
        done: list[dict[str, Any]] = []
        injected: int | None = None
        for event in BpmHistory.open(self.runs / instance.run_id).events():
            kind = event["kind"]
            if kind == "inject":
                injected = event.get("particles")
            elif kind == "ramp":
                done.append({"turn": event["turn"], "wall": event.get("wall"), "injected": injected})
            elif kind == "flattop" and done:
                done[-1]["flattop"] = event["turn"]
            elif kind == "extract" and done:
                detail = {k: event.get(k) for k in ("energy", "particles", "emittance_x", "emittance_y")}
                detail |= {k: event.get(k) for k in ("sigma_delta", "delta", "sigma_zeta")}
                done[-1] |= {"extracted": event["turn"], **detail}
            elif kind == "capture" and event.get("of") == "ramp":  # dated at the ramp's start
                for ramp in done:
                    if ramp["turn"] == event["turn"]:
                        ramp |= {"capture": event.get("id"), "capture_turns": event.get("turns")}
        return done[::-1][:keep]

    def tunes(self, machine_id: str) -> dict[str, Any] | None:
        """The ring's tune monitor: fractional tunes (0 to 0.5), the turn, whether settled; None yet."""
        instance = self._instance(machine_id)
        if instance.spec.kind != "ring" or instance.run is None:
            return None
        return getattr(instance.run, "tunes", None)

    def _orbit_samples(self, instance: Instance):
        from eicvibe.virtual import BpmHistory

        if instance.run is None:
            raise Conflict(f"{instance.spec.id} has not run: start it to measure its orbit")
        return BpmHistory.open(self.runs / instance.run_id).orbit()

    def _measured_orbit(self, instance: Instance) -> dict[str, dict[str, float]]:
        """The latest slow orbit sample, by BPM name, mm."""
        samples = self._orbit_samples(instance)
        if len(samples.turn) == 0:
            raise Conflict(f"{instance.spec.id} has no orbit measured yet: wait a second after its start")
        return {
            p: {n: float(v) for n, v in zip(samples.bpms, getattr(samples, p)[-1], strict=True) if v == v}
            for p in ("x", "y")
        }

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
        if user.agent and role >= Role.OPERATOR:
            raise Forbidden(f"an agent's token cannot do what needs an {role.label}; a person does")

    def _may(self, user: User, instance: Instance, shared_role: Role, *, agent: bool = False) -> None:
        """Shared machines need ``shared_role``; a sandbox, its owner (or an administrator).

        An agent (its user's agent token) acts only in its user's sandboxes: on a shared machine it
        may only do what ``agent`` allows (propose, verify), and a person applies (ADR 0010).
        """
        if instance.spec.shared:
            if user.agent and not agent:
                raise Forbidden(f"{user.name}'s agent only proposes on a shared machine; a person applies")
            self._require(user, shared_role)
            return
        self._require(user, Role.PHYSICIST if user.name == instance.spec.owner else Role.ADMIN)

    def _quota(self, user: User) -> None:
        own = [i for i in self._instances.values() if not i.spec.shared and i.spec.owner == user.name]
        if len(own) >= self.sandbox_quota and not user.can(Role.ADMIN):
            raise Forbidden(f"{user.name} has {len(own)} sandboxes, the most allowed; remove one first")


def _seeded(errors: ErrorSpec, current: int | None) -> ErrorSpec:
    """``errors`` with a seed: its own, else the machine's current one, else one drawn now."""
    if errors.seed is not None:
        return errors
    return errors.replaced(
        seed=current if current is not None else int(np.random.default_rng().integers(1, 2**31))
    )


def _described(errors: ErrorSpec) -> dict[str, Any]:
    """A machine's errors for the log: what kinds and how large, never the seed that draws them."""
    return errors.model_dump(mode="json", exclude={"seed": True, "bpms": {"seed"}})


def _spec_fields(stored: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in stored.items() if k in MachineSpec.model_fields}


def _size(folder: Path) -> int:
    """Bytes in a folder's files; a file removed meanwhile (a run's buffer moving on) counts nothing."""
    total = 0
    for path in folder.rglob("*"):
        try:
            total += path.stat().st_size if path.is_file() else 0
        except OSError:
            pass
    return total


@dataclass(frozen=True)
class _Measured:
    """What a machine measured, for an operation (the catalog's ``Measured``): its slow orbit, read when
    asked for, by plane and BPM, mm."""

    read: Callable[[], dict[str, dict[str, float]]]

    def orbit(self, plane: str) -> dict[str, float]:
        return {name: value * 1e-3 for name, value in self.read()[plane].items()}


def _verify_orbit(proposal: dict, before: dict | None, samples, index: int) -> dict[str, Any]:
    """The orbit change the BPMs measured against the change predicted, at every BPM, mm."""
    predicted = proposal["predicted"]
    plane = predicted["plane"]
    names = list(samples.bpms)
    after = {n: float(v) for n, v in zip(names, getattr(samples, plane)[index], strict=True)}
    if before is None:
        return {"ready": True, "ok": False, "note": "no orbit was measured when it was proposed"}
    if "change" in predicted:  # the change at the BPMs (a bump)
        expected = dict(zip(predicted["bpms"], predicted["change"], strict=True))
    else:  # the orbit before and after (a correction)
        expected = {
            n: a - b
            for n, a, b in zip(predicted["bpms"], predicted["after"], predicted["before"], strict=True)
            if a is not None and b is not None
        }
    measured = {n: after[n] - before[plane][n] for n in expected if n in after and n in before[plane]}
    common = [n for n in expected if n in measured and measured[n] == measured[n]]
    if not common:
        return {"ready": True, "ok": False, "note": "no BPM read before and after"}
    m = np.array([measured[n] for n in common])
    e = np.array([expected[n] for n in common])
    difference = float(np.sqrt(np.mean((m - e) ** 2)))
    size = float(np.sqrt(np.mean(e**2)))
    limit = 0.05 * size + 0.005  # mm: the BPMs' gain errors (1 %) and noise, with margin
    return {
        "ready": True,
        "ok": difference <= limit,
        "plane": plane,
        "rms_difference_mm": difference,
        "rms_predicted_change_mm": size,
        "limit_mm": limit,
        "sample_turn": int(samples.turn[index]),
        "bpms": common,
        "measured_change_mm": m.tolist(),
        "predicted_change_mm": e.tolist(),
        "note": "the orbit change the BPMs measured against the change predicted (offsets cancel)",
    }


def _verify_tunes(proposal: dict, before: dict | None, reading: dict) -> dict[str, Any]:
    """The tune monitor's change against the tune change predicted.

    A pickup sees fractional tunes folded into 0 to 0.5, so a tune above the half integer reads as
    1 - q and moves the other way; the model's tune says which side (as an operator knows).
    """
    if before is None:
        return {"ready": True, "ok": False, "note": "the tune monitor had no reading when it was proposed"}
    (qx0, qy0), (qx1, qy1) = proposal["predicted"]["tunes"]
    out: dict[str, Any] = {"ready": True, "turn": reading["turn"], "planes": {}}
    ok = True
    for plane, q0, q1 in (("x", qx0, qx1), ("y", qy0, qy1)):
        side = -1.0 if (q0 % 1.0) > 0.5 else 1.0
        measured = side * (reading[plane] - before[plane])
        expected = q1 - q0
        tolerance = max(5e-4, 0.1 * abs(expected))
        good = abs(measured - expected) <= tolerance
        ok &= good
        out["planes"][plane] = {
            "measured_change": measured, "predicted_change": expected, "tolerance": tolerance, "ok": good,
            "reading_before": before[plane], "reading_after": reading[plane],
        }  # fmt: skip
    out["ok"] = bool(ok)
    out["note"] = "the tune monitor's change against the change predicted (fractional, folded)"
    return out
