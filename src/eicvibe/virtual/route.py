"""A route the engine keeps sending shots along, while its BPMs record every shot (ADR 0007).

A transfer line, or a path through several branches (a :class:`~eicvibe.domain.lattice.Route`),
is passed once by every bunch: its BPMs read trajectories, shot after shot, where
a ring's read turns. :class:`RouteRun` injects a bunch matched to the optics at
the route's start (the route's optics as the viewer shows them: its first
line's initial conditions, orbit included, or the machine's optics where it
starts part-way), its centroid moved by the shot-to-shot jitter a real
injector has, and tracks it segment by segment (in its own worker), carrying
the coordinates across every fork as they are, offsets where a fork lands
included: an injection's offset is part of the trajectory. Each shot's
readings at the route's BPMs become one "turn" of the run's history, whose unit
is the shot. Settings change between shots, as on the real machine.
"""

import json
import queue
import threading
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import Field

from eicvibe.diagnostics.bpm import NO_BEAM
from eicvibe.domain.lattice import ElementKind, Lattice, Route, route_placements, route_problems
from eicvibe.domain.lattice._base import DomainModel
from eicvibe.domain.machine import MachineRegistry
from eicvibe.domain.machine.power import design_value
from eicvibe.engines.contracts import Fidelity, Options
from eicvibe.engines.convert import reference
from eicvibe.engines.worker import EngineWorker, RemoteSession
from eicvibe.store.beams import DeliveredBeam

from .errors import BpmModel, MachineErrors, VirtualBpms
from .history import BpmHistory, RunInfo
from .matching import twiss_bunch
from .ring import Beam, default_engine


class Jitter(DomainModel):
    """Shot-to-shot jitter of the injected bunch's centroid, rms."""

    x: float = Field(5e-5, ge=0, description="m")
    px: float = Field(5e-6, ge=0)
    y: float = Field(5e-5, ge=0, description="m")
    py: float = Field(5e-6, ge=0)
    delta: float = Field(1e-4, ge=0)
    seed: int = 3


def route_bpms(
    lattice: Lattice, route: Route, registry: MachineRegistry | None = None
) -> list[tuple[int, str, str, float]]:
    """The BPMs along a route as ``(segment, occurrence, name, s along the route at its exit)``."""
    names = {}
    if registry is not None:
        for device in registry.devices:
            if device.type.value == "bpm":
                names.update({occurrence: device.name for occurrence in device.occurrences})
    out = []
    for segment, placed, s in route_placements(lattice, route):
        if placed.kind is ElementKind.INSTRUMENT and (not names or placed.id in names):
            out.append((segment, placed.id, names.get(placed.id, placed.name), s + placed.length))
    return out


class RouteRun:
    """Shots along a route, tracked in an engine worker, their BPM trajectories recorded into a history."""

    def __init__(
        self,
        lattice: Lattice,
        route: Route,
        directory: str | Path,
        *,
        engine: str | None = None,
        fidelity: Fidelity = Fidelity.STANDARD,
        beam: Beam | None = None,
        jitter: Jitter | None = None,
        bpm: BpmModel | None = None,
        rate: float = 2.0,
        chunk: int = 4,
        values: Mapping[tuple[str, str], float] | None = None,
        registry: MachineRegistry | None = None,
        retention_bytes: int = 1 << 30,
        run_id: str | None = None,
        machine: str | None = None,
        shots: int | None = None,
        errors: MachineErrors | None = None,
        delivered: DeliveredBeam | None = None,
    ) -> None:
        if problems := route_problems(lattice, route):
            raise ValueError(f"route {route.name} is not one pass: {'; '.join(problems)}")
        self.lattice = lattice
        self.route = route
        self.directory = Path(directory)
        self.engine = engine or default_engine()
        self.options = Options(fidelity=fidelity)
        self.beam = beam or Beam()
        self.jitter = jitter or Jitter()
        self.bpm = bpm or BpmModel()
        self.rate = rate
        """Shots per second, at most."""
        self.chunk = chunk
        self.values = dict(values or {})
        self.retention_bytes = retention_bytes
        self.run_id = run_id
        """The run's id; by default the branch's or route's name and the time."""
        self.machine = machine
        self.shots = shots
        """Stop by itself after this many shots; None: never."""
        self.errors = errors if errors is not None and not errors.empty else None
        """The machine's own errors (``errors.py``): in the engines' values, never in :attr:`values`."""
        self._fluctuations = self.errors.fluctuations(1.0 / rate) if self.errors else None
        self.warnings: list[str] = list(errors.problems) if errors is not None else []
        """What the run found wrong: its errors' rows that chose nothing."""
        self.delivered = delivered
        """The beam an injector code delivered where the route starts (ADR 0012): each shot is a sample of
        its particles, with the shot-to-shot jitter, rather than a Gaussian matched to the optics there."""
        self._pool: np.ndarray | None = None
        if delivered is not None:
            first = route.segments[0]
            layout = lattice.layout(first.branch)
            if (
                lattice.branch(first.branch).name != delivered.branch
                or layout.placements[0].id != first.start
            ):
                raise ValueError(
                    f"route {route.name} does not start where the beam is delivered ({delivered.branch})"
                )
            self._pool = delivered.about(reference("route", lattice, first.branch)["pc"])
        bpms = route_bpms(lattice, route, registry)
        if not bpms:
            raise ValueError(f"route {route.name} has no BPMs to record")
        self._observe = [[b[1] for b in bpms if b[0] == k] for k in range(len(route.segments))]
        self.bpm_names = [b[2] for b in bpms]
        self.bpm_s = [b[3] for b in bpms]
        charge = delivered.charge if delivered is not None else self.beam.charge
        self._bpms = VirtualBpms(self.bpm, len(bpms), self.beam.particles, charge, self.bpm_names)
        self._accelerate = _accelerations(lattice, route)
        self._bunches = np.random.default_rng(self.beam.seed)
        self._jitter = np.random.default_rng(self.jitter.seed)
        self._commands: queue.Queue[Callable[[], None]] = queue.Queue()
        self._stopping = threading.Event()
        self._stopped = threading.Event()
        self._thread: threading.Thread | None = None
        self._worker: EngineWorker | None = None
        self._sessions: dict[str, RemoteSession] = {}
        self._writer: BpmHistory | None = None
        self._start = None
        self._origin = ""
        self._shot = 0
        self.error: BaseException | None = None

    # -- life --------------------------------------------------------------------------

    def start(self) -> RouteRun:
        """Open a session per branch in one engine worker and start shooting in the background."""
        self._worker = EngineWorker(self.engine)
        try:
            for segment in self.route.segments:
                if segment.branch not in self._sessions:
                    self._sessions[segment.branch] = self._worker.open_session(
                        self.lattice, segment.branch, self.options, self._engine_values()
                    )
            self._start, self._origin = self._start_optics()
        except BaseException:
            self._worker.close()
            raise
        info = RunInfo(
            id=self.run_id or f"{self.route.name}-{datetime.now(UTC):%Y%m%dT%H%M%S}",
            machine=self.machine,
            kind="route",
            unit="shot",
            lattice=self.lattice.name,
            lattice_digest=self.lattice.digest,
            branch=self.route.name,
            branch_id=f"route:{self.route.name}",
            engine=self.engine,
            fidelity=self.options.fidelity.value,
            effects=[],
            start=datetime.now(UTC),
            revolution_period=1.0 / self.rate,
            chunk=self.chunk,
            bpms=self.bpm_names,
            occurrences=[o for observe in self._observe for o in observe],
            s=self.bpm_s,
            beam=self.beam.model_dump()
            | {
                "jitter": self.jitter.model_dump(),
                "matched": {"to": self._origin, **self._start.model_dump()},
            }
            | (
                {
                    "delivered": {
                        "branch": self.delivered.branch,
                        "charge": self.delivered.charge,
                        "particles": len(self.delivered.particles),
                        "source": self.delivered.source,
                    }
                }
                if self.delivered is not None
                else {}
            ),
            resolution=self.bpm.resolution,
            retention_bytes=self.retention_bytes,
            values={f"{o} {p}": v for (o, p), v in self.values.items()},
        )
        self._writer = BpmHistory.create(self.directory, info)
        hidden = self._bpms.hidden(self.bpm_names) | ({"errors": self.errors.hidden()} if self.errors else {})
        (self.directory / "hidden.json").write_text(json.dumps(hidden), encoding="utf-8")
        self._writer.event(0, "start", engine=self.engine, route=list(self.route.branches))
        self._thread = threading.Thread(target=self._loop, name=f"route run {info.id}", daemon=True)
        self._thread.start()
        return self

    def stop(self, timeout: float | None = None) -> None:
        self._stopping.set()
        if self._thread is not None:
            self._thread.join(timeout)
        for session in self._sessions.values():
            session.close()
        if self._worker is not None:
            self._worker.close()

    def __enter__(self) -> RouteRun:
        return self if self._thread else self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    @property
    def shot(self) -> int:
        """Shots recorded so far."""
        return self._shot

    @property
    def done(self) -> bool:
        return self._stopped.is_set()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def history(self) -> BpmHistory:
        return BpmHistory.open(self.directory)

    def wait(self, shot: int | None = None, timeout: float = 600.0) -> None:
        """Return once ``shot`` shots are recorded (by default the run's own end)."""
        shot = self.shots if shot is None else shot
        if shot is None:
            raise ValueError("this run has no end: give the shot to wait for")
        deadline = time.monotonic() + timeout
        while self._shot < shot:
            if self.error is not None:
                raise RuntimeError(f"the run stopped: {self.error}") from self.error
            if self._stopped.is_set():
                raise RuntimeError(f"the run stopped at shot {self._shot}")
            if time.monotonic() > deadline:
                raise TimeoutError(f"shot {self._shot} of {shot} after {timeout} s")
            time.sleep(0.02)

    def update(self, values: Mapping[tuple[str, str], float], **note: Any) -> None:
        """Change settings (occurrence, parameter) between shots; ``note`` goes into the event."""
        values = dict(values)

        def update() -> None:
            self._to_sessions(self._engine_values(values))
            self.values.update(values)
            self._writer.event(
                self._shot, "settings", values={f"{o} {p}": v for (o, p), v in values.items()}, **note
            )

        self._commands.put(update)

    def _to_sessions(self, values: Mapping[tuple[str, str], float]) -> None:
        """Values into the sessions of the branches they are on."""
        by_branch: dict[str, dict] = {}
        for (occurrence, parameter), value in values.items():
            branch = self.lattice.placement(occurrence).branch_id
            by_branch.setdefault(branch, {})[(occurrence, parameter)] = value
        for branch, changes in by_branch.items():
            if branch in self._sessions:
                self._sessions[branch].update(changes)

    def _engine_values(self, values: Mapping[tuple[str, str], float] | None = None) -> dict:
        """Settings as the engines take them, with the machine's errors: ``values`` (an update), or
        the run's, with every error (sessions opened)."""
        if self.errors is None:
            return dict(self.values if values is None else values)
        if values is None:
            return self.errors.engine_values(self.values, complete=True)
        return self.errors.engine_values(values)

    def _supplies(self, shot: int) -> None:
        """The power supplies as this shot finds them: their settings, off by their noise, pulse to pulse."""
        if self._fluctuations is None:
            return
        values = {}
        for supply, deviation in self._fluctuations.turns(shot, 1):
            for key in supply.keys:
                setting = self.values.get(key, design_value(self.lattice.placement(key[0]), key[1]))
                values[key] = self._engine_values({key: setting})[key] * (1.0 + float(deviation[0]))
        self._to_sessions(values)

    # -- shots -------------------------------------------------------------------------

    def _start_optics(self):
        """The optics and orbit where the route starts, and their origin: where its optics start."""
        return self._worker.route_start(self.lattice, self.route, self.options, self.values)

    def _shoot(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """One shot along the route: the centroid at every BPM, and how many particles reached it."""
        j, t = self.jitter, self._start
        offset = self._jitter.standard_normal(5) * [j.x, j.px, j.y, j.py, j.delta]
        if self._pool is not None:  # a sample of the delivered beam, moved by the jitter
            count = min(self.beam.particles, len(self._pool))
            particles = self._pool[self._bunches.choice(len(self._pool), size=count, replace=False)]
            particles = particles + np.insert(offset, 4, 0.0)
        else:
            centre = np.array([t.x, t.px, t.y, t.py, t.zeta, t.delta]) + np.insert(offset, 4, 0.0)
            particles = twiss_bunch(self._bunches, self.beam, t, centre)
        xs, ys, counts = [], [], []
        for k, (segment, observe) in enumerate(zip(self.route.segments, self._observe, strict=True)):
            if len(particles) == 0:
                xs += [np.nan] * len(observe)
                ys += [np.nan] * len(observe)
                counts += [0] * len(observe)
                continue
            track = self._sessions[segment.branch].track_beam(
                particles, 1, observe, self.options, start=segment.start, end=segment.end
            )
            xs += list(np.asarray(track.x)[:, 0])
            ys += list(np.asarray(track.y)[:, 0])
            counts += list(np.asarray(track.count)[:, 0])
            particles = np.asarray(track.final)[~np.asarray(track.lost)]
            if (ratio := self._accelerate.get(k)) is not None:
                particles = accelerated(particles, ratio)
        return np.array(xs), np.array(ys), np.array(counts)

    def _loop(self) -> None:
        state = "stopped"
        try:
            while not self._stopping.is_set():
                if self.shots is not None and self._shot >= self.shots:
                    state = "done"
                    break
                started = time.monotonic()
                first = self._shot
                n = self.chunk if self.shots is None else min(self.chunk, self.shots - first)
                shots = []
                for _ in range(n):
                    while not self._commands.empty():
                        self._commands.get_nowait()()
                    self._supplies(first + len(shots))
                    shots.append(self._shoot())
                    if self._stopping.is_set():
                        break
                x, y, count = (np.column_stack([s[k] for s in shots]) for k in range(3))
                readings = self._bpms.read(x, y, count)
                self._writer.append(first, *readings, wall=time.time())
                self._shot += len(shots)
                lost = int(np.count_nonzero(readings[3][-1] == NO_BEAM))
                if lost:
                    self._writer.event(self._shot, "loss", shots=lost, at=self.bpm_names[-1])
                self._writer.set_state("running", self._shot)
                if self.shots is not None and self._shot >= self.shots:
                    state = "done"
                    break
                time.sleep(max(0.0, len(shots) / self.rate - (time.monotonic() - started)))
        except BaseException as error:  # noqa: BLE001 - kept for whoever waits on the run
            self.error = error
            state = "failed"
            if self._writer is not None:
                self._writer.event(self._shot, "error", message=f"{type(error).__name__}: {error}")
        finally:
            if self._writer is not None:
                self._writer.set_state(state, self._shot)
            self._stopped.set()


def _accelerations(lattice: Lattice, route: Route) -> dict[int, float]:
    """Where a route leaves a ring the beam is accelerated in: segment index -> p before / p after."""
    out = {}
    for k, (before, after) in enumerate(zip(route.segments, route.segments[1:], strict=False)):
        if before.branch == after.branch or before.branch not in route.accelerated:
            continue
        a, b = (lattice.initial_conditions(s.branch) for s in (before, after))
        pa = a.reference.pc_ref if a and a.reference else None
        pb = b.reference.pc_ref if b and b.reference else None
        if pa and pb:
            out[k] = pa / pb
    return out


def accelerated(particles: np.ndarray, ratio: float) -> np.ndarray:
    """A bunch (rows of x, px, y, py, zeta, delta) after an acceleration from momentum p0 to p1 =
    p0 / ``ratio``: every particle gains the same momentum, so the angles (px, py: transverse momenta
    over the reference momentum) and the momentum deviation shrink by p0 / p1 (adiabatic damping);
    positions stay."""
    out = np.array(particles, dtype=float, copy=True)
    out[:, 1] *= ratio
    out[:, 3] *= ratio
    out[:, 5] *= ratio
    return out
