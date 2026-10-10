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
from eicvibe.engines.contracts import Fidelity, Options
from eicvibe.engines.worker import EngineWorker, RemoteSession

from .history import BpmHistory, RunInfo
from .matching import twiss_bunch
from .ring import Beam, BpmModel, VirtualBpms, default_engine


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
        bpms = route_bpms(lattice, route, registry)
        if not bpms:
            raise ValueError(f"route {route.name} has no BPMs to record")
        self._observe = [[b[1] for b in bpms if b[0] == k] for k in range(len(route.segments))]
        self.bpm_names = [b[2] for b in bpms]
        self.bpm_s = [b[3] for b in bpms]
        self._bpms = VirtualBpms(self.bpm, len(bpms), self.beam.particles, self.beam.charge)
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
                        self.lattice, segment.branch, self.options, self.values
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
            },
            resolution=self.bpm.resolution,
            retention_bytes=self.retention_bytes,
            values={f"{o} {p}": v for (o, p), v in self.values.items()},
        )
        self._writer = BpmHistory.create(self.directory, info)
        (self.directory / "hidden.json").write_text(
            json.dumps(self._bpms.hidden(self.bpm_names)), encoding="utf-8"
        )
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
            by_branch: dict[str, dict] = {}
            for (occurrence, parameter), value in values.items():
                branch = self.lattice.placement(occurrence).branch_id
                by_branch.setdefault(branch, {})[(occurrence, parameter)] = value
            for branch, changes in by_branch.items():
                if branch in self._sessions:
                    self._sessions[branch].update(changes)
            self.values.update(values)
            self._writer.event(
                self._shot, "settings", values={f"{o} {p}": v for (o, p), v in values.items()}, **note
            )

        self._commands.put(update)

    # -- shots -------------------------------------------------------------------------

    def _start_optics(self):
        """The optics and orbit where the route starts, and their origin: where its optics start."""
        return self._worker.route_start(self.lattice, self.route, self.options, self.values)

    def _shoot(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """One shot along the route: the centroid at every BPM, and how many particles reached it."""
        j, t = self.jitter, self._start
        offset = self._jitter.standard_normal(5) * [j.x, j.px, j.y, j.py, j.delta]
        centre = np.array([t.x, t.px, t.y, t.py, t.zeta, t.delta]) + np.insert(offset, 4, 0.0)
        particles = twiss_bunch(self._bunches, self.beam, t, centre)
        xs, ys, counts = [], [], []
        for segment, observe in zip(self.route.segments, self._observe, strict=True):
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
