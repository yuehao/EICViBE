"""A ring the engine keeps tracking, turn after turn, while its BPMs record every turn (ADR 0007).

:class:`RingRun` holds a bunch of macroparticles in a ring and asks an engine
(in its own worker process) to carry it on, a chunk of turns at a time. The
engine returns the bunch's centroid at every BPM on every turn; the run turns
those into readings as BPMs give them, in mm, with each BPM's hidden offset
and gain error and its resolution noise, and appends them to the run's
:class:`~.history.BpmHistory`. Anyone can read the history, while the run goes
on, as :class:`~eicvibe.diagnostics.TurnByTurn` acquisitions, the same type
the readers of real BPM files give.

Between chunks the run takes commands, as a machine takes them from its
operators: kick the beam (a pinger), change settings (the session updates in
place), inject a fresh bunch, ramp. Each is logged as an event with its turn. A
bunch that is lost leaves the BPMs reading nothing (status NO_BEAM) until the
next injection.

A synchrotron with a ramp profile (ADR 0013) holds its beam at injection until
it is told to ramp; then its energy and RF follow the profile turn by turn, the
bunch is extracted after the flattop, and the ring is at injection again with a
fresh bunch (its ramp down, which has no beam, is not tracked).
"""

import json
import math
import queue
import threading
import time
import warnings
from collections.abc import Callable, Mapping
from concurrent.futures.process import BrokenProcessPool
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import Field

from eicvibe.diagnostics.analysis import averaged_tune
from eicvibe.diagnostics.bpm import NO_BEAM
from eicvibe.domain.lattice import ElementKind, Lattice
from eicvibe.domain.lattice._base import DomainModel
from eicvibe.domain.machine import MachineRegistry
from eicvibe.domain.machine.power import design_value
from eicvibe.domain.machine.ramps import RampProfile, ramp_cavities, ramp_period, ramp_problems
from eicvibe.engines import engine as find_engine
from eicvibe.engines.adapter import rereferenced
from eicvibe.engines.contracts import Effect, Fidelity, OpticsRequest, Options, OrbitRequest, Support
from eicvibe.engines.convert import C_LIGHT, ZERO_PHASE, reference, rf_phase
from eicvibe.engines.model import effective
from eicvibe.engines.remote import EngineHost
from eicvibe.engines.worker import EngineWorker, RemoteSession

from .errors import BpmModel, MachineErrors, VirtualBpms
from .history import BpmHistory, RunInfo
from .matching import Matching, ring_matching
from .schedule import Program, Schedule


class Beam(DomainModel):
    """The bunch a run injects: a Gaussian matched to the optics where it is injected.

    In a ring that is the periodic solution (the one-turn map) about the closed orbit; with RF the
    momentum spread sets the longitudinal emittance and the bunch length follows. A single pass
    starts from its line's initial optics (ADR 0007).
    """

    particles: int = Field(100, ge=1, description="Macroparticles; one tracks the centroid alone")
    emittance_x: float = Field(1e-8, ge=0, description="Geometric, m rad (the horizontal mode's, if coupled)")
    emittance_y: float = Field(1e-9, ge=0)
    sigma_delta: float = Field(5e-4, ge=0, description="Relative momentum spread")
    sigma_zeta: float = Field(
        5e-3, ge=0, description="Bunch length, m: where nothing matches it (a single pass, a ring without RF)"
    )
    charge: float = Field(1e-9, gt=0, description="Bunch charge, C; the BPMs' intensity is in nC")
    seed: int = 1


def default_engine() -> str:
    """TrackPad when it is installed (faster for bunches here), else Xsuite."""
    return "trackpad" if find_engine("trackpad").available() else "xsuite"


def ring_bpms(
    lattice: Lattice, branch: str, registry: MachineRegistry | None = None
) -> list[tuple[str, str, float]]:
    """The BPMs of a ring as ``(occurrence, name, s)`` in beam order.

    With a registry, its BPM devices and their EIC names; without one, the ring's instruments.
    """
    layout = lattice.layout(branch)
    placed = {p.id: p for p in layout.placements}
    if registry is not None:
        found = [
            (occurrence, device.name, placed[occurrence].s_exit)
            for device in registry.devices
            if device.type.value == "bpm"
            for occurrence in device.occurrences
            if occurrence in placed
        ]
        if found:
            return sorted(found, key=lambda b: b[2])
    return [(p.id, p.name, p.s_exit) for p in layout.placements if p.kind is ElementKind.INSTRUMENT]


def rf_warnings(
    lattice: Lattice, branch: str, momentum_compaction: float, values: Mapping[tuple[str, str], float]
) -> list[str]:
    """Cavities whose phase cannot hold a stored beam: on the unstable side of the transition.

    The energy gain goes as sin(phase - omega tau) (MAD-X's convention, see engines.convert): a bunch is
    stable where the slip factor eta = alpha_c - 1/gamma^2 times cos(phase) is negative, phase 0 below
    transition and pi (MAD-X's LAG 0.5) above it, when no energy is lost.
    """
    particle = reference("virtual ring", lattice, branch)
    gamma = 1.0 / math.sqrt(1.0 - particle["beta0"] ** 2)
    eta = momentum_compaction - 1.0 / gamma**2
    ring = lattice.layout(branch).branch.name
    out = []
    for placed in lattice.layout(branch).placements:
        if placed.kind is not ElementKind.RFCAVITY:
            continue
        rf = effective(placed, values).get("RFP", {})
        if not rf.get("voltage"):
            continue
        phase = rf_phase("virtual ring", placed.name, rf)
        if eta * math.cos(phase) >= 0:
            # What people set is RFP.phase, which the cavity's zero_phase convention shifts.
            setting = rf.get("phase", 0.0) or 0.0
            needed = ((math.pi if eta > 0 else 0.0) - (phase - setting)) % (2 * math.pi)
            side, lag = ("above", " (MAD-X LAG 0.5)") if eta > 0 else ("below", " (MAD-X LAG 0)")
            now, wanted = math.degrees(setting) % 360, math.degrees(needed)
            out.append(
                f"{ring}: cavity {placed.name} at RF phase {now:.0f} degrees is unstable {side} transition "
                f"(eta = {eta:.3g}); a stored beam needs {wanted:.0f} degrees{lag}"
            )
    return out


class RingRun:
    """A ring tracked turn after turn in an engine worker, its BPMs recorded into a history."""

    def __init__(
        self,
        lattice: Lattice,
        branch: str,
        directory: str | Path,
        *,
        engine: str | None = None,
        fidelity: Fidelity = Fidelity.STANDARD,
        effects: frozenset[Effect] | None = None,
        beam: Beam | None = None,
        bpm: BpmModel | None = None,
        chunk: int = 1024,
        values: Mapping[tuple[str, str], float] | None = None,
        registry: MachineRegistry | None = None,
        retention_bytes: int = 1 << 30,
        run_id: str | None = None,
        machine: str | None = None,
        turns: int | None = None,
        max_rate: float | None = None,
        buffer_turns: int = 4096,
        slow_period: float = 1.0,
        ramp_turns: int = 1024,
        call_seconds: float = 0.25,
        errors: MachineErrors | None = None,
        ramp: RampProfile | None = None,
        host: EngineHost | None = None,
    ) -> None:
        layout = lattice.layout(branch)
        if not layout.periodic:
            raise ValueError(
                f"{lattice.branch(branch).name} is not a ring; a run tracks a ring turn after turn"
            )
        self.lattice = lattice
        self.branch = layout.id
        self.directory = Path(directory)
        self.ramp_profile = ramp
        """The ring's ramp (ADR 0013), which :meth:`ramp` runs; None: it stays at injection."""
        self.engine = engine or ("xsuite" if ramp is not None else default_engine())
        self.host = host
        """Where the engine runs (ADR 0015): None here, in a process of its own."""
        if effects is None:  # the RF acts; a ring that ramps radiates, where its engine tracks radiation
            effects = frozenset({Effect.LONGITUDINAL}) | (_radiation(self.engine) if ramp else frozenset())
        self.options = Options(fidelity=fidelity, effects=effects)
        self._matching_options = Options(fidelity=fidelity, effects=effects - {Effect.QUANTUM_EXCITATION})
        """A bunch's matching: quantum excitation, being random, has no closed orbit."""
        self._cycle: dict[str, Any] | None = None
        """The ramp under way: the turns it started at, reaches the flattop at and extracts at."""
        self.beam = beam or Beam()
        self.bpm = bpm or BpmModel()
        self.chunk = chunk
        """The most turns an engine call tracks; calls are sized to ``call_seconds`` within it."""
        self.values = dict(values or {})
        self.retention_bytes = retention_bytes
        self.run_id = run_id
        """The run's id; by default the branch's or route's name and the time."""
        self.machine = machine
        self.turns = turns
        """Stop by itself once this many turns are recorded (the last chunk shortened to fit); None: never."""
        self.max_rate = max_rate
        """Turns per second at most (a demo need not take a whole core); None: as fast as the engine goes."""
        self.buffer_turns = max(buffer_turns, chunk)
        """The BPMs' turn-by-turn buffer: the latest turns kept, and the length of a capture."""
        self.slow_period = slow_period
        """The slow orbit's sample, in s of wall clock: the readings averaged over each."""
        self.ramp_turns = ramp_turns
        """Turns a setting change takes: a smooth ramp (a half cosine) from the value at that turn, as a
        magnet follows its supply, so the beam's orbit and optics follow without a mismatch; 0: a step."""
        self.call_seconds = call_seconds
        """Wall clock each engine call aims at: commands are taken up within about as long."""
        self.errors = errors if errors is not None and not errors.empty else None
        """The machine's own errors (``errors.py``): in the engine's values, never in :attr:`values`."""
        self._standing: list[str] = list(errors.problems) if errors is not None else []
        """Warnings that stay while the run goes: its errors' (a choice of nothing, no closed orbit)."""
        self._schedule = Schedule(
            self.values, lambda key: design_value(self.lattice.placement(key[0]), key[1]), self._engine_values
        )
        """The settings turn by turn: the changes ramping in, the ramp's program, the supplies' noise."""
        self._rate: float | None = None
        self._pending: list[dict[str, Any]] = []
        bpms = ring_bpms(lattice, branch, registry)
        if not bpms:
            raise ValueError(f"{lattice.branch(branch).name} has no BPMs to record")
        self.observe = tuple(b[0] for b in bpms)
        self.bpm_names = [b[1] for b in bpms]
        self.bpm_s = [b[2] for b in bpms]
        particle = reference("virtual ring", lattice, self.branch)
        self.revolution_period = layout.length / (particle["beta0"] * C_LIGHT)
        self._design = (particle["pc"], particle["mass"])
        self._pc = particle["pc"]
        """The reference momentum the bunch's coordinates are about: the design's, or a ramp's turn's."""
        self._bpms = VirtualBpms(self.bpm, len(bpms), self.beam.particles, self.beam.charge)
        self._bunches = np.random.default_rng(self.beam.seed)
        self._commands: queue.Queue[Callable[[], None]] = queue.Queue()
        self._stopping = threading.Event()
        self._paused = threading.Event()
        self._stopped = threading.Event()
        self._thread: threading.Thread | None = None
        self._worker: EngineWorker | None = None
        self._session: RemoteSession | None = None
        self._writer: BpmHistory | None = None
        self._coordinates = np.zeros((0, 6))
        self._turn = 0
        self.error: BaseException | None = None
        self.warnings: list[str] = []
        """What the run found wrong: RF that holds no bunch (checked at the start and when RF changes)."""
        self._compaction: float | None = None
        self.tune_turns = 1024
        """Turns the tune monitor's spectrum takes."""
        self.tunes: dict[str, Any] | None = None
        """The tune monitor's reading, once it has ``tune_turns`` turns of beam: fractional tunes (between
        0 and 0.5, as one pickup cannot tell q from 1 - q), the turn, and whether settings were still
        ramping in those turns."""
        self._centroids = np.zeros((2, 0, 0))
        self._synchrotron = 0.0
        self._changed = -(10**9)

    # -- life --------------------------------------------------------------------------

    def start(self) -> RingRun:
        """Open the engine, inject a bunch and start tracking in the background."""
        self._worker = EngineWorker(self.engine, self.host)
        self._session = self._worker.open_session(
            self.lattice, self.branch, self.options, self._engine_values()
        )
        if self.errors is not None:
            self._schedule.noise = self.errors.fluctuations(self.revolution_period)
        matching = self._matching()
        info = RunInfo(
            id=self.run_id or f"{self.lattice.branch(self.branch).name}-{datetime.now(UTC):%Y%m%dT%H%M%S}",
            machine=self.machine,
            lattice=self.lattice.name,
            lattice_digest=self.lattice.digest,
            branch=self.lattice.branch(self.branch).name,
            branch_id=self.branch,
            engine=self.engine,
            host=self.host.name if self.host else None,
            fidelity=self.options.fidelity.value,
            effects=sorted(e.value for e in self.options.effects),
            start=datetime.now(UTC),
            revolution_period=self.revolution_period,
            chunk=self.chunk,
            ramp_turns=self.ramp_turns,
            buffer_turns=self.buffer_turns,
            slow_period=self.slow_period,
            bpms=self.bpm_names,
            occurrences=list(self.observe),
            s=self.bpm_s,
            beam=self.beam.model_dump() | {"matched": matching.summary(self.beam)},
            resolution=self.bpm.resolution,
            orbit_resolution=self.bpm.orbit_resolution,
            retention_bytes=self.retention_bytes,
            values={f"{o} {p}": v for (o, p), v in self.values.items()},
            ramp=self._ramp_summary(),
        )
        self._writer = BpmHistory.create(self.directory, info)
        self._writer.slow_noise = self._bpms.orbit_noise
        self._hide()
        self._writer.event(0, "start", engine=self.engine, fidelity=self.options.fidelity.value)
        if Effect.LONGITUDINAL in self.options.effects:
            self._compaction = self._momentum_compaction()
        self.warnings = self._standing + (
            rf_warnings(self.lattice, self.branch, self._compaction, self.values)
            if self._compaction is not None
            else []
        )
        for warning in self.warnings:
            self._writer.event(0, "warning", message=warning)
            warnings.warn(warning, stacklevel=2)
        self._inject(matching)
        self._thread = threading.Thread(target=self._loop, name=f"ring run {info.id}", daemon=True)
        self._thread.start()
        return self

    def stop(self, timeout: float | None = None) -> None:
        """Stop after the chunk being tracked, and close the engine."""
        self._stopping.set()
        if self._thread is not None:
            self._thread.join(timeout)
        if self._session is not None:
            self._session.close()
        if self._worker is not None:
            self._worker.close()

    def __enter__(self) -> RingRun:
        return self if self._thread else self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    def pause(self) -> None:
        self._paused.set()

    def resume(self) -> None:
        self._paused.clear()

    @property
    def done(self) -> bool:
        """Whether the loop has ended: its turns are recorded, it was stopped, or it failed."""
        return self._stopped.is_set()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def turn(self) -> int:
        """Turns recorded so far."""
        return self._turn

    @property
    def history(self) -> BpmHistory:
        """The run's history, to read (it grows as the run goes on)."""
        return BpmHistory.open(self.directory)

    def wait(self, turn: int | None = None, timeout: float = 600.0) -> None:
        """Return once ``turn`` turns are recorded (by default the run's own end).

        Raises the run's error if it stopped on one.
        """
        turn = self.turns if turn is None else turn
        if turn is None:
            raise ValueError("this run has no end: give the turn to wait for")
        deadline = time.monotonic() + timeout
        while self._turn < turn:
            if self.error is not None:
                raise RuntimeError(f"the run stopped: {self.error}") from self.error
            if self._stopped.is_set():
                raise RuntimeError(f"the run stopped at turn {self._turn}")
            if time.monotonic() > deadline:
                raise TimeoutError(f"turn {self._turn} of {turn} after {timeout} s")
            time.sleep(0.02)

    # -- commands, carried out between chunks -------------------------------------------

    def kick(self, x: float = 0.0, y: float = 0.0) -> None:
        """Kick the beam by angles x and y (rad) at the ring's start, as a pinger does; the BPMs capture
        the turns around it, as a pinger's timing event triggers them."""

        def kick() -> None:
            self._coordinates[:, 1] += x
            self._coordinates[:, 3] += y
            self._writer.event(self._turn, "kick", x=x, y=y)
            self._trigger("kick", self.buffer_turns // 8, x=x, y=y)

        self._commands.put(kick)

    def acquire(self) -> None:
        """Capture the buffer as it is: its latest turns (an acquisition on request)."""
        self._commands.put(lambda: self._trigger("request", self.buffer_turns))

    def _trigger(self, kind: str, before: int, turns: int | None = None, **detail: Any) -> None:
        """Capture ``turns`` turns (``buffer_turns`` by default) around the turn now, ``before`` of them
        before it; the buffer keeps them all until the capture is saved, however long it is."""
        trigger = self._turn
        first = max(trigger - before, self._writer.span()[0] if self._turn else 0)
        turns = self.buffer_turns if turns is None else turns
        self._pending.append({"kind": kind, "trigger": trigger, "first": first, "turns": turns, **detail})
        self._writer.protect_from = min(c["first"] for c in self._pending)
        self._save_captures()

    def _save_captures(self, final: bool = False) -> None:
        """Save the captures whose turns are all recorded (at the end, what they have)."""
        for capture in list(self._pending):
            available = min(capture["first"] + capture["turns"], self._turn) - capture["first"]
            if available < capture["turns"] and not (final and available > 0):
                continue
            self._pending.remove(capture)
            detail = {k: v for k, v in capture.items() if k not in ("kind", "trigger", "first", "turns")}
            if available < capture["turns"]:
                detail["partial"] = True
            self._writer.save_capture(
                capture["kind"], capture["trigger"], capture["first"], available, **detail
            )
        self._writer.protect_from = min((c["first"] for c in self._pending), default=None)

    def update(self, values: Mapping[tuple[str, str], float], **note: Any) -> None:
        """Change settings (occurrence, parameter) as a control system would; ``note`` goes into the event.

        Each setting ramps from its value at the next turn to the new one over ``ramp_turns`` turns,
        turn by turn in the engine; one still ramping starts again from where it is, so nothing jumps.
        """
        values = dict(values)

        def update() -> None:
            now = self._turn
            self._changed = now
            if self._schedule.change(now, values, self.ramp_turns):  # a step: into the session at once
                self._session.update(self._engine_values(values))
            self._writer.event(
                now,
                "settings",
                values={f"{o} {p}": v for (o, p), v in values.items()},
                ramp_turns=self.ramp_turns,
                **note,
            )
            if self._schedule.owned(values):
                message = (
                    "the RF follows the ramp's program until extraction; its settings apply at injection"
                )
                self._writer.event(now, "warning", message=message)
            if self._compaction is not None and any(p.startswith("RFP.") for _, p in values):
                before, self.warnings = (
                    self.warnings,
                    self._standing
                    + rf_warnings(self.lattice, self.branch, self._compaction, self.values | values),
                )
                for warning in set(self.warnings) - set(before):
                    self._writer.event(now, "warning", message=warning)

        self._commands.put(update)

    def set_bpms(self, model: BpmModel) -> None:
        """The BPMs read with ``model``'s errors and noise from the next turn on (ADR 0011): they read the
        beam differently and leave it alone. The event says the noise, as ``run.json`` does; the errors
        stay hidden, ``hidden.json`` keeping what they were before."""

        def change() -> None:
            self.bpm = model
            self._bpms.remodel(model, self._turn)
            self._hide()
            self._writer.event(
                self._turn, "bpms", resolution=model.resolution, orbit_resolution=model.orbit_resolution
            )

        self._commands.put(change)

    def _hide(self) -> None:
        """What the run's errors are, and its BPMs' were, into ``hidden.json``: not part of the data."""
        hidden = self._bpms.hidden(self.bpm_names) | ({"errors": self.errors.hidden()} if self.errors else {})
        (self.directory / "hidden.json").write_text(json.dumps(hidden), encoding="utf-8")

    def _value(self, key: tuple[str, str], turn: int) -> float:
        """A setting's value before ``turn``: its change's, the run's, or the lattice's."""
        return self._schedule.value(key, turn)

    def _engine_values(self, values: Mapping[tuple[str, str], float] | None = None) -> dict:
        """Settings as the engine takes them, with the machine's errors: ``values`` (an update), or
        the run's, with every error (a session opened)."""
        if self.errors is None:
            return dict(self.values if values is None else values)
        if values is None:
            return self.errors.engine_values(self.values, complete=True)
        return self.errors.engine_values(values)

    def _call_turns(self, first: int) -> int:
        """Turns for the next engine call: about ``call_seconds`` of the run, at most ``chunk``.

        The run goes at the engine's pace, or at :attr:`max_rate` where that is slower: a call and
        the wait after it last about ``call_seconds`` either way, so commands are taken up as soon.
        """
        # A first, short call measures the engine's pace.
        pace = self._rate if self._rate is None or not self.max_rate else min(self._rate, self.max_rate)
        turns = 64 if pace is None else int(pace * self.call_seconds)
        turns = min(self.chunk, max(16, turns))
        return turns if self.turns is None else min(turns, self.turns - first)

    def inject(self) -> None:
        """Replace the beam with a fresh bunch (at injection: a ramping ring takes none)."""

        def inject() -> None:
            if self._cycle is not None:
                self._writer.event(
                    self._turn, "warning", message="the ring is ramping: it takes a bunch at injection"
                )
                return
            self._inject()

        self._commands.put(inject)

    # -- the ramp (ADR 0013) ---------------------------------------------------------------

    def ramp(self) -> None:
        """Ramp the ring as its ramp profile says: the energy from injection to extraction, the magnets
        following it (their normalized strengths kept) and the RF its program, turn by turn; then the
        flattop, the bunch extracted, and the ring at injection again, with a fresh bunch.

        Raises ValueError where the ring cannot ramp (no profile, one that does not fit it, or an engine
        without the synchrotron radiation its RF program assumes) and RuntimeError while it is ramping.
        """
        if reason := self._cannot_ramp():
            raise ValueError(reason)
        if self._cycle is not None:
            raise RuntimeError("the ring is ramping already")
        self._commands.put(self._begin_ramp)

    @property
    def ramp_refusal(self) -> str | None:
        """Why the ring cannot ramp now, or None: no profile, one that does not fit it, an engine without
        the radiation its RF program assumes, or a ramp under way."""
        if reason := self._cannot_ramp():
            return reason
        return "the ring is ramping" if self._cycle is not None else None

    def _cannot_ramp(self) -> str | None:
        profile = self.ramp_profile
        name = self.lattice.branch(self.branch).name
        if profile is None:
            return f"{name} has no ramp profile (ramps/ in the machine folder)"
        if problems := ramp_problems(profile, self.lattice):
            return "; ".join(problems)
        if Effect.LONGITUDINAL not in self.options.effects:
            return "a ramp needs the RF to act (effect longitudinal)"
        if Effect.RADIATION_DAMPING not in self.options.effects:
            end = f"{profile.rf.voltage[-1] / 1e6:.4g} MV at {math.degrees(profile.rf.phase[-1]):.0f} degrees"
            return (
                f"the RF program of {profile.name} makes up for synchrotron radiation (it ends at {end}), "
                f"which this run of {self.engine} leaves out: run the ring with radiation damping"
            )
        return None

    def _ramp_summary(self) -> dict[str, Any] | None:
        profile = self.ramp_profile
        if profile is None:
            return None
        period = ramp_period(profile, self.lattice)
        return {
            "name": profile.name,
            "injection": profile.energy.injection,
            "extraction": profile.energy.extraction,
            "flattop_turn": profile.ramp_turns(period),
            "extraction_turn": profile.extraction_turn(period),
            "period": period,
        }

    def _begin_ramp(self) -> None:
        if self._cycle is not None:
            return
        profile, start = self.ramp_profile, self._turn
        period = ramp_period(profile, self.lattice)
        above = (
            self._compaction is not None and self._compaction > 1.0 / _gamma(self.lattice, self.branch) ** 2
        )
        cavities = [
            (p.id, p.element.RFP.zero_phase.value if p.element.RFP else "ACCELERATING")
            for p in ramp_cavities(profile, self.lattice)
        ]
        self._cycle = {
            "start": start,
            "flattop": start + profile.ramp_turns(period),
            "extraction": start + profile.extraction_turn(period),
            "period": period,
            "mass": reference("virtual ring", self.lattice, self.branch)["mass"],
            "above": above,
            "cavities": cavities,
            "flat": False,
        }
        self._writer.event(
            start,
            "ramp",
            ramp=profile.name,
            injection=profile.energy.injection,
            extraction=profile.energy.extraction,
            flattop=self._cycle["flattop"],
            extract=self._cycle["extraction"],
        )
        # The BPMs' readings of every turn of the ramp, its flattop and extraction, kept whole: a ramp is
        # longer than the buffer, which keeps its turns until the capture is saved at extraction.
        self._trigger("ramp", 0, self._cycle["extraction"] - start, ramp=profile.name)
        # Until extraction the ramp's program sets the reference and the cavities, turn by turn.
        rf = frozenset((o, p) for o, _ in cavities for p in ("RFP.voltage", "RFP.phase"))
        self._schedule.program = Program(rf, self._ramp_values)

    def _ramp_values(self, turns: np.ndarray) -> tuple[np.ndarray, dict[tuple[str, str], np.ndarray]]:
        """The ramp's reference momentum and RF on ``turns``: the momenta, and the cavities' voltage and
        phase settings, turn by turn."""
        cycle, profile = self._cycle, self.ramp_profile
        rows = [profile.at(n - cycle["start"], cycle["period"], cycle["mass"]) for n in turns]
        at = {key: np.array([row[key] for row in rows]) for key in ("pc", "voltage", "phase")}
        share = at["voltage"] / len(cycle["cavities"])
        # The stable side of the zero crossing: pi - phi_s above transition, phi_s below (MAD-X's phase).
        phase = math.pi - at["phase"] if cycle["above"] else at["phase"]
        values = {}
        for occurrence, zero in cycle["cavities"]:
            values[(occurrence, "RFP.voltage")] = share
            values[(occurrence, "RFP.phase")] = phase - ZERO_PHASE[zero]
        return at["pc"], values

    def _after_ramp_chunk(self, first: int) -> None:
        """The flattop reached, or extraction: the bunch leaves and the ring is at injection again."""
        cycle, profile = self._cycle, self.ramp_profile
        if not cycle["flat"] and self._turn >= cycle["flattop"]:
            cycle["flat"] = True
            self._writer.event(cycle["flattop"], "flattop", energy=profile.energy.extraction)
        if self._turn < cycle["extraction"]:
            return
        bunch = self._coordinates
        self._writer.event(self._turn, "extract", energy=profile.energy.extraction, **_bunch_summary(bunch))
        self._coordinates = np.zeros((0, 6))
        self._cycle = None
        rf = self._schedule.end_program()  # the cavities at their settings again
        self._session.update(self._engine_values({key: self._value(key, self._turn) for key in rf}))
        self._writer.event(
            self._turn,
            "porch",
            energy=profile.energy.injection,
            note="the ramp down to injection has no beam and is not tracked",
        )
        self._inject()

    @property
    def cycle(self) -> dict[str, Any] | None:
        """Where the ring is in its ramp: ``state`` (injection, ramping, flattop), ``energy`` (eV), and
        while ramping the turns since it began and to extraction; None for a ring without a ramp."""
        profile, cycle = self.ramp_profile, self._cycle
        if profile is None:
            return None
        if cycle is None:
            return {"ramp": profile.name, "state": "injection", "energy": profile.energy.injection}
        turn = self._turn - cycle["start"]
        return {
            "ramp": profile.name,
            "state": "flattop" if self._turn >= cycle["flattop"] else "ramping",
            "energy": float(profile.energy.energy(turn, cycle["period"])),
            "turn": turn,
            "extraction": cycle["extraction"] - cycle["start"],
        }

    def _inject(self, matching: Matching | None = None) -> None:
        matching = matching or self._matching()
        self._coordinates = matching.bunch(self._bunches, self.beam)
        self._pc = self._design[0]
        qs = abs(matching.tunes[2]) % 1.0 if len(matching.tunes) > 2 else 0.0
        self._synchrotron = min(qs, 1.0 - qs)  # the tune of the line a spectrum shows
        self._centroids = np.zeros((2, 0, 0))  # a new bunch: the tune monitor starts again
        self._writer.event(
            self._turn,
            "inject",
            particles=len(self._coordinates),
            sigma_zeta=matching.sigma_zeta(self.beam),
            tunes=list(matching.tunes),
        )
        self._trigger("inject", self.buffer_turns // 8)  # the injection's oscillations

    def _matching(self) -> Matching:
        """How a bunch fits the ring now: its one-turn map about the closed orbit, at its start.

        A ring whose errors leave it no closed orbit, or no stable optics, cannot hold a beam, as a
        real one could not: the injector still sends the bunch the design asks for (matched to the
        ring without its errors, on its design orbit), the run says so, and the beam is lost.
        """
        where = f"{self.lattice.branch(self.branch).name}'s start"
        try:
            return _matched(self._session, self._matching_options, where)
        except Exception as error:  # noqa: BLE001 - the engines' own errors, whichever engine
            if self.errors is None:
                raise
            reason = str(error).splitlines()[0][:160]
        design = self._worker.open_session(self.lattice, self.branch, self.options, self.values)
        try:
            matching = _matched(design, self._matching_options, where)
        finally:
            design.close()
        name = self.lattice.branch(self.branch).name
        warning = (
            f"{name} has no closed orbit or no stable optics with its errors ({reason}): "
            "the bunch is injected as its design asks, and is lost"
        )
        if warning not in self._standing:
            self._standing.append(warning)
            self.warnings = [*self.warnings, warning]
        return matching

    def _momentum_compaction(self) -> float:
        """The ring's momentum compaction, for the RF's check: with its errors, or its design's
        where its errors leave it no optics."""
        request = OpticsRequest(options=Options(fidelity=self.options.fidelity))
        try:
            return self._session.optics(request).momentum_compaction or 0.0
        except Exception:  # noqa: BLE001 - as in _matching
            if self.errors is None:
                raise
        design = self._worker.open_session(self.lattice, self.branch, self.options, self.values)
        try:
            return design.optics(request).momentum_compaction or 0.0
        finally:
            design.close()

    # -- the loop ----------------------------------------------------------------------

    def _loop(self) -> None:
        state = "stopped"
        try:
            while not self._stopping.is_set():
                while not self._commands.empty():
                    self._commands.get_nowait()()
                if self.turns is not None and self._turn >= self.turns:
                    state = "done"
                    break
                if self._paused.is_set():
                    time.sleep(0.05)
                    continue
                started = time.monotonic()
                first = self._turn
                self._chunk()
                self._writer.set_state("running", self._turn)
                if self.turns is not None and self._turn >= self.turns:
                    state = "done"
                    break
                if self.max_rate:
                    time.sleep(max(0.0, (self._turn - first) / self.max_rate - (time.monotonic() - started)))
        except BaseException as error:  # noqa: BLE001 - kept for whoever waits on the run
            self.error = error
            state = "failed"
            if self._writer is not None:
                self._writer.event(self._turn, "error", message=f"{type(error).__name__}: {error}")
        finally:
            if self._writer is not None:
                try:
                    self._save_captures(final=True)
                    self._writer.flush()
                finally:
                    self._writer.set_state(state, self._turn)
            self._stopped.set()

    def _chunk(self) -> None:
        first, n_bpm = self._turn, len(self.observe)
        turns = self._call_turns(first)
        if self._cycle is not None:  # a call ends where the bunch is extracted
            turns = max(1, min(turns, self._cycle["extraction"] - first))
        started = time.time()
        if len(self._coordinates) == 0:  # no beam: the BPMs read nothing
            time.sleep(0.05)
            nothing = np.full((n_bpm, turns), np.nan)
            status = np.full((n_bpm, turns), NO_BEAM)
            self._writer.append(
                first, nothing, nothing, np.zeros((n_bpm, turns)), status, (started, time.time())
            )
            self._centroids, self.tunes = np.zeros((2, 0, 0)), None
            self._turn += turns
            self._schedule.advance(self._turn)
            if self._cycle is not None:
                self._after_ramp_chunk(first)
            self._save_captures()
            return
        momentum, ramps = self._schedule.window(first, turns)
        # The bunch about this call's first reference: the last call's was a turn's step below it.
        to = float(momentum[0]) if momentum is not None else self._design[0]
        self._coordinates = rereferenced(self._coordinates, self._pc, to, self._design[1])
        self._pc = to
        try:
            track = self._session.track_beam(
                self._coordinates,
                turns,
                self.observe,
                self.options,
                ramps=ramps,
                momentum=momentum,
            )
        except BrokenProcessPool:  # the engine's worker died: start another and go on from the same beam
            self._session = self._worker.open_session(
                self.lattice, self.branch, self.options, self._engine_values()
            )
            self._writer.event(first, "engine restarted")
            return
        x, y, intensity, status = self._bpms.read(track.x, track.y, track.count)
        slow = self._bpms.systematic(track.x, track.y, track.count)  # the closed-orbit acquisition's
        self._tune_monitor(np.asarray(track.x), np.asarray(track.y), first + turns)
        lost = int(np.count_nonzero(track.lost))
        remaining = np.asarray(track.final)[~np.asarray(track.lost)]
        if lost and len(remaining) == 0:  # the beam is gone: the post-mortem buffer freezes at once
            dark = np.flatnonzero((np.asarray(track.count) == 0).all(axis=0))
            gone = first + int(dark[0]) if len(dark) else first + turns
            self._pending.append(
                {
                    "kind": "post-mortem",
                    "trigger": gone,
                    "first": max(gone - 3 * self.buffer_turns // 4, self._writer.span()[0]),
                    "turns": self.buffer_turns,
                }
            )
            self._writer.protect_from = min(c["first"] for c in self._pending)
        finished = time.time()
        self._writer.append(first, x, y, intensity, status, (started, finished), slow=slow)
        rate = turns / max(finished - started, 1e-6)
        self._rate = rate if self._rate is None else 0.7 * self._rate + 0.3 * rate
        self._coordinates = remaining
        self._pc = float(momentum[-1]) if momentum is not None else self._design[0]
        self._turn += turns
        self._schedule.advance(self._turn)
        if lost:
            self._writer.event(self._turn, "loss", particles=lost, remaining=len(self._coordinates))
        if self._cycle is not None:
            self._after_ramp_chunk(first)
        self._save_captures()

    def _tune_monitor(self, x: np.ndarray, y: np.ndarray, end: int) -> None:
        """The tune monitor: the spectrum of the beam's own centroid motion, as a BBQ reads it.

        A bunch of finitely many particles moves as a whole a little (the sum of its particles'
        oscillations does not cancel), so a sensitive pickup sees the betatron tunes without a
        kick. The centroid at every BPM over the last :attr:`tune_turns` turns, their power spectra
        summed (a Hann window, the peak refined between bins), gives the reading; lines below 1.5
        synchrotron tunes are ignored, as the monitor's window does.
        """
        block = np.stack([x, y])
        if self._centroids.shape[1] != block.shape[1]:
            self._centroids = block
        else:
            self._centroids = np.concatenate([self._centroids, block], axis=2)[:, :, -self.tune_turns :]
        if self._centroids.shape[2] < self.tune_turns:
            return
        low = 1.5 * self._synchrotron
        reading = {plane: averaged_tune(self._centroids[k], low) for k, plane in enumerate(("x", "y"))}
        settled = end - self.tune_turns >= self._changed + self.ramp_turns
        self.tunes = reading | {"turn": end, "turns": self.tune_turns, "settled": bool(settled)}


def _matched(session, options: Options, where: str) -> Matching:
    """A bunch's matching to a session's ring: its one-turn map about its closed orbit."""
    orbit = session.orbit(OrbitRequest(options=options))
    closed = np.array([orbit.x[-1], orbit.px[-1], orbit.y[-1], orbit.py[-1], orbit.zeta[-1], orbit.delta[-1]])
    return ring_matching(session, closed, options, where)


def _radiation(engine: str) -> frozenset[Effect]:
    """The synchrotron radiation an engine tracks: mean damping, and quantum excitation with it."""
    effects = find_engine(engine).capabilities().effects
    wanted = (Effect.RADIATION_DAMPING, Effect.QUANTUM_EXCITATION)
    return frozenset(e for e in wanted if effects.get(e) is Support.NATIVE)


def _gamma(lattice: Lattice, branch: str) -> float:
    beta = reference("virtual ring", lattice, branch)["beta0"]
    return 1.0 / math.sqrt(1.0 - beta * beta)


def _bunch_summary(bunch: np.ndarray) -> dict[str, Any]:
    """What an extracted bunch is like: particles, rms emittances (m rad, at the ring's start, with
    dispersion), momentum spread and offset, bunch length (m)."""
    if len(bunch) < 2:
        return {"particles": len(bunch)}

    def emittance(u, up):
        u, up = u - u.mean(), up - up.mean()
        return float(math.sqrt(max((u * u).mean() * (up * up).mean() - (u * up).mean() ** 2, 0.0)))

    return {
        "particles": len(bunch),
        "emittance_x": emittance(bunch[:, 0], bunch[:, 1]),
        "emittance_y": emittance(bunch[:, 2], bunch[:, 3]),
        "sigma_delta": float(bunch[:, 5].std()),
        "delta": float(bunch[:, 5].mean()),
        "sigma_zeta": float(bunch[:, 4].std()),
    }


def run_ring(
    lattice: Lattice, branch: str, directory: str | Path, **options: Any
) -> RingRun:  # pragma: no cover - a convenience for notebooks
    """Start a :class:`RingRun` (see its arguments) and return it, tracking."""
    return RingRun(lattice, branch, directory, **options).start()
