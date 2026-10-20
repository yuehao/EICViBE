"""The Xsuite engine: a branch of the lattice as an ``xtrack.Line`` (ADR 0006).

Each occurrence becomes one or more Xsuite elements; the last of them is
named with the occurrence id, so results map back to occurrences at their
exits. Conventions follow MAD-X, as the lattice's do (ADR 0001): a bend's
positive angle and a positive ``Kn0L`` both deflect toward -x
(``knl[0] = Kn0L``), a positive ``Ks0L`` toward +y (``ksl[0] = Ks0L``), and
the RF phase in MAD-X's convention (2 pi LAG: ``zero_phase`` below transition)
is Xsuite's ``phase``.

Fidelity selects Xsuite's element models: ``fast`` keeps only the linear
part of bend edges and integrates bends in ten Yoshida kicks,
``standard`` adds the edges' chromatic terms (as MAD-X's thick bends do: the
chromaticity of strongly bent rings needs them), and
``precise`` uses the full bend model and exact drifts. Xsuite's expanded bend
body is not used: it misses half the chromaticity of a ring with rho ~ 5 m.

Bunches (``track_beam``, the virtual accelerator's tracking) are tracked where
``EICVIBE_XSUITE_CONTEXT`` says: ``cpu``, the default, on one thread; ``cpu:N`` or
``cpu:auto`` on OpenMP threads; ``cuda`` or ``cuda:N`` on a GPU, with CuPy, and
``cuda:N,M,...`` on several, a share of a large bunch each. Threads and GPUs share
out the particles, so they speed up large bunches only. GPU kernels are compiled
fully optimized unless ``XSUITE_CUDA_FAST_COMPILE`` says otherwise. The other
computations track a particle or a few and stay on one thread. An engine host sets
it (``eicvibe-engine serve --xsuite-context``, ADR 0015).
"""

import itertools
import os
from collections import defaultdict
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from functools import cache
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from eicvibe.domain.lattice import ElementKind, Lattice, PlacedElement

from .adapter import Changes, EngineAdapter, EngineSession
from .contracts import (
    BeamTrack,
    Capabilities,
    Computation,
    Effect,
    Fidelity,
    OpticsRequest,
    OpticsResult,
    Options,
    OrbitRequest,
    OrbitResult,
    ParticleLost,
    Ramp,
    Support,
    TimeScale,
    TrackRequest,
    TrackResult,
    TwissInit,
    UnsupportedRequest,
)
from .convert import (
    C_LIGHT,
    bend_edges,
    bend_field,
    common_tilt,
    displacement,
    highest_order,
    integrated,
    patch_frame,
    per_length,
    reference,
    rf_frequency,
    rf_phase,
)
from .model import SETTINGS, Values, effective

try:
    import numpy as np
    import xtrack as xt
except ImportError:  # the engine reports itself unavailable
    xt = None

NAME = "xsuite"

#: The environment variable that says where bunches are tracked (the module's docstring).
CONTEXT = "EICVIBE_XSUITE_CONTEXT"

KINDS = frozenset(
    {
        ElementKind.DRIFT,
        ElementKind.BEND,
        ElementKind.MULTIPOLE,
        ElementKind.QUADRUPOLE,
        ElementKind.SEXTUPOLE,
        ElementKind.OCTUPOLE,
        ElementKind.SOLENOID,
        ElementKind.KICKER,
        ElementKind.RFCAVITY,
        ElementKind.INSTRUMENT,
        ElementKind.MARKER,
        ElementKind.BEGINNING_ELE,
        ElementKind.FORK,
        ElementKind.PATCH,
    }
)


def _version() -> str | None:
    try:
        return version("xtrack")
    except PackageNotFoundError:
        return None


class XsuiteEngine(EngineAdapter):
    name = NAME
    unavailable = "Xsuite is not installed: pip install 'eicvibe[xsuite]'"

    def available(self) -> bool:
        return xt is not None

    def capabilities(self) -> Capabilities:
        return Capabilities(
            engine=NAME,
            version=_version(),
            computations=frozenset(Computation),
            fidelities=frozenset({Fidelity.FAST, Fidelity.STANDARD, Fidelity.PRECISE}),
            effects={
                Effect.LONGITUDINAL: Support.NATIVE,
                Effect.RADIATION_DAMPING: Support.NATIVE,
                Effect.QUANTUM_EXCITATION: Support.NATIVE,
            },
            time_scales={
                TimeScale.EQUILIBRIUM: Support.NATIVE,
                TimeScale.SINGLE_PASS: Support.NATIVE,
                TimeScale.TURNS: Support.NATIVE,
                TimeScale.CYCLE: Support.NATIVE,
            },
            kinds=KINDS,
            updatable=SETTINGS,
            notes={
                Effect.SPACE_CHARGE.value: "in Xsuite (xfields); not yet in this adapter",
                Effect.WAKEFIELDS.value: "in Xsuite (xwakes); not yet in this adapter",
                Effect.BEAM_BEAM.value: "in Xsuite (xfields); not yet in this adapter",
                Effect.INTRABEAM_SCATTERING.value: "in Xsuite; not yet in this adapter",
                Effect.SPIN.value: "in Xsuite; not yet in this adapter",
                Effect.APERTURES.value: "lattice apertures are not converted yet",
                TimeScale.CYCLE.value: "tracking with a moving reference and settings, turn by turn; "
                "optics as quasi-static snapshots (model.quasi_static)",
                Effect.QUANTUM_EXCITATION.value: "tracking only",
            },
        )

    def _open(
        self, capabilities: Capabilities, lattice: Lattice, branch: str, options: Options, values: Values
    ) -> XsuiteSession:
        return XsuiteSession(capabilities, lattice, branch, options, values)


#: Particles a GPU takes at least when a bunch is shared out: an A100 tracks 10 000 particles of the RCS in a
#: sixth of the time of 100 000, so a smaller share gains little over one GPU (ADR 0015).
PER_GPU = 25_000

#: The GPU of each context made here, by its id: CuPy's current device is a thread's.
_DEVICES: dict[int, int] = {}


def compute_contexts(spec: str | None = None) -> tuple:
    """Xsuite's contexts for tracking bunches, from ``spec`` or ``EICVIBE_XSUITE_CONTEXT``: ``(None,)`` for
    one CPU thread (Xsuite's default context), else contexts kept for the process, their kernels compiled
    once; several for ``cuda:N,M,...``, which share out a large bunch."""
    spec = (os.environ.get(CONTEXT, "") if spec is None else spec).strip().lower()
    return (None,) if spec in ("", "cpu") else _contexts(spec)


def compute_context(spec: str | None = None):
    """The first of :func:`compute_contexts`: None for one CPU thread."""
    return compute_contexts(spec)[0]


def describe_context(context) -> str:
    """Where contexts track: ``cpu``, ``cpu, 32 threads``, ``cuda:1, NVIDIA A100-SXM4-40GB``,
    ``cuda:1,3, 2 NVIDIA A100-SXM4-40GB``."""
    contexts = context if isinstance(context, tuple) else (context,)
    first = contexts[0]
    if first is None:
        return "cpu"
    threads = getattr(first, "omp_num_threads", None)
    if threads is not None:
        return f"cpu, {threads} threads" if threads != "auto" else "cpu, a thread per core"
    import cupy

    devices = [_DEVICES[id(c)] for c in contexts]
    name = cupy.cuda.runtime.getDeviceProperties(devices[0])["name"]
    name = name.decode() if isinstance(name, bytes) else name
    many = f"{len(devices)} " if len(devices) > 1 else ""
    return f"cuda:{','.join(map(str, devices))}, {many}{name}"


@cache
def _contexts(spec: str) -> tuple:
    import xobjects as xo

    kind, _, count = spec.partition(":")
    if kind == "cpu":
        return (xo.ContextCpu(omp_num_threads="auto" if count == "auto" else _count(spec, count, 1)),)
    if kind == "cuda":
        devices = [_count(spec, d, 0) for d in (count or "0").split(",")]
        if len(set(devices)) < len(devices):
            raise ValueError(f"{CONTEXT}={spec}: a GPU named twice")
        try:
            import cupy  # noqa: F401 - Xsuite's GPU context needs it
        except ImportError:
            raise ValueError(f"{CONTEXT}={spec}: CuPy is not installed (pip install cupy-cuda12x)") from None
        if "XSUITE_CUDA_FAST_COMPILE" not in os.environ:
            # Optimized kernels track 1.3 to 1.8 times as fast on an A100; they compile once, in minutes,
            # and CuPy keeps them (ADR 0015).
            xo.settings.cuda_fast_compile = False
        contexts = []
        for device in devices:
            context = xo.ContextCupy(device=device)
            _DEVICES[id(context)] = device
            contexts.append(context)
        return tuple(contexts)
    raise ValueError(f"{CONTEXT}={spec}: expected cpu, cpu:N, cpu:auto, cuda, cuda:N or cuda:N,M,...")


def _wait(context) -> None:
    """Wait for ``context``'s GPU to finish what it was given, letting other threads run meanwhile: reading
    its arrays would wait too, but holding them back (CuPy's copies keep Python's lock)."""
    device = _DEVICES.get(id(context))
    if device is not None:
        import cupy

        cupy.cuda.Device(device).synchronize()


def _on(context):
    """Where ``context``'s arrays and kernels are: its GPU made the thread's current device; nothing else for
    a CPU's."""
    device = _DEVICES.get(id(context))
    if device is None:
        return nullcontext()
    import cupy

    return cupy.cuda.Device(device)


def _count(spec: str, text: str, least: int) -> int:
    if not text.isdigit() or int(text) < least:
        raise ValueError(f"{CONTEXT}={spec}: {text or 'nothing'} is not a count from {least}")
    return int(text)


@dataclass
class _Monitored:
    """A copy of the branch's line with beam monitors, kept for the tracking calls that observe the same
    points with the same options: building one takes a quarter of a second on the RCS, about as long as
    tracking it 30 turns."""

    line: Any
    monitors: dict[str, Any]
    turns: int
    particles: int
    longitudinal: bool
    context: Any = None


#: Monitored lines a session keeps, the most recently used.
_KEPT = 2


class XsuiteSession(EngineSession):
    """One branch as an Xsuite line; settings change it in place."""

    engine = NAME

    def __init__(
        self, capabilities: Capabilities, lattice: Lattice, branch: str, options: Options, values: Values
    ) -> None:
        super().__init__(capabilities, lattice, branch, options, values)
        self._particle = reference(NAME, lattice, self.branch)
        self._revolution = (
            self._particle["beta0"] * C_LIGHT / self.layout.length if self.layout.length else 0.0
        )
        self._exit: dict[str, str] = {}
        self._pieces: dict[str, list[str]] = {}
        self._kept: dict[tuple, _Monitored] = {}
        self._contexts = compute_contexts()  # where bunches are tracked: a share of one each
        self.line = self._line(options)

    def _line(self, options: Options, monitors: dict[str, Any] | None = None, context=None):
        """The branch as a new Xsuite line, with ``monitors`` (by occurrence) after their occurrences, in
        ``context`` (Xsuite's default: one CPU thread)."""
        monitors = monitors or {}
        names, elements = [], []
        for placed in self.layout.placements:
            pieces = _pieces(placed, effective(placed, self.values), self._revolution)
            names += [name for name, _ in pieces]
            elements += [element for _, element in pieces]
            self._pieces[placed.id] = [name for name, _ in pieces]
            self._exit[placed.id] = pieces[-1][0]
            if placed.id in monitors:
                names.append(f"{placed.id}.monitor")
                elements.append(monitors[placed.id])
        line = xt.Line(elements=elements, element_names=names)
        particle = self._particle
        line.particle_ref = xt.Particles(p0c=particle["pc"], mass0=particle["mass"], q0=particle["charge"])
        _configure(line, options)
        if context is None:
            line.build_tracker()
        else:
            line.build_tracker(_context=context)
        return line

    # -- settings ----------------------------------------------------------------

    def _update(self, changes: Changes) -> None:
        """Change parameter values in place, in the session's line and the monitored lines it keeps; a
        parameter that would change the line's structure is refused."""
        lines = [(self.line, True), *((kept.line, kept.longitudinal) for kept in self._kept.values())]
        for occurrence, paths in changes.items():
            placed = self.lattice.placement(occurrence)
            self.values.update({(occurrence, p): v for p, v in paths.items()})
            pieces = _pieces(placed, effective(placed, self.values), self._revolution)
            if [n for n, _ in pieces] != self._pieces[occurrence]:
                raise UnsupportedRequest(
                    NAME, [f"{placed.name}: the change alters its structure; open a new session"]
                )
            for line, longitudinal in lines:
                if placed.kind is ElementKind.RFCAVITY and not longitudinal:
                    continue  # its voltage stays 0: the RF does not act
                with _on(line._context):
                    for name, element in pieces:
                        target = line[name]
                        for field, value in _fields(element).items():
                            _assign(target, field, value)

    def _monitored(
        self, options: Options, observe: list[str], turns: int, particles: int, share: int = 0
    ) -> _Monitored:
        """The line with beam monitors after ``observe``, for ``turns`` turns of ``particles`` particles, in
        the context of ``share``: a line kept from earlier calls, its monitors cleared, or a new one (with
        room to spare)."""
        key = (tuple(observe), options, share)
        context = self._contexts[share]
        kept = self._kept.pop(key, None)
        if kept is not None and kept.turns >= turns and kept.particles >= particles:
            with _on(context):
                # Only the sums read: Xsuite's clear resets every statistic, on a GPU a launch each.
                for monitor in kept.monitors.values():
                    for field in _SUMS:
                        getattr(monitor.data, field)[...] = 0
        else:
            turns = max(turns, kept.turns if kept else 0, 64)
            turns = 1 << (turns - 1).bit_length()  # a power of two: calls vary in length
            particles = max(particles, kept.particles if kept else 0)
            monitors = {
                occurrence: xt.BeamStatsMonitor(
                    start_at_turn=0,
                    stop_at_turn=turns,
                    particle_id_range=(0, particles),
                    stats=["mean_x", "mean_y", "num_particles"],
                )
                for occurrence in observe
            }
            longitudinal = Effect.LONGITUDINAL in options.effects
            with _on(context):
                line = self._line(options, monitors, context)
                _radiation(line, options)
                if not longitudinal:
                    for element in line.elements:
                        if isinstance(element, xt.Cavity):
                            element.voltage = 0.0
            kept = _Monitored(line, monitors, turns, particles, longitudinal, context)
        self._kept[key] = kept  # the most recent last
        while len(self._kept) > _KEPT * len(self._contexts):
            self._kept.pop(next(iter(self._kept)))
        return kept

    def _set(self, line, values: Values, longitudinal: bool = True) -> None:
        """Set the elements of ``line`` (a tracking call's) to ``values`` on top of the session's."""
        changes = self.checked(values)
        merged = dict(self.values) | dict(values)
        for occurrence in changes:
            placed = self.lattice.placement(occurrence)
            if placed.kind is ElementKind.RFCAVITY and not longitudinal:
                continue  # its voltage stays 0: the RF does not act
            pieces = _pieces(placed, effective(placed, merged), self._revolution)
            if [n for n, _ in pieces] != self._pieces[occurrence]:
                raise UnsupportedRequest(NAME, [f"{placed.name}: the change alters its structure"])
            for name, element in pieces:
                target = line[name]
                for field, value in _fields(element).items():
                    _assign(target, field, value)

    def _ramp_fields(self, line, ramps: Sequence[Ramp], longitudinal: bool) -> tuple[list, set[str]]:
        """The element fields ``ramps`` move, each as ``(element, field, base, [(ramp, delta)])``: before
        a turn it is ``base + sum(fraction[ramp] * delta)``, set straight on the element of ``line``.

        Xsuite's fields are linear in the settings (checked at each ramp's middle), so this equals
        making the elements anew each turn at a fraction of the cost; the occurrences where a field is
        not linear (a combined tilt) are returned to be made anew.
        """
        owner = {key: r for r, ramp in enumerate(ramps) for key in ramp.start}  # the last ramp wins
        by_occurrence = defaultdict(list)
        for key in owner:  # checked already (EngineSession.checked_ramps)
            by_occurrence[key[0]].append(key)
        fields, rebuilt = [], set()
        for occurrence, keys in by_occurrence.items():
            placed = self.lattice.placement(occurrence)
            if placed.kind is ElementKind.RFCAVITY and not longitudinal:
                continue  # its voltage stays 0: the RF does not act

            def made(values, placed=placed):
                pieces = _pieces(placed, effective(placed, values), self._revolution)
                if [n for n, _ in pieces] != self._pieces[placed.id]:
                    raise UnsupportedRequest(NAME, [f"{placed.name}: the change alters its structure"])
                return {name: _fields(element) for name, element in pieces}

            start = dict(self.values) | {key: ramps[owner[key]].start[key] for key in keys}
            base = made(start)
            moved = defaultdict(list)
            linear = True
            for r in sorted({owner[key] for key in keys}):
                own = [key for key in keys if owner[key] == r]
                end = made(start | {key: ramps[r].end[key] for key in own})
                middle = made(start | {key: 0.5 * (start[key] + ramps[r].end[key]) for key in own})
                for name, values in base.items():
                    for field, value in values.items():
                        delta = np.asarray(end[name][field]) - value
                        expected = value + 0.5 * delta
                        if not np.allclose(middle[name][field], expected, rtol=1e-9, atol=1e-15):
                            linear = False
                        if np.any(delta != 0):
                            moved[name, field].append((r, delta))
            if not linear:
                rebuilt.add(occurrence)
                continue
            for (name, field), deltas in moved.items():
                fields.append((line.element_dict[name], field, base[name][field], deltas))
        return fields, rebuilt

    # -- computations --------------------------------------------------------------

    def _method(self, options: Options) -> str:
        return "6d" if Effect.LONGITUDINAL in options.effects else "4d"

    def _optics(self, request: OpticsRequest, occurrences: list[str], partial: bool) -> OpticsResult:
        options = request.options
        _radiation(self.line, options)
        method = self._method(options)
        if self.layout.periodic and not partial:
            table = self.line.twiss(method=method)
        else:
            first = self._pieces[occurrences[0]][0]
            init = _twiss_init(self.line, request.initial, first)
            table = self.line.twiss(start=first, end=self._exit[occurrences[-1]], init=init, method=method)
        rows = {name: i for i, name in enumerate(table.name)}
        exits = [rows[self._exit[o]] + 1 for o in occurrences]

        def column(name):
            values = getattr(table, name)
            return tuple(float(values[i]) for i in exits)

        start = TwissInit(
            beta_x=float(table.betx[0]),
            alpha_x=float(table.alfx[0]),
            beta_y=float(table.bety[0]),
            alpha_y=float(table.alfy[0]),
            eta_x=float(table.dx[0]),
            etap_x=float(table.dpx[0]),
            eta_y=float(table.dy[0]),
            etap_y=float(table.dpy[0]),
            mu_x=float(table.mux[0]),
            mu_y=float(table.muy[0]),
            x=float(table.x[0]),
            px=float(table.px[0]),
            y=float(table.y[0]),
            py=float(table.py[0]),
        )
        periodic = self.layout.periodic and not partial
        return OpticsResult(
            engine=NAME,
            fidelity=options.fidelity,
            effects=options.effects,
            branch=self.branch,
            periodic=periodic,
            occurrences=tuple(occurrences),
            s=column("s"),
            beta_x=column("betx"),
            alpha_x=column("alfx"),
            beta_y=column("bety"),
            alpha_y=column("alfy"),
            mu_x=column("mux"),
            mu_y=column("muy"),
            eta_x=column("dx"),
            etap_x=column("dpx"),
            eta_y=column("dy"),
            etap_y=column("dpy"),
            x=column("x"),
            px=column("px"),
            y=column("y"),
            py=column("py"),
            start=start,
            tune_x=float(table.qx) if periodic else None,
            tune_y=float(table.qy) if periodic else None,
            chroma_x=float(table.dqx) if periodic else None,
            chroma_y=float(table.dqy) if periodic else None,
            momentum_compaction=float(table.momentum_compaction_factor) if periodic else None,
            method="tracking with finite differences",
        )

    def _orbit(self, request: OrbitRequest, occurrences: list[str], partial: bool) -> OrbitResult:
        options = request.options
        _radiation(self.line, options)
        longitudinal = Effect.LONGITUDINAL in options.effects
        names = ("x", "px", "y", "py", "zeta", "delta")
        if self.layout.periodic and not longitudinal and not partial:
            # The closed orbit at fixed momentum with the RF frozen: the 4D Twiss's.
            table = self.line.twiss(method="4d", delta0=request.delta)
            rows = {name: i for i, name in enumerate(table.name)}
            exits = [rows[self._exit[o]] + 1 for o in self._occurrence_ids]
            columns = {k: tuple(float(getattr(table, k)[i]) for i in exits) for k in names}
        else:
            positions = {name: i for i, name in enumerate(self.line.element_names)}
            first = positions[self._pieces[occurrences[0]][0]]
            stop = positions[self._exit[occurrences[-1]]] + 1
            with _rf(self.line, longitudinal):
                if self.layout.periodic and not partial:  # 6D: the RF sets the momentum
                    particle = self.line.find_closed_orbit().copy()
                else:
                    x, px, y, py, zeta, delta = request.initial or (0.0, 0.0, 0.0, 0.0, 0.0, request.delta)
                    particle = self.line.build_particles(
                        x=[x], px=[px], y=[y], py=[py], zeta=[zeta], delta=[delta]
                    )
                if partial:
                    self.line.track(
                        particle, ele_start=first, ele_stop=stop, turn_by_turn_monitor="ONE_TURN_EBE"
                    )
                else:
                    self.line.track(particle, turn_by_turn_monitor="ONE_TURN_EBE")
            record = self.line.record_last_track
            exits = [positions[self._exit[o]] + 1 for o in occurrences]
            if particle.state[0] <= 0:
                where = int(particle.at_element[0])
                lost_at = next(
                    (o for o, e in zip(occurrences, exits, strict=True) if e > where), occurrences[-1]
                )
                raise ParticleLost(NAME, lost_at)
            columns = {k: tuple(float(getattr(record, k)[0, i]) for i in exits) for k in names}
        return OrbitResult(
            engine=NAME,
            fidelity=options.fidelity,
            effects=options.effects,
            branch=self.branch,
            closed=self.layout.periodic and not partial,
            occurrences=tuple(occurrences),
            **columns,
        )

    def _track(self, request: TrackRequest) -> TrackResult:
        options = request.options
        n = len(request.particles)
        monitors = {
            occurrence: xt.ParticlesMonitor(start_at_turn=0, stop_at_turn=request.turns, num_particles=n)
            for occurrence in request.observe
        }
        line = self._line(options, monitors)
        _radiation(line, options)
        if Effect.LONGITUDINAL not in options.effects:
            for element in line.elements:
                if isinstance(element, xt.Cavity):
                    element.voltage = 0.0
        columns = list(zip(*request.particles, strict=True))
        particles = line.build_particles(
            x=columns[0], px=columns[1], y=columns[2], py=columns[3], zeta=columns[4], delta=columns[5]
        )
        line.track(particles, num_turns=request.turns)
        by_point = {occurrence: line[f"{occurrence}.monitor"] for occurrence in monitors}

        def series(field):
            out = []
            for occurrence in request.observe:
                monitor = by_point[occurrence]
                data = np.where(monitor.state > 0, getattr(monitor, field), np.nan)  # (particle, turn)
                out.append(tuple(tuple(float(data[p, t]) for p in range(n)) for t in range(request.turns)))
            return tuple(out)

        order = np.argsort(particles.particle_id)
        return TrackResult(
            engine=NAME,
            fidelity=options.fidelity,
            effects=options.effects,
            branch=self.branch,
            observe=request.observe,
            turns=request.turns,
            x=series("x"),
            px=series("px"),
            y=series("y"),
            py=series("py"),
            lost=tuple(bool(particles.state[i] <= 0) for i in order),
        )

    def _track_beam(
        self,
        coordinates: np.ndarray,
        turns: int,
        observe: list[str],
        options: Options,
        occurrences: list[str],
        once: bool,
        ramps: tuple[Ramp, ...],
        momentum: np.ndarray | None = None,
    ) -> BeamTrack:
        """Xsuite's beam monitors average the bunch at every observation point, every turn. With ``ramps``
        or ``momentum`` the line is tracked a turn at a time while they move, its elements set and the
        bunch put about the reference before each turn (as Xsuite's own energy programs do: Xsuite's
        magnet strengths are normalized to the particles' reference). Several GPUs track a share of a
        large bunch each, at once; their monitors' counts and means are combined."""
        n = len(coordinates)
        shares = min(len(self._contexts), max(1, n // PER_GPU))
        bounds = list(itertools.pairwise(np.linspace(0, n, shares + 1).astype(int)))
        kept = [self._monitored(options, observe, turns, int(b - a), k) for k, (a, b) in enumerate(bounds)]

        def track(k: int):
            a, b = bounds[k]
            with _on(kept[k].context):
                return self._track_kept(
                    kept[k], coordinates[a:b], turns, observe, options, occurrences, once, ramps, momentum
                )

        try:
            if shares == 1:
                results = [track(0)]
            else:
                with ThreadPoolExecutor(shares, thread_name_prefix="xsuite gpu") as pool:
                    results = list(pool.map(track, range(shares)))
        except BaseException:
            if ramps or momentum is not None:  # their elements are part-way through the ramps: not to be kept
                self._kept = {key: v for key, v in self._kept.items() if all(v is not k for k in kept)}
            raise
        if ramps:
            after: dict[tuple[str, str], float] = {}
            for ramp in ramps:
                after |= ramp.at(turns - 1)
            self.update(after)  # the session goes on from where the ramps are
        count = sum(r[2] for r in results)
        with np.errstate(invalid="ignore", divide="ignore"):
            x = sum(np.nan_to_num(r[0]) * r[2] for r in results) / count
            y = sum(np.nan_to_num(r[1]) * r[2] for r in results) / count
        return BeamTrack(
            observe=tuple(observe),
            turns=turns,
            x=np.where(count > 0, x, np.nan),
            y=np.where(count > 0, y, np.nan),
            count=count.astype(int),
            final=np.concatenate([r[3] for r in results]).reshape(-1, 6),
            lost=np.concatenate([r[4] for r in results]),
        )

    def _track_kept(
        self,
        kept: _Monitored,
        coordinates: np.ndarray,
        turns: int,
        observe: list[str],
        options: Options,
        occurrences: list[str],
        once: bool,
        ramps: tuple[Ramp, ...],
        momentum: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """A bunch (or a share of one) tracked on ``kept``'s line, its device current: the monitors' mean x
        and y and counts (``[point, turn]``), and the final coordinates and losses, by particle."""
        line, monitors = kept.line, kept.monitors
        c = coordinates.T
        reference = None
        if momentum is not None:
            ref = self._particle
            reference = xt.Particles(p0c=float(momentum[0]), mass0=ref["mass"], q0=ref["charge"])
        particles = line.build_particles(
            particle_ref=reference, x=c[0], px=c[1], y=c[2], py=c[3], zeta=c[4], delta=c[5]
        )
        if once:
            positions = {name: i for i, name in enumerate(line.element_names)}
            last = self._exit[occurrences[-1]]
            after = f"{occurrences[-1]}.monitor" if f"{occurrences[-1]}.monitor" in positions else last
            line.track(
                particles, ele_start=positions[self._pieces[occurrences[0]][0]], ele_stop=positions[after] + 1
            )
        elif ramps or momentum is not None:
            longitudinal = Effect.LONGITUDINAL in options.effects
            fractions = np.array([np.asarray(r.fraction, dtype=float)[:turns] for r in ramps]).reshape(
                len(ramps), turns
            )
            moved = (fractions[:, 1:] != fractions[:, :-1]).any(axis=0)
            if momentum is not None:
                moved |= momentum[1:] != momentum[:-1]
            moves = [0, *(int(t) + 1 for t in np.flatnonzero(moved)), turns]
            fields, rebuilt = self._ramp_fields(line, ramps, longitudinal)
            for here, there in itertools.pairwise(moves):
                if momentum is not None and here > 0:
                    particles.update_p0c_and_energy_deviations(p0c=float(momentum[here]), update_pxpy=True)
                f = fractions[:, here]
                for element, field, base, deltas in fields:
                    _assign(element, field, base + sum(f[r] * delta for r, delta in deltas))
                if rebuilt:
                    values: dict[tuple[str, str], float] = {}
                    for ramp in ramps:
                        values |= ramp.at(here)
                    self._set(line, {k: v for k, v in values.items() if k[0] in rebuilt}, longitudinal)
                line.track(particles, num_turns=there - here)
        else:
            line.track(particles, num_turns=turns)
        _wait(kept.context)  # the GPU done before its monitors are read
        count, x, y = _centroids(monitors, observe, turns)
        host = _on_host(particles)
        order = np.argsort(host("particle_id"))
        final = np.column_stack([host(k)[order] for k in ("x", "px", "y", "py", "zeta", "delta")])
        return x, y, count, final, host("state")[order] <= 0

    def close(self) -> None:
        self.line = None
        self._kept.clear()


#: The beam monitors' sums the BPMs read: particles counted, and their x and y added up.
_SUMS = ("num_particles", "sum_x", "sum_y")


def _centroids(monitors: dict[str, Any], observe: list[str], turns: int) -> tuple[np.ndarray, ...]:
    """The monitors' particle counts and centroids (``[point, turn]``, NaN where none is left), their sums
    read at once from wherever they are: Xsuite's own reading copies each statistic of each monitor apart,
    a copy each from a GPU."""
    if not observe:
        empty = np.zeros((0, turns))
        return empty, empty, empty
    context = monitors[observe[0]]._context
    rows = [_nplike(getattr(monitors[o].data, f))[:turns] for o in observe for f in _SUMS]
    sums = np.asarray(context.nparray_from_context_array(context.nplike_lib.stack(rows)), dtype=float)
    count, sum_x, sum_y = sums.reshape(len(observe), len(_SUMS), turns).transpose(1, 0, 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        return count, np.where(count > 0, sum_x / count, np.nan), np.where(count > 0, sum_y / count, np.nan)


def _nplike(array):
    """An xobjects array as its context's own (numpy's, CuPy's); a monitor's data may be one already."""
    return array.to_nplike() if hasattr(array, "to_nplike") else array


def _assign(element, field: str, value) -> None:
    """Set an element's field, an array moved first to the element's context (a GPU's memory)."""
    if isinstance(value, np.ndarray) and value.ndim:
        value = element._context.nparray_to_context_array(value)
    setattr(element, field, value)


def _on_host(particles):
    """The particles' fields as numpy arrays, from the context (CPU or GPU) that tracked them."""
    context = particles._context
    return lambda name: np.asarray(context.nparray_from_context_array(getattr(particles, name)))


# -- conversion ------------------------------------------------------------------


def _configure(line, options: Options) -> None:
    if options.fidelity is Fidelity.FAST:
        line.configure_bend_model(core="adaptive", edge="linear")
        line.configure_drift_model("expanded")
        # Xsuite's adaptive integrator takes many kicks in strong combined-function bends (the BAR's:
        # 170 of its 173 us per particle and turn); ten Yoshida kicks keep tunes within 1e-3.
        for element in line.elements:
            if isinstance(element, xt.Bend):
                element.integrator = "yoshida-6"
                element.num_multipole_kicks = 10
    elif options.fidelity is Fidelity.PRECISE:
        line.configure_bend_model(core="full", edge="full")
        line.configure_drift_model("exact")
    else:
        line.configure_bend_model(core="adaptive", edge="full")
        line.configure_drift_model("expanded")


@contextmanager
def _rf(line, on: bool):
    """The line's RF cavities as they are, or switched off for a transverse computation."""
    cavities = [] if on else [e for e in line.elements if isinstance(e, xt.Cavity)]
    voltages = [c.voltage for c in cavities]
    for cavity in cavities:
        cavity.voltage = 0.0
    try:
        yield
    finally:
        for cavity, voltage in zip(cavities, voltages, strict=True):
            cavity.voltage = voltage


def _radiation(line, options: Options) -> None:
    """Synchrotron radiation in the line's magnets. Quantum excitation is Xsuite's ``quantum-kick``: the
    photons a slice emits counted, their total energy drawn from tables, where ``quantum`` draws each
    photon's; the same energy loss, ten times faster on a GPU (ADR 0015)."""
    if Effect.QUANTUM_EXCITATION in options.effects:
        line.configure_radiation(model="quantum-kick")
    elif Effect.RADIATION_DAMPING in options.effects:
        line.configure_radiation(model="mean")
    else:
        line.configure_radiation(model=None)


def _twiss_init(line, start: TwissInit, element: str | None = None):
    return xt.TwissInit(
        element_name=element or line.element_names[0],
        line=line,
        betx=start.beta_x,
        alfx=start.alpha_x,
        bety=start.beta_y,
        alfy=start.alpha_y,
        dx=start.eta_x,
        dpx=start.etap_x,
        dy=start.eta_y,
        dpy=start.etap_y,
        mux=start.mu_x,
        muy=start.mu_y,
        x=start.x,
        px=start.px,
        y=start.y,
        py=start.py,
        zeta=start.zeta,
        delta=start.delta,
    )


def _fields(element) -> dict[str, Any]:
    """The settable fields of a freshly made element, to copy onto the one in the line."""
    out = {}
    for name in (
        "k0",
        "k1",
        "k1s",
        "k2",
        "k2s",
        "k3",
        "k3s",
        "ks",
        "voltage",
        "frequency",
        "phase",
        "knl",
        "ksl",
        "shift_x",
        "shift_y",
        "rot_s_rad",
        "rot_x_rad",
        "rot_y_rad",
        "rot_shift_anchor",
    ):
        if hasattr(element, name):
            value = getattr(element, name)
            out[name] = np.array(value) if hasattr(value, "__len__") else value
    return out


def _orders(multipoles: dict[str, Any], length: float, skip: set[int] = frozenset()) -> tuple[list, list]:
    """knl and ksl (integrated) of every order present, except the element's main ones."""
    top = highest_order(multipoles, length)
    knl = [0.0 if n in skip else integrated(multipoles, "Kn", n, length) for n in range(top + 1)]
    ksl = [0.0 if n in skip else integrated(multipoles, "Ks", n, length) for n in range(top + 1)]
    return knl, ksl


def _misalignment(groups: dict[str, Any], tilt: float = 0.0, length: float = 0.0) -> dict[str, float]:
    """Xsuite's shifts and rotations of a body, about its centre (``rot_shift_anchor``, from the entry).

    Xsuite's ``rot_y_rad`` is PALS's right-handed yaw; its ``rot_x_rad`` turns the exit towards +y,
    so it is minus PALS's pitch. The anchor moves only what pitch and yaw do.
    """
    d = displacement(NAME, groups)
    return {
        "shift_x": d.x,
        "shift_y": d.y,
        "rot_s_rad": tilt + d.roll,
        "rot_x_rad": -d.pitch,
        "rot_y_rad": d.yaw,
        "rot_shift_anchor": length / 2,
    }


def _pieces(placed: PlacedElement, groups: dict[str, Any], revolution: float) -> list[tuple[str, Any]]:
    """The Xsuite elements of one occurrence, the last one named with its id."""
    oid, kind, length = placed.id, placed.kind, placed.length
    mm = groups.get("MagneticMultipoleP", {})
    if kind in (
        ElementKind.DRIFT,
        ElementKind.MARKER,
        ElementKind.INSTRUMENT,
        ElementKind.BEGINNING_ELE,
        ElementKind.FORK,
    ):
        return [(oid, xt.Drift(length=length) if length else xt.Marker())]
    if kind is ElementKind.BEND:
        return _bend(oid, length, groups, mm)
    if kind in (ElementKind.QUADRUPOLE, ElementKind.SEXTUPOLE, ElementKind.OCTUPOLE):
        order = {ElementKind.QUADRUPOLE: 1, ElementKind.SEXTUPOLE: 2, ElementKind.OCTUPOLE: 3}[kind]
        shifts = _misalignment(groups, common_tilt(NAME, mm, order), length)
        if length == 0.0:
            knl, ksl = _orders(mm, 0.0)
            return [(oid, xt.Multipole(knl=knl, ksl=ksl, **shifts))]
        knl, ksl = _orders(mm, length, skip={order})
        cls = {1: xt.Quadrupole, 2: xt.Sextupole, 3: xt.Octupole}[order]
        strength = {
            f"k{order}": per_length(mm, "Kn", order, length),
            f"k{order}s": per_length(mm, "Ks", order, length),
        }
        return [(oid, cls(length=length, knl=knl, ksl=ksl, **strength, **shifts))]
    if kind in (ElementKind.MULTIPOLE, ElementKind.KICKER):
        knl, ksl = _orders(mm, length)
        kick = xt.Multipole(knl=knl, ksl=ksl, **_misalignment(groups, common_tilt(NAME, mm, 0)))
        return _centred(oid, length, kick)
    if kind is ElementKind.SOLENOID:
        if length == 0.0:
            raise UnsupportedRequest(NAME, [f"{placed.name}: a thin solenoid"])
        return [(oid, xt.UniformSolenoid(length=length, ks=groups.get("SolenoidP", {}).get("Ksol", 0.0)))]
    if kind is ElementKind.RFCAVITY:
        rf = groups.get("RFP", {})
        frequency = rf_frequency(NAME, placed.name, rf, revolution)
        phase = rf_phase(NAME, placed.name, rf)
        cavity = xt.Cavity(voltage=rf.get("voltage", 0.0), frequency=frequency, phase=phase)
        return _centred(oid, length, cavity)
    if kind is ElementKind.PATCH:
        return _patch(oid, groups.get("PatchP", {}))
    raise UnsupportedRequest(NAME, [f"{placed.name}: element kind {kind.value}"])


def _centred(oid: str, length: float, thin) -> list[tuple[str, Any]]:
    """A thin element in the middle of a length, as drift, element, drift; the last is named ``oid``."""
    if length == 0.0:
        return [(oid, thin)]
    return [
        (f"{oid}.in", xt.Drift(length=length / 2)),
        (f"{oid}.kick", thin),
        (oid, xt.Drift(length=length / 2)),
    ]


def _bend(oid: str, length: float, groups: dict[str, Any], mm: dict[str, Any]) -> list[tuple[str, Any]]:
    bp = groups.get("BendP", {})
    angle = bp.get("angle_ref", 0.0)
    entry_angle, exit_angle = bend_edges(NAME, bp)
    shifts = _misalignment(groups, bp.get("tilt_ref", 0.0), length)
    if length == 0.0:
        knl, ksl = _orders(mm, 0.0)
        knl[0] = knl[0] or angle
        return [(oid, xt.Multipole(knl=knl, ksl=ksl, hxl=angle, **shifts))]
    if per_length(mm, "Ks", 1, length):
        raise UnsupportedRequest(NAME, ["a skew gradient in a bend is not converted yet"])
    knl, ksl = _orders(mm, length, skip={0, 1})
    entry, exit_ = bp.get("edge1_int", 0.0), bp.get("edge2_int", 0.0)
    bend = xt.Bend(
        length=length,
        angle=angle,
        k0=bend_field(bp, mm, length),
        k0_from_h=False,
        k1=per_length(mm, "Kn", 1, length),
        knl=knl,
        ksl=ksl,
        edge_entry_angle=entry_angle,
        edge_exit_angle=exit_angle,
        # MAD-X uses FINT x HGAP; the lattice stores the product (ADR 0001).
        edge_entry_fint=1.0 if entry else 0.0,
        edge_entry_hgap=entry,
        edge_exit_fint=1.0 if exit_ else 0.0,
        edge_exit_hgap=exit_,
        **shifts,
    )
    return [(oid, bend)]


def _patch(oid: str, patch: dict[str, Any]) -> list[tuple[str, Any]]:
    """A patch as Xsuite's Translation and Rotations, which take MAD-X's values as they are."""
    (x, y, z), rotations = patch_frame(NAME, patch)
    pieces = [xt.Translation(shift_x=x, shift_y=y, shift_s=z)] if x or y or z else []
    fields = {"y_rot": "rot_y_rad", "x_rot": "rot_x_rad", "z_rot": "rot_s_rad"}
    pieces += [xt.Rotation(**{fields[key]: angle}) for key, angle in rotations]
    if not pieces:
        pieces.append(xt.Marker())
    names = [f"{oid}.{n}" for n in range(1, len(pieces))] + [oid]
    return list(zip(names, pieces, strict=True))
