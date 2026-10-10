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
"""

import itertools
from collections import defaultdict
from collections.abc import Sequence
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from eicvibe.domain.lattice import ElementKind, Lattice, PlacedElement

from .contracts import (
    BeamTrack,
    Capabilities,
    Computation,
    Effect,
    EngineUnavailable,
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
from .model import SETTINGS, Values, effective, require, span

try:
    import numpy as np
    import xtrack as xt
except ImportError:  # the engine reports itself unavailable
    xt = None

NAME = "xsuite"

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


class XsuiteEngine:
    name = NAME

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
                TimeScale.CYCLE: Support.APPROXIMATE,
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
                TimeScale.CYCLE.value: "quasi-static snapshots (model.quasi_static); no time-dependent knobs",
                Effect.QUANTUM_EXCITATION.value: "tracking only",
            },
        )

    def open(
        self, lattice: Lattice, branch: str, options: Options | None = None, values: Values | None = None
    ) -> XsuiteSession:
        if xt is None:
            raise EngineUnavailable("Xsuite is not installed: pip install 'eicvibe[xsuite]'")
        options = options or Options()
        require(self.capabilities(), lattice, branch, options)
        return XsuiteSession(lattice, branch, options, values or {})


class XsuiteSession:
    """One branch as an Xsuite line; settings change it in place."""

    engine = NAME

    def __init__(self, lattice: Lattice, branch: str, options: Options, values: Values) -> None:
        self.lattice = lattice
        self.layout = lattice.layout(branch)
        self.branch = self.layout.id
        self.options = options
        self.values: dict[tuple[str, str], float] = dict(values)
        self._particle = reference(NAME, lattice, self.branch)
        self._revolution = (
            self._particle["beta0"] * C_LIGHT / self.layout.length if self.layout.length else 0.0
        )
        self._exit: dict[str, str] = {}
        self._pieces: dict[str, list[str]] = {}
        self._occurrence_ids = [p.id for p in self.layout.placements]
        self.line = self._line(options)

    def _line(self, options: Options, monitors: dict[str, Any] | None = None):
        """The branch as a new Xsuite line, with ``monitors`` (by occurrence) after their occurrences."""
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
        line.build_tracker()
        return line

    # -- settings ----------------------------------------------------------------

    def update(self, values: Values) -> None:
        """Change parameter values in place; a parameter that would change the line's structure is refused."""
        changed = defaultdict(dict)
        for (occurrence, path), value in values.items():
            if path not in SETTINGS:
                raise UnsupportedRequest(NAME, [f"{path} cannot change in place; open a new session"])
            changed[occurrence][path] = value
        for occurrence, paths in changed.items():
            placed = self.lattice.placement(occurrence)
            if placed.branch_id != self.branch:
                raise UnsupportedRequest(NAME, [f"occurrence {occurrence} is not in this branch"])
            self.values.update({(occurrence, p): v for p, v in paths.items()})
            pieces = _pieces(placed, effective(placed, self.values), self._revolution)
            if [n for n, _ in pieces] != self._pieces[occurrence]:
                raise UnsupportedRequest(
                    NAME, [f"{placed.name}: the change alters its structure; open a new session"]
                )
            for name, element in pieces:
                target = self.line[name]
                for field, value in _fields(element).items():
                    setattr(target, field, value)

    def _set(self, line, values: Values, longitudinal: bool = True) -> None:
        """Set the elements of ``line`` (a tracking call's) to ``values`` on top of the session's."""
        changed = defaultdict(dict)
        for (occurrence, path), value in values.items():
            if path not in SETTINGS:
                raise UnsupportedRequest(NAME, [f"{path} cannot change in place; open a new session"])
            changed[occurrence][path] = value
        merged = dict(self.values) | dict(values)
        for occurrence in changed:
            placed = self.lattice.placement(occurrence)
            if placed.branch_id != self.branch:
                raise UnsupportedRequest(NAME, [f"occurrence {occurrence} is not in this branch"])
            if placed.kind is ElementKind.RFCAVITY and not longitudinal:
                continue  # its voltage stays 0: the RF does not act
            pieces = _pieces(placed, effective(placed, merged), self._revolution)
            if [n for n, _ in pieces] != self._pieces[occurrence]:
                raise UnsupportedRequest(NAME, [f"{placed.name}: the change alters its structure"])
            for name, element in pieces:
                target = line[name]
                for field, value in _fields(element).items():
                    setattr(target, field, value)

    def _ramp_fields(self, line, ramps: Sequence[Ramp], longitudinal: bool) -> tuple[list, set[str]]:
        """The element fields ``ramps`` move, each as ``(element, field, base, [(ramp, delta)])``: before
        a turn it is ``base + sum(fraction[ramp] * delta)``, set straight on the element of ``line``.

        Xsuite's fields are linear in the settings (checked at each ramp's middle), so this equals
        making the elements anew each turn at a fraction of the cost; the occurrences where a field is
        not linear (a combined tilt) are returned to be made anew.
        """
        owner = {key: r for r, ramp in enumerate(ramps) for key in ramp.start}  # the last ramp wins
        by_occurrence = defaultdict(list)
        for key in owner:
            if key[1] not in SETTINGS:
                raise UnsupportedRequest(NAME, [f"{key[1]} cannot change in place; open a new session"])
            by_occurrence[key[0]].append(key)
        fields, rebuilt = [], set()
        for occurrence, keys in by_occurrence.items():
            placed = self.lattice.placement(occurrence)
            if placed.branch_id != self.branch:
                raise UnsupportedRequest(NAME, [f"occurrence {occurrence} is not in this branch"])
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

    def optics(self, request: OpticsRequest) -> OpticsResult:
        options = request.options
        require(XsuiteEngine().capabilities(), self.lattice, self.branch, options, Computation.OPTICS)
        _radiation(self.line, options)
        method = self._method(options)
        partial = request.start is not None or request.end is not None
        occurrences = span(self._occurrence_ids, request.start, request.end, NAME)
        if self.layout.periodic and not partial:
            table = self.line.twiss(method=method)
        else:
            if request.initial is None:
                raise UnsupportedRequest(NAME, ["a line's optics need initial conditions"])
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

    def orbit(self, request: OrbitRequest) -> OrbitResult:
        options = request.options
        require(XsuiteEngine().capabilities(), self.lattice, self.branch, options, Computation.ORBIT)
        _radiation(self.line, options)
        longitudinal = Effect.LONGITUDINAL in options.effects
        names = ("x", "px", "y", "py", "zeta", "delta")
        partial = request.start is not None or request.end is not None
        occurrences = span(self._occurrence_ids, request.start, request.end, NAME)
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

    def track(self, request: TrackRequest) -> TrackResult:
        options = request.options
        time_scale = (
            TimeScale.TURNS
            if request.turns > 1
            else (TimeScale.EQUILIBRIUM if self.layout.periodic else TimeScale.SINGLE_PASS)
        )
        require(
            XsuiteEngine().capabilities(),
            self.lattice,
            self.branch,
            options,
            Computation.TRACKING,
            time_scale,
        )
        if request.turns > 1 and not self.layout.periodic:
            raise UnsupportedRequest(NAME, ["a line is passed once; turns > 1 needs a ring"])
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

    def track_beam(
        self,
        coordinates,
        turns: int,
        observe,
        options: Options,
        start: str | None = None,
        end: str | None = None,
        ramps: Sequence[Ramp] = (),
    ) -> BeamTrack:
        """Track a bunch for ``turns`` turns of the ring, or once through a line or part of a branch.

        Xsuite's beam monitors average it at every observation point, every turn. With ``ramps`` the
        line is tracked a turn at a time while settings move, its elements set before each turn.
        """
        partial = start is not None or end is not None or not self.layout.periodic
        if partial and turns != 1:
            raise UnsupportedRequest(NAME, ["a line, or part of a branch, is passed once: turns = 1"])
        if partial and ramps:
            raise UnsupportedRequest(NAME, ["settings ramp while a ring turns, not in one pass"])
        occurrences = span(self._occurrence_ids, start, end, NAME)
        if missing := [o for o in observe if o not in occurrences]:
            raise UnsupportedRequest(NAME, [f"observation points outside the part tracked: {missing[:3]}"])
        time_scale = TimeScale.SINGLE_PASS if partial else TimeScale.TURNS
        require(
            XsuiteEngine().capabilities(),
            self.lattice,
            self.branch,
            options,
            Computation.TRACKING,
            time_scale,
        )
        coordinates = np.asarray(coordinates, dtype=float).reshape(-1, 6)
        n = len(coordinates)
        monitors = {
            occurrence: xt.BeamStatsMonitor(
                start_at_turn=0,
                stop_at_turn=turns,
                particle_id_range=(0, n),
                stats=["mean_x", "mean_y", "num_particles"],
            )
            for occurrence in observe
        }
        line = self._line(options, monitors)
        _radiation(line, options)
        if Effect.LONGITUDINAL not in options.effects:
            for element in line.elements:
                if isinstance(element, xt.Cavity):
                    element.voltage = 0.0
        c = coordinates.T
        particles = line.build_particles(x=c[0], px=c[1], y=c[2], py=c[3], zeta=c[4], delta=c[5])
        if partial:
            positions = {name: i for i, name in enumerate(line.element_names)}
            last = self._exit[occurrences[-1]]
            after = f"{occurrences[-1]}.monitor" if f"{occurrences[-1]}.monitor" in positions else last
            line.track(
                particles, ele_start=positions[self._pieces[occurrences[0]][0]], ele_stop=positions[after] + 1
            )
        elif ramps:
            longitudinal = Effect.LONGITUDINAL in options.effects
            fractions = np.array([np.asarray(r.fraction, dtype=float)[:turns] for r in ramps])
            moves = [
                0,
                *(t for t in range(1, turns) if (fractions[:, t] != fractions[:, t - 1]).any()),
                turns,
            ]
            fields, rebuilt = self._ramp_fields(line, ramps, longitudinal)
            for here, there in itertools.pairwise(moves):
                f = fractions[:, here]
                for element, field, base, deltas in fields:
                    setattr(element, field, base + sum(f[r] * delta for r, delta in deltas))
                if rebuilt:
                    values: dict[tuple[str, str], float] = {}
                    for ramp in ramps:
                        values |= ramp.at(here)
                    self._set(line, {k: v for k, v in values.items() if k[0] in rebuilt}, longitudinal)
                line.track(particles, num_turns=there - here)
            after: dict[tuple[str, str], float] = {}
            for ramp in ramps:
                after |= ramp.at(turns - 1)
            self.update(after)  # the session goes on from where the ramps are
        else:
            line.track(particles, num_turns=turns)

        def stat(name: str) -> np.ndarray:
            return np.array([np.asarray(monitors[o].get(name), dtype=float)[:turns] for o in observe])

        count = stat("num_particles")
        x = np.where(count > 0, stat("mean_x"), np.nan)
        y = np.where(count > 0, stat("mean_y"), np.nan)
        order = np.argsort(particles.particle_id)
        final = np.column_stack(
            [np.asarray(getattr(particles, k))[order] for k in ("x", "px", "y", "py", "zeta", "delta")]
        )
        return BeamTrack(
            observe=tuple(observe),
            turns=turns,
            x=x.reshape(len(observe), turns),
            y=y.reshape(len(observe), turns),
            count=count.reshape(len(observe), turns).astype(int),
            final=final,
            lost=np.asarray(particles.state)[order] <= 0,
        )

    def close(self) -> None:
        self.line = None


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
    if Effect.QUANTUM_EXCITATION in options.effects:
        line.configure_radiation(model="quantum")
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


def _misalignment(groups: dict[str, Any], tilt: float = 0.0) -> dict[str, float]:
    x, y, roll = displacement(NAME, groups)
    return {"shift_x": x, "shift_y": y, "rot_s_rad": tilt + roll}


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
        shifts = _misalignment(groups, common_tilt(NAME, mm, order))
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
    shifts = _misalignment(groups, bp.get("tilt_ref", 0.0))
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
