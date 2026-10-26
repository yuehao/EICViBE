"""The TrackPad engine: a branch of the lattice as a TrackPad.jl ``Lattice`` (ADR 0006).

TrackPad runs in Julia, reached through juliacall; the conversion stays here
and ``trackpad.jl`` only calls TrackPad's constructors and computations.
Each occurrence becomes one or more TrackPad elements, so results map back
to occurrences at their exits, as for Xsuite.

TrackPad's elements follow Accelerator Toolbox conventions:
``polynom_b[n+1] = Kn_n / n!`` per length in thick magnets and integrated in
thin ones, ``polynom_a`` likewise for the skew strengths; a positive
``polynom_b[1]`` deflects toward -x, as a positive ``Kn0L`` does, and a
corrector's ``xkick`` is the deflection, ``-Kn0L``. A bend's ``polynom_b[1]``
is its field beyond the reference curvature. Displacement and roll are AT's
entrance and exit maps (T1, T2, R1, R2). TrackPad's coordinates are
``(x, px, y, py, z, dE)`` with ``z = zeta / beta0`` and ``dE = (E - E0)/(P0 c)``;
results are converted back to ``zeta`` and ``delta``.

Fidelity sets the steps of TrackPad's fourth-order integrator in thick
magnets: 2 (``fast``), 10 (``standard``, as TrackPad's MAD-X reader) or 40
(``precise``). Drifts are always exact. Optics are TrackPad's uncoupled 4D
Twiss from truncated power series (TPSA) maps: the closed orbit, transfer
matrices and chromaticity are exact derivatives of the maps, with no
finite-difference steps. Tracking and the 6D closed orbit follow the RF.
"""

import json
import math
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from collections.abc import Sequence
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from eicvibe.domain.lattice import ElementKind, Lattice, PlacedElement

from .adapter import Changes, EngineAdapter, EngineSession
from .contracts import (
    BeamTrack,
    Capabilities,
    Computation,
    Effect,
    EngineUnavailable,
    Fidelity,
    OpticsProfile,
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
    reference,
    rf_frequency,
    rf_phase,
)
from .model import SETTINGS, Values, effective

try:
    import numpy as np
except ImportError:  # the engine reports itself unavailable
    np = None

NAME = "trackpad"

#: Element kinds of trackpad.jl, by their code (the position).
KINDS = (
    "drift",
    "marker",
    "quadrupole",
    "sextupole",
    "octupole",
    "bend",
    "multipole",
    "corrector",
    "solenoid",
    "rfcavity",
    "patch",
    "translation",
    "exactbend",
)

#: The parameters of every TrackPad element, as trackpad.jl reads them by name.
COLUMNS = (
    "L",
    "angle",
    "e1",
    "e2",
    "fint1",
    "fint2",
    "gap",
    "gk",
    "b0",
    "b1",
    "b2",
    "b3",
    "a0",
    "a1",
    "a2",
    "a3",
    "order",
    "steps",
    "closed",
    "xkick",
    "ykick",
    "ks",
    "volt",
    "freq",
    "harmon",
    "philag",
    "dx",
    "dy",
    "roll",
    "pitch",
    "yaw",
    "x_pitch",
    "y_pitch",
    "tilt",
    "t_offset",
    "shift_x",
    "shift_y",
    "shift_s",
)

STEPS = {Fidelity.FAST: 2, Fidelity.STANDARD: 10, Fidelity.PRECISE: 40}
"""Integration steps of a thick magnet at the precise fidelity, and of what the others integrate fully: a
quadrupole with more than its gradient, or a pure one where TrackPad's LBend cannot radiate. Otherwise
steps are chosen as Xsuite chooses its kicks (``_pieces``)."""
KICKS_PER_STEP = 3  # TrackPad's Yoshida step kicks three times
BEND_KICK = math.pi * 1e-3  # Xsuite's automatic mode: a bend kicks once per pi mrad of its angle
TOP_ORDER = 3  # polynom_a and polynom_b hold dipole to octupole

LATTICE_KINDS = frozenset(
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

NOTES = {
    "optics.longitudinal": "TrackPad's Twiss is 4D; its 6D closed orbit and tracking follow the RF",
    Effect.RADIATION_DAMPING.value: "classical damping in bends and thick multipoles (rad=1, TrackPad since "
    "607dc13); tracking and the closed orbit (the adapter's Newton search), not optics",
    Effect.QUANTUM_EXCITATION.value: "Gaussian photon diffusion per slice (rad=2, Sands'), its random "
    "numbers keyed by seed, turn, element, particle and slice; tracking only",
    Effect.SPACE_CHARGE.value: "in TrackPad (space-charge elements); not yet in this adapter",
    Effect.WAKEFIELDS.value: "in TrackPad (longitudinal wakes); not yet in this adapter",
    Effect.BEAM_BEAM.value: "in TrackPad (strong Gaussian beam); not yet in this adapter",
    Effect.APERTURES.value: "lattice apertures are not converted yet",
    TimeScale.CYCLE.value: "tracking with a moving reference and settings, turn by turn (the adapter's "
    "own loop); optics as quasi-static snapshots",
}

_julia_module = None

#: Julia's threads in a process that tracks with TrackPad, a count or ``auto`` (a thread a core, the default):
#: where TrackPad observes the BPMs itself, a bunch is spread over them (ADR 0006).
THREADS = "EICVIBE_TRACKPAD_THREADS"


def _point_juliacall_at_julia() -> None:
    """Give juliacall a real Julia binary: juliaup's launcher cannot be loaded as a library.

    Where the launcher cannot say which (juliaup busy updating, say), the one TrackPad's environment was
    last resolved with: juliapkg takes any other as a new Julia and resolves the environment again,
    removing its Manifest first, and that resolution fails where Python's OpenSSL is newer than Julia's.
    """
    if os.environ.get("PYTHON_JULIAPKG_EXE"):
        return
    binary = None
    if launcher := shutil.which("julia"):
        try:
            binary = subprocess.run(
                [launcher, "--startup-file=no", "-e", 'print(joinpath(Sys.BINDIR, "julia"))'],
                capture_output=True,
                text=True,
                timeout=300,
                check=True,
            ).stdout.strip()
        except OSError, subprocess.SubprocessError:
            pass
    binary = binary or _resolved_julia()
    if binary:
        os.environ["PYTHON_JULIAPKG_EXE"] = binary
        if "juliapkg.state" in sys.modules:  # it read the setting as it was imported
            sys.modules["juliapkg.state"].reset_state()


def _unresolvable() -> str | None:
    """Why TrackPad's Julia environment cannot be set up here, if it has to be (first use, a new Julia,
    an update): juliacall asks for Julia's OpenSSL to be Python's major version and no newer, and
    Julia's registry has no OpenSSL 4 (October 2026). juliapkg would remove the environment's Manifest,
    fail, and leave TrackPad unable to start; refused first, the environment stays as it is."""
    import ssl

    if ssl.OPENSSL_VERSION_INFO[0] <= 3 or find_spec("juliapkg") is None:
        return None
    import juliapkg

    if juliapkg.resolve(dry_run=True):
        return None  # set up already: it loads
    return (
        f"TrackPad's Julia environment has to be set up, and this Python is linked to "
        f"{ssl.OPENSSL_VERSION}, which Julia's OpenSSL cannot match (Homebrew's Python 3.14 is); "
        "make the environment with uv's own Python, linked to OpenSSL 3: "
        "rm -rf .venv && uv sync --managed-python"
    )


def _resolved_julia() -> str | None:
    """The Julia that juliapkg last resolved TrackPad's environment with, if it is still installed."""
    if find_spec("juliapkg") is None:
        return None
    from juliapkg.state import STATE

    try:
        meta = json.loads(Path(STATE["meta"]).read_text())
    except OSError, ValueError:
        return None
    binary = meta.get("override_executable") or meta.get("executable")
    return binary if binary and os.path.isfile(binary) else None


def _julia():
    """The trackpad.jl module, loading Julia and TrackPad on first use."""
    global _julia_module
    if _julia_module is None:
        if np is None or find_spec("juliacall") is None:
            raise EngineUnavailable("juliacall is not installed: pip install 'eicvibe[trackpad]'")
        _point_juliacall_at_julia()
        if reason := _unresolvable():
            raise EngineUnavailable(reason)
        os.environ.setdefault("PYTHON_JULIACALL_THREADS", os.environ.get(THREADS, "auto"))
        if os.environ["PYTHON_JULIACALL_THREADS"] != "1":  # juliacall's condition for Julia's threads
            os.environ.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes")
        try:
            from juliacall import Main as jl

            jl.seval("using TrackPad")
            current = bool(jl.seval("isdefined(TrackPad, :CURVED_MULTIPOLES)"))
            module = jl.include(str(Path(__file__).with_name("trackpad.jl")))
        except Exception as error:  # juliacall raises its own errors, and on any install failure
            raise EngineUnavailable(f"TrackPad cannot start: {error}") from None
        if not current:  # its bends kicked with a straight-frame field: wrong in combined-function bends
            raise EngineUnavailable(
                "this TrackPad predates curved-frame multipoles in bends; update it: "
                "python -c 'import juliapkg; juliapkg.update()'"
            )
        _julia_module = module
    return _julia_module


def _version() -> str | None:
    if _julia_module is None:
        return None
    from juliacall import Main as jl

    return str(jl.seval("string(pkgversion(TrackPad))"))


class TrackPadEngine(EngineAdapter):
    name = NAME
    unavailable = "juliacall is not installed: pip install 'eicvibe[trackpad]'"

    def available(self) -> bool:
        return np is not None and find_spec("juliacall") is not None

    def capabilities(self) -> Capabilities:
        return Capabilities(
            engine=NAME,
            version=_version(),
            computations=frozenset(Computation),
            fidelities=frozenset(STEPS),
            effects={
                Effect.LONGITUDINAL: Support.NATIVE,
                Effect.RADIATION_DAMPING: Support.NATIVE,
                Effect.QUANTUM_EXCITATION: Support.NATIVE,
            },
            computation_gaps={
                Computation.OPTICS: frozenset(
                    {Effect.LONGITUDINAL, Effect.RADIATION_DAMPING, Effect.QUANTUM_EXCITATION}
                ),
                Computation.ORBIT: frozenset({Effect.QUANTUM_EXCITATION}),
            },
            time_scales={
                TimeScale.EQUILIBRIUM: Support.NATIVE,
                TimeScale.SINGLE_PASS: Support.NATIVE,
                TimeScale.TURNS: Support.NATIVE,
                TimeScale.CYCLE: Support.NATIVE,
            },
            kinds=LATTICE_KINDS,
            updatable=SETTINGS,
            notes=NOTES,
        )

    def _open(
        self, capabilities: Capabilities, lattice: Lattice, branch: str, options: Options, values: Values
    ) -> TrackPadSession:
        return TrackPadSession(_julia(), capabilities, lattice, branch, options, values)


class TrackPadSession(EngineSession):
    """One branch as a TrackPad lattice; settings replace its elements in place."""

    engine = NAME

    def __init__(
        self, jl, capabilities: Capabilities, lattice: Lattice, branch: str, options: Options, values: Values
    ) -> None:
        self.jl = jl
        super().__init__(capabilities, lattice, branch, options, values)
        particle = reference(NAME, lattice, self.branch)
        self._beta0 = particle["beta0"]
        self._revolution = self._beta0 * C_LIGHT / self.layout.length if self.layout.length else 0.0
        self._fidelity = options.fidelity
        pieces: list[tuple[str, str, dict[str, float]]] = []
        self._first: dict[str, int] = {}
        self._kinds: dict[str, list[str]] = {}
        for placed in self.layout.placements:
            mine = self._pieces(placed)
            self._first[placed.id] = len(pieces)
            self._kinds[placed.id] = [kind for _, kind, _ in mine]
            pieces += mine
        # Index of every occurrence's exit in TrackPad's arrays, which start at the entrance.
        self._exits = [self._first[o] + len(self._kinds[o]) for o in self._occurrence_ids]
        # The occurrence (by position) of every TrackPad element.
        self._occurrence_of = np.repeat(
            np.arange(len(self._occurrence_ids)), [len(self._kinds[o]) for o in self._occurrence_ids]
        )
        self.beam = jl.beam(particle["pc"], particle["mass"], particle["charge"])
        self.model = jl.lattice(*_arrays(pieces), self.beam, self.layout.periodic)
        self._random = np.random.default_rng()
        """Seeds of the quantum excitation's random numbers: a new one each call, whose turns count from 1."""

    def _pieces(self, placed: PlacedElement) -> list[tuple[str, str, dict[str, float]]]:
        return _pieces(placed, effective(placed, self.values), self._beta0, self._revolution, self._fidelity)

    # -- settings ----------------------------------------------------------------

    def _update(self, changes: Changes) -> None:
        """Change parameter values in place; a change of the elements' structure is refused."""
        for occurrence, paths in changes.items():
            placed = self.lattice.placement(occurrence)
            self.values.update({(occurrence, p): v for p, v in paths.items()})
            pieces = self._pieces(placed)
            if [kind for _, kind, _ in pieces] != self._kinds[occurrence]:
                raise UnsupportedRequest(
                    NAME, [f"{placed.name}: the change alters its structure; open a new session"]
                )
            self.jl.setelements_b(self.model, self._first[occurrence] + 1, *_arrays(pieces), self.beam)

    def _rows(self, placed: PlacedElement, values: Values):
        """The kind codes, names and rows of an occurrence's pieces with ``values``."""
        pieces = _pieces(placed, effective(placed, values), self._beta0, self._revolution, self._fidelity)
        kinds, names, _, rows = _arrays(pieces)
        return kinds, names.split("\n"), rows

    def _ramped(self, ramps: Sequence[Ramp], turns: int, options: Options, cavities: bool = False) -> tuple:
        """The ramps as the element groups trackpad.jl rebuilds turn by turn (see its ``centroids``).

        An occurrence's rows at the ramps' start, and each ramp's change of them at its end: element
        inputs are linear in the settings (strengths per length, rolls, RF voltage and phase), which a
        midpoint checks, so the rows before any turn are the start plus the fractions times the changes.
        With ``cavities`` (a moving reference), every RF cavity is a group too, to be built anew with it.
        """
        longitudinal = Effect.LONGITUDINAL in options.effects
        moving: dict[str, list[tuple[int, str, float, float]]] = defaultdict(list)
        if cavities:
            for placed in self.layout.placements:
                if placed.kind is ElementKind.RFCAVITY:
                    moving[placed.id] = []
        for k, ramp in enumerate(ramps):
            for (occurrence, path), value in ramp.start.items():  # checked already (checked_ramps)
                placed = self.lattice.placement(occurrence)
                if placed.kind is ElementKind.RFCAVITY and not longitudinal:
                    continue  # a drift here: the RF does not act
                moving[occurrence].append((k, path, value, ramp.end[(occurrence, path)]))
        order = COLUMNS.index("order")
        firsts, counts, kinds, names, base, group, ramp_of, delta = [], [], [], [], [], [], [], []
        for g, (occurrence, entries) in enumerate(moving.items(), start=1):
            placed = self.lattice.placement(occurrence)
            start = dict(self.values) | {(occurrence, p): v0 for _, p, v0, _ in entries}
            kinds0, names0, rows0 = self._rows(placed, start)
            if [KINDS[k] for k in kinds0] != self._kinds[occurrence]:
                raise UnsupportedRequest(NAME, [f"{placed.name}: the change alters its structure"])
            ends: dict[int, dict[tuple[str, str], float]] = defaultdict(dict)
            for k, path, _, v1 in entries:
                ends[k][(occurrence, path)] = v1
            changes = []
            for k, target in ends.items():
                kinds1, _, rows1 = self._rows(placed, start | target)
                middle = {key: 0.5 * (start[key] + v) for key, v in target.items()}
                _, _, rows_mid = self._rows(placed, start | middle)
                if list(kinds1) != list(kinds0):
                    raise UnsupportedRequest(NAME, [f"{placed.name}: the change alters its structure"])
                rows0[:, order] = np.maximum(rows0[:, order], rows1[:, order])
                change = rows1 - rows0
                change[:, order] = 0.0
                check = np.delete(rows0 + 0.5 * change - rows_mid, order, axis=1)
                if not np.allclose(check, 0.0, rtol=0.0, atol=1e-12 * max(1.0, np.abs(rows1).max())):
                    raise UnsupportedRequest(
                        NAME, [f"{placed.name}: its elements do not follow the setting linearly"]
                    )
                changes.append((k, change))
            firsts.append(self._first[occurrence] + 1)
            counts.append(len(kinds0))
            kinds.extend(kinds0)
            names.extend(names0)
            base.append(rows0)
            for k, change in changes:
                group.append(g)
                ramp_of.append(k + 1)
                delta.append(change)
        columns = len(COLUMNS)
        fraction = np.array([np.asarray(r.fraction, dtype=float)[:turns] for r in ramps]).reshape(
            len(ramps), turns
        )
        return (
            np.array(firsts, dtype=np.int64),
            np.array(counts, dtype=np.int64),
            np.array(kinds, dtype=np.int64),
            "\n".join(names),
            ",".join(COLUMNS),
            np.vstack(base) if base else np.zeros((0, columns)),
            np.array(group, dtype=np.int64),
            np.array(ramp_of, dtype=np.int64),
            np.vstack(delta) if delta else np.zeros((0, columns)),
            fraction,
        )

    # -- coordinates ---------------------------------------------------------------

    def _to_trackpad(self, x, px, y, py, zeta, delta):
        return np.array([x, px, y, py, zeta / self._beta0, float(self.jl.deltae(delta, self.beam))])

    def _delta(self, energy: float) -> float:
        """delta from TrackPad's dE: (1 + delta)^2 = 1 + 2 dE / beta0 + dE^2."""
        return math.sqrt(1.0 + 2.0 * energy / self._beta0 + energy * energy) - 1.0

    def _check(self, orbit, occurrences=None, exits=None) -> None:
        """Raise :class:`ParticleLost` if the orbit (6 x elements+1) was lost on the way."""
        occurrences = occurrences if occurrences is not None else self._occurrence_ids
        exits = exits if exits is not None else self._exits
        bad = np.flatnonzero(~np.isfinite(orbit).all(axis=0))
        if bad.size:
            lost_at = next(
                (o for o, e in zip(occurrences, exits, strict=True) if e >= bad[0]), occurrences[-1]
            )
            raise ParticleLost(NAME, lost_at)

    def _model(self, options: Options):
        """The lattice, with RF cavities as drifts unless the longitudinal motion is wanted."""
        return self.model if Effect.LONGITUDINAL in options.effects else self.jl.transverse(self.model)

    def _radiation(self, options: Options) -> int:
        """TrackPad's ``rad`` for the options: 0 none, 1 damping, 2 damping and quantum excitation."""
        mode = (
            2
            if Effect.QUANTUM_EXCITATION in options.effects
            else int(Effect.RADIATION_DAMPING in options.effects)
        )
        if mode and not bool(self.jl.RADIATES):
            raise UnsupportedRequest(
                NAME,
                ["this TrackPad does not radiate in its magnets: update it (607dc13, 2026-10-19, or later)"],
            )
        return mode

    def _seed(self) -> int:
        return int(self._random.integers(0, 2**63))

    def _part(self, options: Options, occurrences: list[str], partial: bool):
        """What to compute on: the lattice or part of it (as a line), its occurrences, and where they end.

        Returns the TrackPad lattice, the occurrence ids, every occurrence's exit index in its arrays
        (which start at the entrance), the occurrence (by position) of each of its elements, and whether
        it is a part.
        """
        model = self._model(options)
        if not partial:
            return model, occurrences, self._exits, self._occurrence_of, False
        first = self._first[occurrences[0]]
        stop = self._first[occurrences[-1]] + len(self._kinds[occurrences[-1]])
        exits = [self._first[o] + len(self._kinds[o]) - first for o in occurrences]
        owners = self._occurrence_of[first:stop] - self._occurrence_of[first]
        return self.jl.segment(model, first + 1, stop), occurrences, exits, owners, True

    # -- computations --------------------------------------------------------------

    def _optics(self, request: OpticsRequest, occurrences: list[str], partial: bool) -> OpticsResult:
        options = request.options
        model, occurrences, orbit_exits, owners, partial = self._part(options, occurrences, partial)
        periodic = self.layout.periodic and not partial
        if periodic:
            start = self.jl.closed_orbit(model, self.beam, 0.0, False)
            entrance = None
        else:
            i = request.initial
            start = self._to_trackpad(i.x, i.px, i.y, i.py, i.zeta, i.delta)
            entrance = self.jl.entrance(
                i.beta_x,
                i.alpha_x,
                2 * math.pi * i.mu_x,
                i.eta_x,
                i.etap_x,
                i.beta_y,
                i.alpha_y,
                2 * math.pi * i.mu_y,
                i.eta_y,
                i.etap_y,
            )
        result = self.jl.optics(model, self.beam, entrance, start, request.inside)
        table = np.asarray(result.columns)
        # The occurrence each piece belongs to, and the point where each occurrence's last piece ends.
        pieces = owners[np.asarray(result.owner) - 1]
        exits = np.searchsorted(pieces, np.arange(len(occurrences)), side="right")
        orbit = np.asarray(self.jl.along(model, self.beam, start))
        self._check(orbit, occurrences, orbit_exits)
        s, betx, alfx, bety, alfy, mux, muy, dx, dpx, dy, dpy = (list(row) for row in table)
        if partial:  # TrackPad's s starts at the part's start; the branch's at the branch's
            origin = self.lattice.placement(occurrences[0]).s_enter
            s = [v + origin for v in s]

        def column(values, scale=1.0):
            return tuple(float(values[i]) * scale for i in exits)

        def orbit_column(values):
            return tuple(float(values[i]) for i in orbit_exits)

        turn = 1 / (2 * math.pi)
        profile = None
        if request.inside:
            profile = OpticsProfile(
                index=tuple(int(i) for i in pieces),
                s=tuple(s),
                beta_x=tuple(betx),
                alpha_x=tuple(alfx),
                beta_y=tuple(bety),
                alpha_y=tuple(alfy),
                mu_x=tuple(v * turn for v in mux),
                mu_y=tuple(v * turn for v in muy),
                eta_x=tuple(dx),
                etap_x=tuple(dpx),
                eta_y=tuple(dy),
                etap_y=tuple(dpy),
            )
        return OpticsResult(
            engine=NAME,
            fidelity=options.fidelity,
            effects=options.effects,
            branch=self.branch,
            periodic=periodic,
            occurrences=tuple(occurrences),
            s=column(s),
            beta_x=column(betx),
            alpha_x=column(alfx),
            beta_y=column(bety),
            alpha_y=column(alfy),
            mu_x=column(mux, turn),
            mu_y=column(muy, turn),
            eta_x=column(dx),
            etap_x=column(dpx),
            eta_y=column(dy),
            etap_y=column(dpy),
            x=orbit_column(orbit[0]),
            px=orbit_column(orbit[1]),
            y=orbit_column(orbit[2]),
            py=orbit_column(orbit[3]),
            start=TwissInit(
                beta_x=betx[0],
                alpha_x=alfx[0],
                beta_y=bety[0],
                alpha_y=alfy[0],
                eta_x=dx[0],
                etap_x=dpx[0],
                eta_y=dy[0],
                etap_y=dpy[0],
                mu_x=mux[0] * turn,
                mu_y=muy[0] * turn,
                x=float(orbit[0][0]),
                px=float(orbit[1][0]),
                y=float(orbit[2][0]),
                py=float(orbit[3][0]),
            ),
            tune_x=_tune(result.tunex, (mux[-1] - mux[0]) * turn) if periodic else None,
            tune_y=_tune(result.tuney, (muy[-1] - muy[0]) * turn) if periodic else None,
            chroma_x=_number(result.chromx),
            chroma_y=_number(result.chromy),
            momentum_compaction=_number(result.alphac),
            method="TPSA",
            profile=profile,
        )

    def _orbit(self, request: OrbitRequest, occurrences: list[str], partial: bool) -> OrbitResult:
        options = request.options
        model, occurrences, exits, _, partial = self._part(options, occurrences, partial)
        radiating = self._radiation(options) > 0  # damping: an orbit has no quantum excitation
        if radiating:
            model = self.jl.radiating(model, 1)
        if self.layout.periodic and not partial:
            six = Effect.LONGITUDINAL in options.effects
            if radiating and six:  # where the energy loss puts it: TrackPad's own finders ignore rad
                start = self.jl.closed_orbit_radiating(model, self.beam)
            else:
                start = self.jl.closed_orbit(model, self.beam, request.delta, six)
        else:
            x, px, y, py, zeta, delta = request.initial or (0.0, 0.0, 0.0, 0.0, 0.0, request.delta)
            start = self._to_trackpad(x, px, y, py, zeta, delta)
        orbit = np.asarray(self.jl.along(model, self.beam, start, radiating=radiating))
        self._check(orbit, occurrences, exits)
        return OrbitResult(
            engine=NAME,
            fidelity=options.fidelity,
            effects=options.effects,
            branch=self.branch,
            closed=self.layout.periodic and not partial,
            occurrences=tuple(occurrences),
            x=tuple(float(orbit[0][i]) for i in exits),
            px=tuple(float(orbit[1][i]) for i in exits),
            y=tuple(float(orbit[2][i]) for i in exits),
            py=tuple(float(orbit[3][i]) for i in exits),
            zeta=tuple(float(orbit[4][i]) * self._beta0 for i in exits),
            delta=tuple(self._delta(float(orbit[5][i])) for i in exits),
        )

    def _track(self, request: TrackRequest) -> TrackResult:
        options = request.options
        index = dict(zip(self._occurrence_ids, self._exits, strict=True))
        cuts = sorted({index[o] for o in request.observe})
        coords = np.array([self._to_trackpad(*p) for p in request.particles], dtype=float)
        model = self._model(options)
        record, lost = self.jl.observe(
            model,
            self.beam,
            coords,
            request.turns,
            np.array(cuts, dtype=np.int64),
            rad=self._radiation(options),
            seed=self._seed(),
        )
        record = np.asarray(record)  # (cut, turn, particle, coordinate)
        n = len(request.particles)

        def series(k):
            return tuple(
                tuple(
                    tuple(float(record[cuts.index(index[o]), t, p, k]) for p in range(n))
                    for t in range(request.turns)
                )
                for o in request.observe
            )

        return TrackResult(
            engine=NAME,
            fidelity=options.fidelity,
            effects=options.effects,
            branch=self.branch,
            observe=request.observe,
            turns=request.turns,
            x=series(0),
            px=series(1),
            y=series(2),
            py=series(3),
            lost=tuple(bool(flag) for flag in np.asarray(lost)),
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
        """Julia averages the bunch at every observation point, every turn, and with ``ramps`` rebuilds
        the elements whose settings move before each turn they change; with ``momentum`` it puts the
        bunch about each turn's reference and rebuilds the cavities with it."""
        part = len(occurrences) < len(self._occurrence_ids)
        model, occurrences, exits, _, _ = self._part(options, occurrences, part)
        index = dict(zip(occurrences, exits, strict=True))
        cuts = sorted({index[o] for o in observe})
        beam, beta_first, beta_last = self.beam, self._beta0, self._beta0
        if momentum is not None:
            particle = reference(NAME, self.lattice, self.branch)
            beam = self.jl.beam(float(momentum[0]), particle["mass"], particle["charge"])
            beta_first, beta_last = (p / math.hypot(p, particle["mass"]) for p in (momentum[0], momentum[-1]))
        initial = coordinates.copy()
        initial[:, 4] = coordinates[:, 4] / beta_first
        initial[:, 5] = [float(self.jl.deltae(d, beam)) for d in coordinates[:, 5]]
        radiation = {"rad": self._radiation(options), "seed": self._seed()}
        if ramps or momentum is not None:
            groups = self._ramped(ramps, turns, options, cavities=momentum is not None)
            x, y, count, final, lost = self.jl.centroids(
                model, beam, initial, turns, np.array(cuts, dtype=np.int64), *groups, momentum, **radiation
            )
            if ramps:
                after: dict[tuple[str, str], float] = {}
                for ramp in ramps:
                    after |= ramp.at(turns - 1)
                self.update(after)  # the session goes on from where the ramps are
        else:
            x, y, count, final, lost = self.jl.centroids(
                model, self.beam, initial, turns, np.array(cuts, dtype=np.int64), **radiation
            )
        rows = [cuts.index(index[o]) for o in observe]
        final = np.asarray(final).copy()
        energy = final[:, 5]
        final[:, 4] *= beta_last
        final[:, 5] = np.sqrt(1.0 + 2.0 * energy / beta_last + energy * energy) - 1.0
        return BeamTrack(
            observe=tuple(observe),
            turns=turns,
            x=np.asarray(x)[rows],
            y=np.asarray(y)[rows],
            count=np.asarray(count)[rows],
            final=final,
            lost=np.asarray(lost) != 0,
        )

    def close(self) -> None:
        self.model = None


# -- conversion ------------------------------------------------------------------


def _tune(fraction, advance: float) -> float:
    """A ring's tune: the one-turn map's fractional tune, with the integer of the summed phase advance.

    With coupling, uncoupled phase advances summed element by element depend on how
    the ring is cut into pieces (sampling inside magnets cuts it finer); the one-turn
    map does not.
    """
    fraction = float(fraction)
    return fraction + round(advance - fraction)


def _number(value) -> float | None:
    return None if value is None else float(value)


def _arrays(pieces: list[tuple[str, str, dict[str, float]]]):
    """The kind codes, newline-separated names, column header and parameter rows trackpad.jl takes."""
    kinds = np.array([KINDS.index(kind) for _, kind, _ in pieces], dtype=np.int64)
    rows = np.zeros((len(pieces), len(COLUMNS)))
    for i, (_, _, parameters) in enumerate(pieces):
        for name, value in parameters.items():
            rows[i, COLUMNS.index(name)] = value
    return kinds, "\n".join(name for name, _, _ in pieces), ",".join(COLUMNS), rows


def _strengths(multipoles: dict[str, Any], length: float, *, thin: bool = False) -> dict[str, float]:
    """AT's polynom_b and polynom_a, ``Kn/n!``: per length, or integrated in a thin element."""
    top = highest_order(multipoles, length)
    if top > TOP_ORDER:
        raise UnsupportedRequest(NAME, [f"multipoles of order {top}; TrackPad's elements stop at octupoles"])
    out = {}
    for n in range(TOP_ORDER + 1):
        normal = integrated(multipoles, "Kn", n, length)
        skew = integrated(multipoles, "Ks", n, length)
        scale = math.factorial(n) * (1.0 if thin else length)
        out[f"b{n}"], out[f"a{n}"] = normal / scale, skew / scale
    out["order"] = top
    return out


def _placement(groups: dict[str, Any], roll: float, *, body_roll: bool = True) -> dict[str, float]:
    """The body's offsets, roll, and PALS pitch and yaw (about its centre; see trackpad.jl ``placement``).

    The roll is AT's R1 and R2, which turn the element's frame: ``roll`` (a multipole's tilt, a bend's
    ``tilt_ref``) and, with ``body_roll``, the body's own. A bend's body roll is not a turned frame
    (the beam would follow the bend out of its plane and come back on the reference, unkicked) but a
    turned field: :func:`_turned`.
    """
    d = displacement(NAME, groups)
    own = d.roll if body_roll else 0.0
    return {"dx": d.x, "dy": d.y, "roll": roll + own, "pitch": d.pitch, "yaw": d.yaw}


def _turned(strengths: dict[str, float], roll: float) -> dict[str, float]:
    """polynom_b and polynom_a of a field turned by ``roll`` about the reference (PALS's right-handed
    z_rot): order n by (n + 1) roll, ``b + i a`` times ``exp(-i (n + 1) roll)``, as R1 and R2 would turn
    it with the frame. A rolled bend's dipole so becomes its own field less a skew part, which kicks
    the beam by minus its angle times sin(roll) vertically."""
    if not roll:
        return strengths
    out = dict(strengths)
    for n in range(TOP_ORDER + 1):
        b, a = strengths[f"b{n}"], strengths[f"a{n}"]
        c, s = math.cos((n + 1) * roll), math.sin((n + 1) * roll)
        out[f"b{n}"], out[f"a{n}"] = b * c + a * s, a * c - b * s
    return out


def _pieces(
    placed: PlacedElement, groups: dict[str, Any], beta0: float, revolution: float, fidelity: Fidelity
) -> list[tuple[str, str, dict[str, float]]]:
    """The TrackPad elements of one occurrence as (name, kind, parameters); the last is named with its id."""
    oid, kind, length = placed.id, placed.kind, placed.length
    mm = groups.get("MagneticMultipoleP", {})
    steps = STEPS[fidelity]
    if kind in (
        ElementKind.DRIFT,
        ElementKind.MARKER,
        ElementKind.INSTRUMENT,
        ElementKind.BEGINNING_ELE,
        ElementKind.FORK,
    ):
        return [(oid, "drift", {"L": length})] if length else [(oid, "marker", {})]
    # Steps as Xsuite chooses its kicks, but at the precise fidelity (ADR 0006): a quadrupole's gradient in
    # closed form (Xsuite's thick quadrupole map; TrackPad's LBend), one step for a sextupole or an
    # octupole (Xsuite's one kick), and a bend's by its angle (_bend).
    auto = fidelity is not Fidelity.PRECISE
    if kind is ElementKind.BEND:
        return [(oid, *_bend(placed, groups, mm, steps, exact=fidelity is not Fidelity.FAST, auto=auto))]
    if kind in (ElementKind.QUADRUPOLE, ElementKind.SEXTUPOLE, ElementKind.OCTUPOLE):
        order = {ElementKind.QUADRUPOLE: 1, ElementKind.SEXTUPOLE: 2, ElementKind.OCTUPOLE: 3}[kind]
        roll = _placement(groups, common_tilt(NAME, mm, order))
        if length == 0.0:
            return [(oid, "multipole", _strengths(mm, 0.0, thin=True) | roll)]
        strengths = _strengths(mm, length)
        strengths["order"] = max(strengths["order"], order)
        name = {1: "quadrupole", 2: "sextupole", 3: "octupole"}[order]
        values = {"L": length, "steps": steps} | strengths | roll
        if auto and order == 1 and _only(strengths, "b1"):
            values["closed"] = 1.0  # integrated in `steps` where TrackPad's LBend cannot radiate
        elif auto and order > 1:
            values["steps"] = 1
        return [(oid, name, values)]
    if kind in (ElementKind.MULTIPOLE, ElementKind.KICKER):
        roll = _placement(groups, common_tilt(NAME, mm, 0))
        strengths = _strengths(mm, 0.0, thin=True)
        if strengths["order"] == 0:  # a corrector: its kick is the deflection
            kick = ("corrector", {"xkick": -strengths["b0"], "ykick": strengths["a0"]} | roll)
        else:
            kick = ("multipole", strengths | roll)
        return _centred(oid, length, kick)
    if kind is ElementKind.SOLENOID:
        if length == 0.0:
            raise UnsupportedRequest(NAME, [f"{placed.name}: a thin solenoid"])
        ks = groups.get("SolenoidP", {}).get("Ksol", 0.0)
        return [(oid, "solenoid", {"L": length, "ks": ks} | _placement(groups, 0.0))]
    if kind is ElementKind.RFCAVITY:
        rf = groups.get("RFP", {})
        cavity = {
            "L": length,
            "volt": rf.get("voltage", 0.0),
            "freq": rf_frequency(NAME, placed.name, rf, revolution),
            "harmon": rf.get("harmon") or 1.0,
            # TrackPad's energy gain is sin(philag + 2 pi f z / c), MAD-X's sin(phase - 2 pi f z / c).
            "philag": math.pi - rf_phase(NAME, placed.name, rf),
        }
        return [(oid, "rfcavity", cavity)]
    if kind is ElementKind.PATCH:
        return _patch(oid, groups.get("PatchP", {}), beta0)
    raise UnsupportedRequest(NAME, [f"{placed.name}: element kind {kind.value}"])


def _centred(
    oid: str, length: float, thin: tuple[str, dict[str, float]]
) -> list[tuple[str, str, dict[str, float]]]:
    """A thin element in the middle of a length (as MAD-X places a thick kicker's kick)."""
    if length == 0.0:
        return [(oid, *thin)]
    return [
        (f"{oid}.in", "drift", {"L": length / 2}),
        (f"{oid}.kick", *thin),
        (oid, "drift", {"L": length / 2}),
    ]


def _only(strengths: dict[str, float], name: str) -> bool:
    """Whether ``name`` is the only multipole of the strengths (b0 to b3, a0 to a3) that is not zero."""
    return all(strengths[f"{c}{n}"] == 0.0 for c in "ab" for n in range(TOP_ORDER + 1) if f"{c}{n}" != name)


def _bend(
    placed: PlacedElement,
    groups: dict[str, Any],
    mm: dict[str, Any],
    steps: int,
    *,
    exact: bool,
    auto: bool = False,
) -> tuple[str, dict]:
    """TrackPad's exact sector bend (Forest's maps), or its AT-style bend (expanded body, linear edges).

    The AT-style body lacks the curvature terms of the kinetic energy, which carry much of
    the chromaticity of strongly bent rings (half of it at rho ~ 5 m). With ``auto``, its steps are
    Xsuite's kicks: one per pi mrad of its angle, three a step; none for an exact pure dipole, whose
    body is exact.
    """
    bp = groups.get("BendP", {})
    length, angle = placed.length, bp.get("angle_ref", 0.0)
    if length == 0.0:
        raise UnsupportedRequest(NAME, [f"{placed.name}: a thin bend (no curvature in a thin multipole)"])
    e1, e2 = bend_edges(NAME, bp)
    strengths = _strengths(mm, length)
    strengths["b0"] = bend_field(bp, mm, length)  # the whole dipole, turned with the body's roll
    strengths = _turned(strengths, displacement(NAME, groups).roll)
    strengths["b0"] -= angle / length  # the field beyond the curvature
    strengths["order"] = highest_order({**mm, "Kn0": strengths["b0"], "Kn0L": 0.0}, length)
    if auto:
        kicks = max(1, int(abs(angle) / BEND_KICK))
        steps = 0 if exact and _only(strengths, "") else math.ceil(kicks / KICKS_PER_STEP)
    shape = {"L": length, "angle": angle, "e1": e1, "e2": e2, "steps": steps}
    shape |= strengths | _placement(groups, bp.get("tilt_ref", 0.0), body_roll=False)
    # The fringe is full gap x FINT = 2 HGAP x FINT in MAD-X's terms; the lattice stores HGAP x FINT.
    entry, exit_ = bp.get("edge1_int", 0.0), bp.get("edge2_int", 0.0)
    if not exact:
        return "bend", shape | {"gap": 2.0, "fint1": entry, "fint2": exit_}
    if entry != exit_:
        raise UnsupportedRequest(NAME, [f"{placed.name}: an exact bend with different fringe integrals"])
    return "exactbend", shape | {"gk": 2.0 * entry}


def _patch(oid: str, patch: dict[str, Any], beta0: float) -> list[tuple[str, str, dict[str, float]]]:
    """A patch as TrackPad's Translation, then one Patch per rotation (TrackPad's own order is R_x, R_y).

    A patch has no length, so a z offset delays the reference particle by z / (beta0 c), as in
    Xsuite; TrackPad's Translation moves the reference time with it, so the delay is added back.
    """
    (x, y, z), rotations = patch_frame(NAME, patch)
    pieces = [("translation", {"shift_x": x, "shift_y": y, "shift_s": z})] if x or y or z else []
    if z:
        pieces.append(("patch", {"t_offset": -z / (beta0 * C_LIGHT)}))
    fields = {"y_rot": "y_pitch", "x_rot": "x_pitch", "z_rot": "tilt"}
    pieces += [("patch", {fields[key]: angle}) for key, angle in rotations]
    if not pieces:
        pieces.append(("marker", {}))
    names = [f"{oid}.{n}" for n in range(1, len(pieces))] + [oid]
    return [(name, kind, parameters) for name, (kind, parameters) in zip(names, pieces, strict=True)]
