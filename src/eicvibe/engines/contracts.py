"""What engines are asked and what they answer, independent of any engine (ADR 0006).

Requests and results are plain data in the lattice's conventions (MAD-X/PALS
coordinates x, px, y, py, zeta, delta; px = Px/P0; positions in m, phase
advances in turns), reported at occurrence ids, so callers never see engine
objects and a session can run in another process.

Four axes describe what an engine can do, and a request says what it needs:

- :class:`Fidelity`: how exactly the elements are integrated, from fast
  approximate models for live use to precise ones for benchmarks;
- :class:`Effect`: physics beyond the reference transverse dynamics;
- :class:`TimeScale`: equilibrium, single pass, turn by turn, or a machine
  cycle in which settings change with time;
- :class:`Computation`: optics, orbit, tracking.

An engine that cannot meet a request refuses it and says why
(:class:`UnsupportedRequest`); it never drops an effect silently.
"""

from enum import StrEnum
from typing import Self

from pydantic import Field, model_validator

from eicvibe.domain.lattice import ElementKind
from eicvibe.domain.lattice._base import DomainModel, Real


class Fidelity(StrEnum):
    LINEAR = "linear"
    """First-order transfer maps only (response-matrix models); fastest."""
    FAST = "fast"
    """Approximate element models and few integration steps, for live use."""
    STANDARD = "standard"
    """The engine's default element models: what results mean unless stated."""
    PRECISE = "precise"
    """Exact Hamiltonian, full fringe fields and fine integration, for benchmarks."""


class Effect(StrEnum):
    LONGITUDINAL = "longitudinal"
    """The RF acts: 6D optics, a ring's closed orbit at the momentum the RF sets, 6D tracking.

    Without it every computation is transverse: RF cavities are drifts and the
    momentum stays as given (MAD-X's TWISS instead searches a 6D closed orbit
    whenever cavities are on).
    """
    RADIATION_DAMPING = "radiation_damping"
    """Mean synchrotron radiation: energy loss and damping."""
    QUANTUM_EXCITATION = "quantum_excitation"
    """Stochastic synchrotron radiation (tracking)."""
    SPACE_CHARGE = "space_charge"
    WAKEFIELDS = "wakefields"
    BEAM_BEAM = "beam_beam"
    INTRABEAM_SCATTERING = "intrabeam_scattering"
    SPIN = "spin"
    APERTURES = "apertures"
    """Particle losses on apertures (tracking)."""


class TimeScale(StrEnum):
    EQUILIBRIUM = "equilibrium"
    """A ring's periodic solution (closed orbit, optics) at fixed settings."""
    SINGLE_PASS = "single_pass"
    """One pass through a line from initial conditions."""
    TURNS = "turns"
    """Turn-by-turn tracking at fixed settings."""
    CYCLE = "cycle"
    """Settings that change in time: a ramp, an injection or extraction program."""


class Computation(StrEnum):
    OPTICS = "optics"
    ORBIT = "orbit"
    TRACKING = "tracking"


class Support(StrEnum):
    NATIVE = "native"
    APPROXIMATE = "approximate"
    """Provided, with a stated approximation (e.g. a quasi-static cycle)."""
    NONE = "none"


class EngineUnavailable(RuntimeError):
    """The engine is not installed or cannot start here."""


class ParticleLost(RuntimeError):
    """The orbit's particle was lost: an unstable or misdirected orbit."""

    def __init__(self, engine: str, occurrence: str) -> None:
        super().__init__(f"{engine}: the orbit's particle is lost at occurrence {occurrence}")
        self.engine = engine
        self.occurrence = occurrence

    def __reduce__(self):  # crosses process boundaries (engines.worker)
        return type(self), (self.engine, self.occurrence)


class UnsupportedRequest(ValueError):
    """The engine cannot do what was asked; ``reasons`` says what is missing."""

    def __init__(self, engine: str, reasons: list[str]) -> None:
        super().__init__(f"{engine} cannot do this: " + "; ".join(reasons))
        self.engine = engine
        self.reasons = reasons

    def __reduce__(self):
        return type(self), (self.engine, self.reasons)


class Options(DomainModel):
    """How a computation is to be done."""

    fidelity: Fidelity = Fidelity.STANDARD
    effects: frozenset[Effect] = frozenset()
    allow_approximate: bool = Field(False, description="Accept effects an engine only approximates")


class Capabilities(DomainModel):
    """What an engine adapter provides."""

    engine: str
    version: str | None = None
    computations: frozenset[Computation]
    fidelities: frozenset[Fidelity]
    effects: dict[Effect, Support] = Field(default_factory=dict, description="Missing effects: none")
    computation_gaps: dict[Computation, frozenset[Effect]] = Field(
        default_factory=dict,
        description="Effects a computation lacks although the engine has them, e.g. 4D-only optics",
    )
    time_scales: dict[TimeScale, Support]
    kinds: frozenset[ElementKind] = Field(description="Element kinds it models")
    updatable: frozenset[str] = Field(
        description="Parameter paths a session changes in place, e.g. MagneticMultipoleP.Kn1"
    )
    notes: dict[str, str] = Field(default_factory=dict, description="Why something is missing or approximate")

    def gaps(
        self,
        options: Options,
        computation: Computation | None,
        time_scale: TimeScale,
        kinds: frozenset[ElementKind] = frozenset(),
    ) -> list[str]:
        """What this engine lacks for a request (any computation if None); empty when it can do it."""
        reasons = []
        if computation is not None and computation not in self.computations:
            reasons.append(f"no {computation.value} computation")
        if options.fidelity not in self.fidelities:
            reasons.append(f"no {options.fidelity.value} fidelity")
        accepted = {Support.NATIVE, Support.APPROXIMATE} if options.allow_approximate else {Support.NATIVE}
        for effect in sorted(options.effects):
            support = self.effects.get(effect, Support.NONE)
            if support not in accepted:
                note = self.notes.get(effect.value)
                reasons.append(f"effect {effect.value}: {support.value}" + (f" ({note})" if note else ""))
            elif computation is not None and effect in self.computation_gaps.get(computation, frozenset()):
                note = self.notes.get(f"{computation.value}.{effect.value}")
                reasons.append(
                    f"effect {effect.value} in {computation.value}: none" + (f" ({note})" if note else "")
                )
        support = self.time_scales.get(time_scale, Support.NONE)
        if support not in accepted:
            note = self.notes.get(time_scale.value)
            reasons.append(f"time scale {time_scale.value}: {support.value}" + (f" ({note})" if note else ""))
        missing = sorted(k.value for k in kinds - self.kinds)
        if missing:
            reasons.append(f"element kinds {', '.join(missing)}")
        return reasons


# -- requests ------------------------------------------------------------------


class TwissInit(DomainModel):
    """Optics and orbit where a line starts (or at any point, to start another line there)."""

    beta_x: Real = Field(gt=0.0)
    alpha_x: Real = 0.0
    beta_y: Real = Field(gt=0.0)
    alpha_y: Real = 0.0
    eta_x: Real = 0.0
    etap_x: Real = 0.0
    eta_y: Real = 0.0
    etap_y: Real = 0.0
    mu_x: Real = Field(0.0, description="Phase advance so far, turns")
    mu_y: Real = 0.0
    x: Real = 0.0
    px: Real = 0.0
    y: Real = 0.0
    py: Real = 0.0
    zeta: Real = 0.0
    delta: Real = 0.0


class OpticsRequest(DomainModel):
    options: Options = Field(default_factory=Options)
    initial: TwissInit | None = Field(None, description="Where a line starts; a ring needs none")
    inside: bool = Field(
        False,
        description="Also report optics inside elements (OpticsResult.profile), where the engine can "
        "sample them without changing its results",
    )


class OrbitRequest(DomainModel):
    options: Options = Field(default_factory=Options)
    initial: tuple[Real, Real, Real, Real, Real, Real] | None = Field(
        None, description="(x, px, y, py, zeta, delta) where a line starts; a ring finds its closed orbit"
    )
    delta: Real = Field(
        0.0,
        description="Momentum deviation of a ring's closed orbit; RF sets it if longitudinal",
    )


class TrackRequest(DomainModel):
    options: Options = Field(default_factory=Options)
    particles: tuple[tuple[Real, Real, Real, Real, Real, Real], ...] = Field(min_length=1)
    turns: int = Field(1, ge=1, description="Passes through the branch; a line takes one")
    observe: tuple[str, ...] = Field((), description="Occurrences to record at, every turn (e.g. BPMs)")


# -- results -------------------------------------------------------------------


class _Result(DomainModel):
    engine: str
    fidelity: Fidelity
    effects: frozenset[Effect]
    branch: str


class OpticsProfile(DomainModel):
    """Optics inside elements as well: at the branch start, then at the end of every piece an engine sampled.

    ``index[k]`` is the occurrence (its position in the result's occurrences)
    whose piece ends at point k + 1, so the segment from point k to k + 1 lies
    in that occurrence. Columns have one more entry than ``index``.
    """

    index: tuple[int, ...]
    s: tuple[Real, ...]
    beta_x: tuple[Real, ...]
    alpha_x: tuple[Real, ...]
    beta_y: tuple[Real, ...]
    alpha_y: tuple[Real, ...]
    mu_x: tuple[Real, ...]
    mu_y: tuple[Real, ...]
    eta_x: tuple[Real, ...]
    etap_x: tuple[Real, ...]
    eta_y: tuple[Real, ...]
    etap_y: tuple[Real, ...]

    @model_validator(mode="after")
    def _columns(self) -> Self:
        n = len(self.index) + 1
        for name in type(self).model_fields:
            if name != "index" and len(getattr(self, name)) != n:
                raise ValueError(f"profile {name} has {len(getattr(self, name))} points, not {n}")
        return self


class OpticsResult(_Result):
    """Optics at the exit of every occurrence of a branch, in beam order."""

    periodic: bool
    occurrences: tuple[str, ...]
    s: tuple[Real, ...]
    beta_x: tuple[Real, ...]
    alpha_x: tuple[Real, ...]
    beta_y: tuple[Real, ...]
    alpha_y: tuple[Real, ...]
    mu_x: tuple[Real, ...] = Field(description="Phase advance from the start, turns")
    mu_y: tuple[Real, ...]
    eta_x: tuple[Real, ...]
    etap_x: tuple[Real, ...]
    eta_y: tuple[Real, ...]
    etap_y: tuple[Real, ...]
    x: tuple[Real, ...]
    px: tuple[Real, ...]
    y: tuple[Real, ...]
    py: tuple[Real, ...]
    start: TwissInit = Field(description="Optics where the branch starts")
    method: str = Field(description="How the maps and chromaticity were computed, e.g. TPSA")
    profile: OpticsProfile | None = Field(
        None, description="Optics inside elements too, when asked for and the engine samples them"
    )
    tune_x: Real | None = Field(
        None, description="A ring's tunes; a line's total phase advance is mu at its end"
    )
    tune_y: Real | None = None
    chroma_x: Real | None = Field(None, description="dQ/d(delta)")
    chroma_y: Real | None = None
    momentum_compaction: Real | None = None

    @model_validator(mode="after")
    def _columns(self) -> Self:
        n = len(self.occurrences)
        for name in (
            "s",
            "beta_x",
            "alpha_x",
            "beta_y",
            "alpha_y",
            "mu_x",
            "mu_y",
            "eta_x",
            "etap_x",
            "eta_y",
            "etap_y",
            "x",
            "px",
            "y",
            "py",
        ):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{name} has {len(getattr(self, name))} values for {n} occurrences")
        return self

    def at(self, occurrence: str) -> TwissInit:
        """The optics at an occurrence's exit, e.g. to start a line that forks there."""
        i = self.occurrences.index(occurrence)
        return TwissInit(
            **{
                name: getattr(self, name)[i]
                for name in TwissInit.model_fields
                if hasattr(self, name) and name not in ("zeta", "delta")
            }
        )


class MachineOptics(DomainModel):
    """Optics of every branch of a lattice, and how each branch got them or why it has none."""

    engine: str
    version: str | None
    fidelity: Fidelity
    results: dict[str, OpticsResult] = Field(description="By branch id")
    notes: dict[str, str] = Field(
        description="By branch id: periodic solution, initial conditions, the fork it starts from, "
        "or why it has no optics"
    )


class OrbitResult(_Result):
    """Orbit at the exit of every occurrence; a ring's is its closed orbit."""

    closed: bool
    occurrences: tuple[str, ...]
    x: tuple[Real, ...]
    px: tuple[Real, ...]
    y: tuple[Real, ...]
    py: tuple[Real, ...]
    zeta: tuple[Real, ...]
    delta: tuple[Real, ...]

    @model_validator(mode="after")
    def _columns(self) -> Self:
        n = len(self.occurrences)
        for name in ("x", "px", "y", "py", "zeta", "delta"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{name} has {len(getattr(self, name))} values for {n} occurrences")
        return self


type Record = tuple[tuple[tuple[float, ...], ...], ...]
"""Values by observation point, turn and particle; NaN once the particle is lost."""


class TrackResult(_Result):
    """Coordinates at the observation points: ``x[point][turn][particle]`` and likewise."""

    observe: tuple[str, ...]
    turns: int
    x: Record
    px: Record
    y: Record
    py: Record
    lost: tuple[bool, ...] = Field(description="Per particle, lost by the end")
