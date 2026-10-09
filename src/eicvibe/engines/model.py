"""Sessions, and computations that span branches or time, for any engine (ADR 0006).

An :class:`Engine` opens a :class:`Session` on one branch of a lattice: it
builds its own model once, then takes :meth:`Session.update` calls that
change parameter values in place, the way a virtual machine's settings and
hidden errors change. Values are keyed ``(occurrence id, parameter path)``,
as the power circuits produce them (``MagneticMultipoleP.Kn1``,
``BodyShiftP.x_offset``), and replace the lattice's values.

On top of sessions, and the same for every engine:

- :func:`machine_optics` computes every branch: rings from their periodic
  solution, lines from their own initial conditions or from the optics of
  the branch they fork from;
- :func:`quasi_static` follows a :class:`Program` of settings over a machine
  cycle by evaluating snapshots, an approximation that holds while settings
  change slowly compared with the beam's dynamics.
"""

from collections.abc import Callable, Iterable, Mapping
from typing import Any, Protocol

from eicvibe.domain.lattice import GROUP_TYPES, ElementKind, Lattice, PlacedElement

from .contracts import (
    Capabilities,
    Computation,
    EngineUnavailable,
    OpticsRequest,
    OpticsResult,
    Options,
    OrbitRequest,
    OrbitResult,
    TimeScale,
    TrackRequest,
    TrackResult,
    TwissInit,
    UnsupportedRequest,
)

type Values = Mapping[tuple[str, str], float]

#: Parameters that settings and errors change, which sessions update in place:
#: multipole strengths and rolls, RF voltage and phase, solenoid strength, and
#: an element's displacement and roll.
SETTINGS = frozenset(
    {f"MagneticMultipoleP.{p}{n}{suffix}" for p in ("Kn", "Ks") for n in range(11) for suffix in ("", "L")}
    | {f"MagneticMultipoleP.tilt{n}" for n in range(11)}
    | {
        "RFP.voltage",
        "RFP.phase",
        "SolenoidP.Ksol",
        "BodyShiftP.x_offset",
        "BodyShiftP.y_offset",
        "BodyShiftP.z_rot",
    }
)


class Session(Protocol):
    engine: str
    branch: str

    def update(self, values: Values) -> None: ...

    def optics(self, request: OpticsRequest) -> OpticsResult: ...

    def orbit(self, request: OrbitRequest) -> OrbitResult: ...

    def track(self, request: TrackRequest) -> TrackResult: ...

    def close(self) -> None: ...


class Engine(Protocol):
    name: str

    def available(self) -> bool:
        """Whether the engine's software is installed (a cheap check; starting it may still fail)."""
        ...

    def capabilities(self) -> Capabilities: ...

    def open(
        self, lattice: Lattice, branch: str, options: Options | None = None, values: Values | None = None
    ) -> Session: ...


def kinds_of(lattice: Lattice, branch: str) -> frozenset[ElementKind]:
    return frozenset(p.kind for p in lattice.layout(branch).placements)


def require(
    capabilities: Capabilities,
    lattice: Lattice,
    branch: str,
    options: Options,
    computation: Computation | None = None,
    time_scale: TimeScale | None = None,
) -> None:
    """Raise :class:`UnsupportedRequest` unless the engine can do this on this branch.

    Without a computation, what any computation would need: what opening a session checks.
    """
    if time_scale is None:
        periodic = lattice.layout(branch).periodic
        time_scale = TimeScale.EQUILIBRIUM if periodic else TimeScale.SINGLE_PASS
    reasons = capabilities.gaps(options, computation, time_scale, kinds_of(lattice, branch))
    if reasons:
        raise UnsupportedRequest(capabilities.engine, reasons)


def effective(placed: PlacedElement, values: Values) -> dict[str, dict[str, Any]]:
    """Parameter groups of an occurrence (definition, overrides) with ``values`` applied."""
    groups: dict[str, dict[str, Any]] = {}
    for name in GROUP_TYPES:
        group = getattr(placed.element, name, None)
        if group is not None:
            groups[name] = group.model_dump()
    for (occurrence, path), value in values.items():
        if occurrence == placed.id:
            group, _, parameter = path.partition(".")
            groups.setdefault(group, {})[parameter] = value
    return groups


def initial_conditions(lattice: Lattice, branch: str) -> TwissInit | None:
    """A line's own initial optics, from its BeginningEle's TwissP (and ParticleP orbit)."""
    conditions = lattice.initial_conditions(branch)
    twiss = conditions.twiss if conditions else None
    if twiss is None or not twiss.beta_a or not twiss.beta_b:
        return None
    return TwissInit(
        beta_x=twiss.beta_a,
        alpha_x=twiss.alpha_a,
        beta_y=twiss.beta_b,
        alpha_y=twiss.alpha_b,
        eta_x=twiss.eta_x,
        etap_x=twiss.etap_x,
        eta_y=twiss.eta_y,
        etap_y=twiss.etap_y,
    )


def machine_optics(
    engine: Engine,
    lattice: Lattice,
    *,
    values: Values | None = None,
    options: Options | None = None,
    branches: Iterable[str] | None = None,
    notes: dict[str, str] | None = None,
    inside: bool = False,
) -> dict[str, OpticsResult]:
    """Optics of every branch (or those named), by branch id.

    A ring uses its periodic solution. A line starts from its own initial
    conditions, or else from the optics at the fork that leads to its start
    in a branch computed before it. A line with neither is reported missing
    from the result rather than given invented optics.

    With ``notes``, it receives for every branch how its optics were started,
    or why it has none, and a branch the engine cannot compute is noted
    instead of stopping the others (an unavailable engine still raises).
    With ``inside``, engines that can also report the optics inside elements.
    """
    values = values or {}
    options = options or Options()
    wanted = (
        {lattice.layout(b).id for b in branches} if branches is not None else {b.id for b in lattice.branches}
    )
    results: dict[str, OpticsResult] = {}
    found: dict[str, str] = {}
    pending = [b.id for b in lattice.branches if b.id in wanted]
    progress = True
    while pending and progress:
        progress = False
        for branch in list(pending):
            layout = lattice.layout(branch)
            if layout.periodic:
                start, origin = None, "periodic solution"
            elif (start := initial_conditions(lattice, branch)) is not None:
                origin = "initial conditions"
            else:
                start, origin = _from_fork(lattice, branch, results)
            if not layout.periodic and start is None:
                if any(_lands_at_start(e) and e.from_branch in pending for e in lattice.forks_into(branch)):
                    continue  # its parent comes first
                pending.remove(branch)
                found[branch] = (
                    "no initial conditions, and no fork from a branch with optics leads to its start"
                )
                continue
            pending.remove(branch)
            progress = True
            try:
                session = engine.open(lattice, branch, options, values)
                try:
                    request = OpticsRequest(options=options, initial=start, inside=inside)
                    results[branch] = session.optics(request)
                finally:
                    session.close()
            except EngineUnavailable:
                raise
            except Exception as error:  # noqa: BLE001 - reported per branch when notes are wanted
                if notes is None:
                    raise
                found[branch] = f"failed: {error}"
                continue
            found[branch] = origin
    for branch in pending:  # waiting on a parent that never got optics
        found[branch] = "no initial conditions, and no fork from a branch with optics leads to its start"
    if notes is not None:
        notes.update(found)
    return results


def _lands_at_start(edge) -> bool:
    """Whether a fork lands where its branch starts: at the start, or at s = 0 (a septum's entrance)."""
    return edge.to_occurrence is None or edge.s_to == 0.0


def _from_fork(
    lattice: Lattice, branch: str, results: Mapping[str, OpticsResult]
) -> tuple[TwissInit | None, str]:
    """The optics at a fork into this branch's start, from a branch already computed, and its name."""
    for edge in lattice.forks_into(branch):
        if _lands_at_start(edge) and edge.from_branch in results:
            name = lattice.placement(edge.id).name
            # A fork is thin: its exit is where it is.
            return results[edge.from_branch].at(
                edge.id
            ), f"fork {name} in {lattice.branch(edge.from_branch).name}"
    return None, ""


class Program:
    """Settings that change in time: for each (occurrence, parameter), (time, value) points.

    Values between points are interpolated linearly and held beyond the ends.
    """

    def __init__(self, points: Mapping[tuple[str, str], Iterable[tuple[float, float]]]) -> None:
        self.points = {key: sorted(series) for key, series in points.items()}
        for key, series in self.points.items():
            if not series:
                raise ValueError(f"program {key}: no points")

    def at(self, time: float) -> dict[tuple[str, str], float]:
        values = {}
        for key, series in self.points.items():
            if time <= series[0][0]:
                values[key] = series[0][1]
            elif time >= series[-1][0]:
                values[key] = series[-1][1]
            else:
                for (t0, v0), (t1, v1) in zip(series, series[1:], strict=False):
                    if t0 <= time <= t1:
                        values[key] = v0 + (v1 - v0) * (time - t0) / (t1 - t0) if t1 > t0 else v1
                        break
        return values


def quasi_static(
    session: Session,
    program: Program,
    times: Iterable[float],
    compute: Callable[[Session], Any] | None = None,
) -> list[tuple[float, Any]]:
    """Evaluate ``compute`` (by default the optics) at each time of a slowly changing program.

    A cycle approximated by snapshots: right while settings change slowly
    compared with the beam's dynamics (a synchrotron ramp turn by turn);
    wrong for fast programs such as kicker pulses, which need tracking.
    """
    compute = compute or (lambda s: s.optics(OpticsRequest()))
    results = []
    for time in times:
        session.update(program.at(time))
        results.append((time, compute(session)))
    return results
