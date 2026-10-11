"""Matching optics: settings that bring lines' optics to targets where they end, by least squares.

A beam that enters a ring with other optics than the ring's own (beta, alpha,
dispersion and its slope where it lands) has its size beating round the ring and
its emittance growing once it filaments: the mismatch ``Bmag`` (1 when matched)
is how much. Matching changes chosen settings (quadrupole strengths) until the
optics where each stretch of line ends are its targets, from the optics where it
starts, with any engine. Stretches are matched together when they depend on each
other: one may start from another's optics at a point (a line leaving a fork),
or have to end with another's optics there (two paths that must meet with the
same beam). Strengths stay within bounds, beta inside each stretch below a limit,
and as close to where they started as the targets allow.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from eicvibe.domain.lattice import Lattice

from .contracts import OpticsRequest, OpticsResult, Options, TwissInit
from .model import Engine, Values

#: Optics a match can aim at, and the size a miss is measured against: beta relative, alpha
#: absolute, dispersion in units of 1 cm and its slope of 1 mrad.
TARGETS = {
    "beta_x": None,
    "alpha_x": 1.0,
    "beta_y": None,
    "alpha_y": 1.0,
    "eta_x": 0.01,
    "etap_x": 0.001,
    "eta_y": 0.01,
    "etap_y": 0.001,
}
#: What "the same optics" means when a stretch must end with another's: beta and alpha in both
#: planes, the horizontal dispersion and its slope.
SAME = ("beta_x", "alpha_x", "beta_y", "alpha_y", "eta_x", "etap_x")


def mismatch(beta: float, alpha: float, target_beta: float, target_alpha: float) -> float:
    """``Bmag``: 1 for a matched beam, (1 + Δβ/β ...) otherwise; the emittance grows by it once filamented."""
    gamma, target_gamma = (1 + alpha * alpha) / beta, (1 + target_alpha**2) / target_beta
    return 0.5 * (beta * target_gamma + target_beta * gamma - 2 * alpha * target_alpha)


@dataclass(frozen=True)
class At:
    """The optics of another stretch of the same match, at an occurrence's exit."""

    stretch: str
    occurrence: str


@dataclass(frozen=True)
class Stretch:
    """A stretch of one branch whose optics a match computes: from ``start``'s entrance to ``end``'s exit."""

    name: str
    branch: str
    start: str
    end: str
    initial: TwissInit | At
    """The optics where it starts, or another stretch's (computed before this one)."""
    targets: Mapping[str, float] | At | None = None
    """What the optics where it ends must be: values of :data:`TARGETS`, the same as another
    stretch's at a point (:data:`SAME`), or None for a stretch only others refer to."""
    beta_limit: float | None = None
    """The largest beta allowed inside it, if not the match's."""


@dataclass(frozen=True)
class StretchMatch:
    """How one stretch came out: the optics where it ends, before and after, against its targets."""

    before: TwissInit
    after: TwissInit
    targets: dict[str, float]
    """The targets met at the end (another stretch's optics as they came out)."""
    bmag_before: tuple[float, float]
    bmag_after: tuple[float, float]
    beta_max: tuple[float, float]
    """The largest beta in the stretch, matched."""


@dataclass(frozen=True)
class Matched:
    """What a match of several stretches found."""

    values: dict[tuple[str, str], float]
    """The variables' values found."""
    start: dict[tuple[str, str], float]
    """Their values before."""
    stretches: dict[str, StretchMatch]
    evaluations: int
    success: bool


@dataclass(frozen=True)
class Match:
    """What a match found: the values, how close the optics came, and the optics before and after."""

    values: dict[tuple[str, str], float]
    """The variables' values found."""
    start: dict[tuple[str, str], float]
    """Their values before."""
    before: TwissInit
    after: TwissInit
    """The optics where the stretch ends, before and after."""
    targets: dict[str, float]
    bmag_before: tuple[float, float]
    bmag_after: tuple[float, float]
    beta_max: tuple[float, float]
    """The largest beta in the stretch, matched."""
    evaluations: int
    success: bool


def match_line(
    engine: Engine,
    lattice: Lattice,
    branch: str,
    start: str,
    end: str,
    initial: TwissInit,
    variables: Sequence[tuple[str, str]],
    targets: Mapping[str, float],
    *,
    values: Values | None = None,
    options: Options | None = None,
    bound: float = 6.0,
    keep_signs: bool = False,
    beta_limit: float | None = 150.0,
    tolerance: float = 1e-2,
) -> Match:
    """Values of ``variables`` that bring the optics at the exit of ``end`` to ``targets``.

    The stretch runs from ``start``'s entrance (where the optics are ``initial``) to ``end``'s
    exit in ``branch``, with ``values`` for everything else; the rest is :func:`match_lines`'.
    """
    stretch = Stretch("line", branch, start, end, initial, dict(targets))
    found = match_lines(
        engine, lattice, [stretch], variables, values=values, options=options, bound=bound,
        keep_signs=keep_signs, beta_limit=beta_limit, tolerance=tolerance,
    )  # fmt: skip
    line = found.stretches["line"]
    return Match(
        values=found.values,
        start=found.start,
        before=line.before,
        after=line.after,
        targets=line.targets,
        bmag_before=line.bmag_before,
        bmag_after=line.bmag_after,
        beta_max=line.beta_max,
        evaluations=found.evaluations,
        success=found.success,
    )


def match_lines(
    engine: Engine,
    lattice: Lattice,
    stretches: Sequence[Stretch],
    variables: Sequence[tuple[str, str]],
    *,
    values: Values | None = None,
    options: Options | None = None,
    bound: float = 6.0,
    keep_signs: bool = False,
    beta_limit: float | None = 150.0,
    tolerance: float = 1e-2,
) -> Matched:
    """Values of ``variables`` that bring every stretch's optics where it ends to its targets.

    Stretches are computed in order, each in a session of its branch with ``values`` for
    everything else; one may start from, or have to end with, an earlier one's optics
    (:class:`At`). Each variable stays within ``±bound`` (or within twice its starting value, if
    larger), and with ``keep_signs`` on its starting side of zero (a magnet on a unipolar supply
    keeps its polarity); beta in each stretch stays below its limit; among the values that meet
    the targets, those closest to the start win. The match succeeds when every target is met
    within ``tolerance`` of its scale (1 % of beta, 0.01 in alpha, 0.1 mm of dispersion, 10 µrad
    of its slope); values that meet them already are kept as they are.
    """
    from scipy.optimize import least_squares

    values, options = dict(values or {}), options or Options()
    names = [s.name for s in stretches]
    for i, stretch in enumerate(stretches):
        refs = [r for r in (stretch.initial, stretch.targets) if isinstance(r, At)]
        if any(r.stretch not in names[:i] for r in refs):
            raise ValueError(f"stretch {stretch.name} refers to one not computed before it")
        if isinstance(stretch.targets, Mapping) and (
            unknown := [k for k in stretch.targets if k not in TARGETS]
        ):
            raise ValueError(f"cannot match {unknown}; targets are {list(TARGETS)}")
    branches = list(dict.fromkeys(s.branch for s in stretches))
    sessions = {b: engine.open(lattice, b, options, values) for b in branches}
    ids = {b: {p.id for p in lattice.layout(b).placements} for b in branches}
    x0 = np.array([values.get(v, _design(lattice, v)) for v in variables], dtype=float)
    limits = np.maximum(bound, 2 * np.abs(x0))
    low = np.where(keep_signs & (x0 > 0), 0.0, -limits)
    high = np.where(keep_signs & (x0 < 0), 0.0, limits)
    x0 = np.clip(x0, low, high)
    evaluations = 0

    def optics(x: np.ndarray) -> dict[str, OpticsResult]:
        nonlocal evaluations
        evaluations += 1
        for branch, session in sessions.items():
            session.update(
                {v: float(value) for v, value in zip(variables, x, strict=True) if v[0] in ids[branch]}
            )
        results: dict[str, OpticsResult] = {}
        for stretch in stretches:
            initial = stretch.initial
            if isinstance(initial, At):
                there = results[initial.stretch].at(initial.occurrence)
                initial = TwissInit(**(there.model_dump() | {"x": 0.0, "px": 0.0, "y": 0.0, "py": 0.0}))
            request = OpticsRequest(options=options, initial=initial, start=stretch.start, end=stretch.end)
            results[stretch.name] = sessions[stretch.branch].optics(request)
        return results

    def targets(stretch: Stretch, results: dict[str, OpticsResult]) -> dict[str, float]:
        if isinstance(stretch.targets, At):
            there = results[stretch.targets.stretch].at(stretch.targets.occurrence)
            return {k: getattr(there, k) for k in SAME}
        return dict(stretch.targets or {})

    def misses(results: dict[str, OpticsResult], x: np.ndarray) -> np.ndarray:
        out, limits_out = [], []
        for stretch in stretches:
            result = results[stretch.name]
            at = result.at(result.occurrences[-1])
            for key, target in targets(stretch, results).items():
                out.append((getattr(at, key) - target) / (TARGETS[key] or abs(target)))
            limit = stretch.beta_limit or beta_limit
            if limit is not None:
                peak = max(max(result.beta_x), max(result.beta_y))
                limits_out.append(max(0.0, peak - limit) / limit)
        regular = 1e-3 * (x - x0) / np.maximum(np.abs(x0), 1.0)  # stay close to the start
        return np.array([*out, *limits_out, *regular])

    count = 0

    def residuals(x: np.ndarray) -> np.ndarray:
        return misses(optics(x), x)

    def met(fit) -> bool:
        return bool(count == 0 or np.max(np.abs(fit.fun[:count])) < tolerance)

    try:
        before = optics(x0)
        count = sum(len(targets(s, before)) for s in stretches)
        best = SimpleNamespace(x=x0, fun=misses(before, x0))
        if not met(best):  # matched already: nothing to change
            best = least_squares(residuals, x0, bounds=(low, high), x_scale="jac", max_nfev=400)
        rng = np.random.default_rng(1)
        for _ in range(2):  # two more starts if the first ends short of the targets
            if met(best):
                break
            trial = least_squares(
                residuals, rng.uniform(low, high) * 0.5, bounds=(low, high), x_scale="jac", max_nfev=400
            )
            if trial.cost < best.cost:
                best = trial
        after = optics(best.x)
    finally:
        for session in sessions.values():
            session.close()

    out = {}
    for stretch in stretches:
        first = before[stretch.name].at(before[stretch.name].occurrences[-1])
        last = after[stretch.name].at(after[stretch.name].occurrences[-1])
        aim_before, aim = targets(stretch, before), targets(stretch, after)
        out[stretch.name] = StretchMatch(
            before=first,
            after=last,
            targets=aim,
            bmag_before=_bmag(first, aim_before),
            bmag_after=_bmag(last, aim),
            beta_max=(max(after[stretch.name].beta_x), max(after[stretch.name].beta_y)),
        )
    return Matched(
        values=dict(zip(variables, (float(v) for v in best.x), strict=True)),
        start=dict(zip(variables, (float(v) for v in x0), strict=True)),
        stretches=out,
        evaluations=evaluations,
        success=met(best),
    )


def _bmag(at: TwissInit, targets: Mapping[str, float]) -> tuple[float, float]:
    return tuple(
        mismatch(
            getattr(at, f"beta_{p}"), getattr(at, f"alpha_{p}"), targets[f"beta_{p}"], targets[f"alpha_{p}"]
        )
        if f"beta_{p}" in targets and f"alpha_{p}" in targets
        else float("nan")
        for p in "xy"
    )


def _design(lattice: Lattice, variable: tuple[str, str]) -> float:
    occurrence, parameter = variable
    group, name = parameter.split(".", 1)
    element = lattice.placement(occurrence).element
    return float(getattr(getattr(element, group), name, 0.0) or 0.0) if getattr(element, group) else 0.0
