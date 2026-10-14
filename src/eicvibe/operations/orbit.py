"""Orbit correction: least squares on the orbit response matrix, as rings and lines are corrected.

The response matrix R (BPMs × correctors, one plane) comes from the model of the
machine as it is set, by finite differences of the closed orbit (a ring) or the
trajectory (a line). The correction Δθ = -R⁺u is found by singular value
decomposition, its small singular values damped (Tikhonov) so that noise and
nearly degenerate corrector patterns do not ask for large strengths; the
damping grows until every corrector is within its limits. MICADO instead picks
the few correctors that do most, one at a time, to find a single error.

References: J. Wenninger, "Orbit steering and correction", CERN Accelerator
School (arXiv:2004.14001); Y. Chung, G. Decker, K. Evans, "Closed orbit
correction using singular value decomposition of the response matrix", PAC'93;
B. Autin, Y. Marti, "Closed orbit correction of A.G. machines using a small
number of magnets" (MICADO), CERN-ISR-MA/73-17.
"""

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .model import Knob, MachineModel
from .proposal import ChannelChange, Check, Proposal, headroom

REFERENCES = [
    "J. Wenninger, Orbit steering and correction, CAS (arXiv:2004.14001)",
    "Y. Chung, G. Decker, K. Evans, Closed orbit correction using SVD of the response matrix, PAC'93",
    "B. Autin, Y. Marti, MICADO, CERN-ISR-MA/73-17",
]


@dataclass(frozen=True)
class Response:
    """The orbit response in one plane: metres at each BPM per unit of each corrector's channel."""

    plane: str
    bpms: tuple[str, ...]
    correctors: tuple[Knob, ...]
    matrix: np.ndarray


_CACHE: OrderedDict[tuple, Response] = OrderedDict()
#: How far the full model's orbit change may be from the matrix's before a correction should be
#: applied in steps.
LINEAR = 0.1


def orbit_response(
    model: MachineModel,
    plane: str,
    correctors: Sequence[Knob] | None = None,
    at: Mapping[str, float] | None = None,
) -> Response:
    """The response of the orbit at the BPMs to each corrector of ``plane`` (all, by default).

    Kept for the machine as it is set (every value but the correctors'), as a control room keeps
    its model's matrix: it hardly depends on them. ``at`` (channel values on top) computes it
    afresh there instead, for an orbit so large that the sextupoles change it.
    """
    correctors = tuple(model.correctors[plane] if correctors is None else correctors)
    if at is not None:
        return _response(model, plane, correctors, dict(at))
    others = tuple(sorted((c, v) for c, v in model.live.items() if c not in {k.channel for k in correctors}))
    key = (
        model.lattice.digest, model.branch, model.engine, model.options.model_dump_json(), others, plane,
        tuple(k.channel for k in correctors),
    )  # fmt: skip
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]
    response = _response(model, plane, correctors, {})
    _CACHE[key] = response
    while len(_CACHE) > 8:
        _CACHE.popitem(last=False)
    return response


def _response(
    model: MachineModel, plane: str, correctors: tuple[Knob, ...], at: dict[str, float]
) -> Response:
    base = model.orbit(at)[plane]
    columns = []
    for knob in correctors:
        now = at.get(knob.channel, model.value(knob.channel))
        step = knob.step(now)
        columns.append((model.orbit(at | {knob.channel: now + step})[plane] - base) / step)
    return Response(plane, tuple(m.name for m in model.bpms), correctors, np.column_stack(columns))


def correct_orbit(
    model: MachineModel,
    plane: str,
    orbit: Mapping[str, float] | None = None,
    *,
    target: Mapping[str, float] | None = None,
    correctors: Sequence[str] | None = None,
    method: str = "svd",
    damping: float = 0.01,
    keep: int | None = None,
    count: int = 3,
) -> Proposal:
    """A correction of the orbit in ``plane`` (``x`` or ``y``) to ``target`` (zero by default).

    ``orbit`` is the orbit at the BPMs by name (m), as measured; BPMs missing from it are not
    used; by default it is the model's own (the machine as set, with its errors). ``method``:
    ``svd``, with the singular values damped by ``damping`` times the largest (or only the
    largest ``keep`` of them kept), the damping raised until every corrector is within its
    limits; or ``micado``, the ``count`` correctors that do most. ``correctors`` restricts the
    correctors (channel names) used.
    """
    if plane not in ("x", "y"):
        raise ValueError(f"plane must be x or y, not {plane!r}")
    knobs = model.correctors[plane]
    if correctors is not None:
        wanted = set(correctors)
        if unknown := wanted - {k.channel for k in knobs}:
            raise ValueError(f"not {plane} correctors of {model.name}: {sorted(unknown)}")
        knobs = [k for k in knobs if k.channel in wanted]
    if not knobs:
        raise ValueError(f"{model.name} has no {plane} correctors")
    response = orbit_response(model, plane, knobs)
    names = response.bpms
    if orbit is None:
        measured = model.orbit()[plane]
        source = "the model's orbit"
    else:
        if unknown := set(orbit) - set(names):
            raise ValueError(f"not BPMs of {model.name}: {sorted(unknown)[:5]}")
        measured = np.array([orbit.get(n, np.nan) for n in names], dtype=float)
        source = "the orbit given"
    aim = np.array([(target or {}).get(n, 0.0) for n in names], dtype=float)
    used = np.isfinite(measured)
    if used.sum() < 2:
        raise ValueError("fewer than two BPMs have readings")
    u = (measured - aim)[used]
    matrix = response.matrix[used]
    now = np.array([model.value(k.channel) for k in knobs])
    low = np.array([k.limits[0] if k.limits else -np.inf for k in knobs])
    high = np.array([k.limits[1] if k.limits else np.inf for k in knobs])

    if method == "micado":
        delta, chosen = _micado(matrix, u, count)
        parameters = {"method": "micado", "correctors": [knobs[j].channel for j in chosen]}

        def again(residual: np.ndarray, matrix: np.ndarray) -> np.ndarray:
            sol, *_ = np.linalg.lstsq(matrix[:, chosen], -residual, rcond=None)
            out = np.zeros(len(knobs))
            out[chosen] = sol
            return out
    elif method == "svd":
        lam = damping
        for _ in range(40):
            delta, kept = _svd(matrix, u, lam, keep)
            if np.all((now + delta >= low) & (now + delta <= high)):
                break
            lam *= 1.5  # damp more until the strengths fit
        parameters = {"method": "svd", "damping": lam, "singular_values_kept": kept}

        def again(residual: np.ndarray, matrix: np.ndarray) -> np.ndarray:
            return _svd(matrix, residual, lam, keep)[0]
    else:
        raise ValueError(f"method must be svd or micado, not {method!r}")

    def full(delta: np.ndarray) -> np.ndarray:
        """The orbit change the full model gives for ``delta``: the prediction."""
        values = {k.channel: float(v + d) for k, v, d in zip(knobs, now, delta, strict=True) if d != 0.0}
        return model.orbit(values)[plane] - model.orbit()[plane]

    linear = response.matrix @ delta
    moved = full(delta)
    steps = 1
    if orbit is None:
        # The model's own orbit: correct again what one linear step left (the sextupoles), as a
        # control room does, until it no longer helps.
        current, fresh = matrix, False
        while steps < 4:
            residual = (measured + moved - aim)[used]
            trial = delta + again(residual, current)
            ok = np.all((now + trial >= low) & (now + trial <= high))
            trial_moved = full(trial) if ok else None
            if not ok or _rms((measured + trial_moved - aim)[used]) >= 0.99 * _rms(residual):
                if fresh:
                    break
                # The orbit changed the response (sextupoles): compute it here, and try again.
                values = {k.channel: float(v + d) for k, v, d in zip(knobs, now, delta, strict=True)}
                current, fresh = orbit_response(model, plane, knobs, at=values).matrix[used], True
                parameters["response recomputed"] = True
                continue
            delta, moved, steps = trial, trial_moved, steps + 1
        linear = response.matrix @ delta
    parameters["steps"] = steps
    changes = [
        ChannelChange(
            channel=k.channel,
            device=k.device,
            unit=k.unit,
            before=float(v),
            after=float(v + d),
            limits=k.limits,
        )
        for k, v, d in zip(knobs, now, delta, strict=True)
        if d != 0.0
    ]
    predicted_after = measured + moved
    mismatch = float(np.sqrt(np.mean((moved - linear) ** 2)))
    scale = float(np.sqrt(np.mean(linear**2))) or 1.0
    before_rms, after_rms = _rms(measured[used] - aim[used]), _rms(predicted_after[used] - aim[used])
    checks = [
        headroom(changes),
        Check(
            name="orbit improves",
            ok=after_rms < before_rms,
            value=after_rms * 1e3,
            limit=before_rms * 1e3,
            note="rms distance from the target at the BPMs used, mm, after against before (full model)",
        ),
        Check(
            name="linear prediction holds",
            ok=mismatch <= LINEAR * scale + 1e-7,
            value=mismatch / scale,
            limit=LINEAR,
            note="how far the full model's orbit change is from the response matrix's, relative; when "
            "large, the orbit is too large for one linear step: apply, measure and correct again",
        ),
    ]
    unit = "mm"
    summary = (
        f"{'Horizontal' if plane == 'x' else 'Vertical'} orbit correction of {model.name} with "
        f"{len(changes)} correctors ({parameters['method']}): rms {before_rms * 1e3:.3f} -> "
        f"{after_rms * 1e3:.3f} {unit} at {int(used.sum())} BPMs, from {source}"
    )
    return Proposal(
        operation="orbit_correction",
        machine=model.name,
        summary=summary,
        changes=changes,
        predicted={
            "plane": plane,
            "unit": unit,
            "bpms": list(names),
            "used": [bool(b) for b in used],
            "before": _mm(measured),
            "after": _mm(predicted_after),
            "rms": [before_rms * 1e3, after_rms * 1e3],
            "max": [_max(measured[used] - aim[used]) * 1e3, _max(predicted_after[used] - aim[used]) * 1e3],
        },
        checks=checks,
        method=(
            "least squares on the model's orbit response matrix: SVD with Tikhonov damping"
            if parameters["method"] == "svd"
            else "MICADO: the correctors that do most, chosen one at a time, then exchanged while that "
            "does better"
        ),
        parameters=parameters,
        engine=model.engine,
        references=REFERENCES,
    )


def _svd(matrix: np.ndarray, u: np.ndarray, damping: float, keep: int | None) -> tuple[np.ndarray, int]:
    left, s, right = np.linalg.svd(matrix, full_matrices=False)
    if keep is not None:
        s_used = np.where(np.arange(len(s)) < keep, s, 0.0)
        filt = np.divide(1.0, s_used, out=np.zeros_like(s), where=s_used > 0)
        kept = int(min(keep, len(s)))
    else:
        lam = damping * s[0]
        filt = s / (s**2 + lam**2)  # Tikhonov: small singular values damped, not cut
        kept = int(np.sum(s > lam))
    return -(right.T @ (filt * (left.T @ u))), kept


def _micado(matrix: np.ndarray, u: np.ndarray, count: int) -> tuple[np.ndarray, list[int]]:
    """MICADO: the corrector that does most is chosen, then the next with it, up to ``count``; then
    each chosen one is exchanged for another while that does better (greedy choice alone can miss
    the pair that explains two errors)."""

    def residual(chosen: list[int]) -> float:
        cols = matrix[:, chosen]
        sol, *_ = np.linalg.lstsq(cols, -u, rcond=None)
        return float(np.sum((u + cols @ sol) ** 2))

    count = min(count, matrix.shape[1])
    chosen: list[int] = []
    for _ in range(count):
        rest = [j for j in range(matrix.shape[1]) if j not in chosen]
        chosen.append(min(rest, key=lambda j: residual([*chosen, j])))
    best = residual(chosen)
    improved = True
    while improved:
        improved = False
        for i in range(len(chosen)):
            for j in range(matrix.shape[1]):
                if j in chosen:
                    continue
                trial = [*chosen[:i], j, *chosen[i + 1 :]]
                if (r := residual(trial)) < best * (1 - 1e-9):
                    chosen, best, improved = trial, r, True
    sol, *_ = np.linalg.lstsq(matrix[:, chosen], -u, rcond=None)
    delta = np.zeros(matrix.shape[1])
    delta[chosen] = sol
    return delta, chosen


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values**2))) if len(values) else 0.0


def _max(values: np.ndarray) -> float:
    return float(np.max(np.abs(values))) if len(values) else 0.0


def _mm(values: np.ndarray) -> list[float | None]:
    return [None if not np.isfinite(v) else float(v * 1e3) for v in values]
