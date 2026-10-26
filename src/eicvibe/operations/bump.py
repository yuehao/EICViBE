"""Closed local orbit bumps: an orbit offset (and angle) at a point that stays inside a few correctors.

Three correctors give a closed bump of a chosen amplitude: in a linear machine
their kicks follow from the beta functions and phase advances,
θ₂/θ₁ = −√(β₁/β₂)·sin(φ₃−φ₁)/sin(φ₃−φ₂), θ₃/θ₁ = √(β₁/β₃)·sin(φ₂−φ₁)/sin(φ₃−φ₂);
four give the offset and the angle at a point. Here the kicks are solved on
the model's own orbit response (so that thick elements, coupling and the
machine as it is set count): the offset (and angle) at the point, and zero
orbit and angle after the last corrector, so that the orbit everywhere else is
unchanged. Of the correctors around the point, the set that needs the least of
their range is chosen, the point between the first and the last; the bump is
then checked in the model: its amplitude, and the orbit change at every BPM
outside it (its leak).

References: D. Rubin, G. Dugan, USPAS lecture 8 (three-bump); J. Wenninger,
"Orbit steering and correction", CAS (arXiv:2004.14001); AT's BumpAtBPM4D;
NSLS-II local bumps in the slow orbit feedback (NAPAC'22 WEPA71).
"""

import itertools

import numpy as np

from eicvibe.engines.contracts import Effect

from .model import Knob, MachineModel
from .proposal import ChannelChange, Check, Proposal, headroom

REFERENCES = [
    "G. Dugan, USPAS lecture 8, closed orbit bumps (classe.cornell.edu/~dugan/USPAS/Lect8.pdf)",
    "J. Wenninger, Orbit steering and correction, CAS (arXiv:2004.14001)",
    "AT (Accelerator Toolbox), BumpAtBPM4D",
    "NSLS-II local bumps in the slow orbit feedback, NAPAC'22 WEPA71",
]
#: How many correctors on each side of the point are tried.
NEAREST = 3


def local_bump(
    model: MachineModel,
    plane: str,
    at: str,
    amplitude: float,
    angle: float | None = None,
    *,
    correctors: list[str] | None = None,
) -> Proposal:
    """A closed bump in ``plane`` of ``amplitude`` (m) at ``at`` (a BPM's name or an occurrence),
    and with ``angle`` (rad) there if given (four correctors, else three).

    ``correctors`` names the channels to use (three or four); by default they are chosen among
    the nearest on each side.
    """
    if plane not in ("x", "y"):
        raise ValueError(f"plane must be x or y, not {plane!r}")
    point, label = _point(model, at)
    # In a ring whose RF holds the energy (6D), a horizontal bump that lengthens the path changes the
    # beam's energy, and the orbit everywhere by the dispersion times it: one more corrector keeps
    # the energy (the path length, sum of kick times dispersion) as it was.
    energy = plane == "x" and model.periodic and Effect.LONGITUDINAL in model.options.effects
    count = (3 if angle is None else 4) + (1 if energy else 0)
    knobs = model.correctors[plane]
    base = model.orbit_result()
    index = {o: i for i, o in enumerate(base.occurrences)}
    length = model.layout.length
    s0 = model.position[point]

    def ahead(s: float) -> float:
        """Distance downstream of the point (a ring wraps)."""
        return (s - s0) % length if model.periodic else s - s0

    if correctors is not None:
        by_name = {k.channel: k for k in knobs}
        if missing := [c for c in correctors if c not in by_name]:
            raise ValueError(f"not {plane} correctors of {model.name}: {missing}")
        if len(correctors) != count:
            raise ValueError(
                f"a bump {'with an angle ' if angle is not None else ''}"
                f"{'keeping the energy ' if energy else ''}takes {count} correctors"
            )
        candidates = [tuple(by_name[c] for c in correctors)]
    else:
        up = sorted((k for k in knobs if _upstream(k, s0, model)), key=lambda k: -ahead(k.s))[:NEAREST]
        down = sorted((k for k in knobs if not _upstream(k, s0, model)), key=lambda k: ahead(k.s))[:NEAREST]
        candidates = [
            c
            for c in itertools.combinations(up + down, count)
            if any(k in up for k in c) and any(k in down for k in c)
        ]
    if not candidates:
        raise ValueError(f"no {plane} correctors on both sides of {label} in {model.name}")

    columns: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def column(knob: Knob):
        if knob.channel not in columns:
            now = model.value(knob.channel)
            step = knob.step(now)
            moved = model.orbit_result({knob.channel: now + step})
            columns[knob.channel] = tuple(
                (np.asarray(getattr(moved, c)) - np.asarray(getattr(base, c))) / step
                for c in (plane, "p" + plane, "delta")
            )
        return columns[knob.channel]

    def conditions(pos, ang, dlt, last: str) -> list[float]:
        """What the bump sets: the offset (and angle) at the point; zero orbit and angle after the last
        corrector; and in a ring with RF, the energy unchanged."""
        return [
            pos[index[point]],
            *([] if angle is None else [ang[index[point]]]),
            pos[index[last]],
            ang[index[last]],
            *([dlt[index[last]]] if energy else []),
        ]

    def order(knob: Knob) -> float:
        """Position in beam order from the first candidate upstream: the bump's first is smallest."""
        return ahead(knob.s) if not _upstream(knob, s0, model) else ahead(knob.s) - length

    wanted = [amplitude] + ([] if angle is None else [angle]) + [0.0, 0.0] + ([0.0] if energy else [])
    best = None
    for chosen in candidates:
        chosen = tuple(sorted(chosen, key=order))
        last = chosen[-1].occurrences[-1]
        rows = []
        for knob in chosen:
            rows.append(conditions(*column(knob), last))
        matrix = np.array(rows).T
        if np.linalg.cond(matrix) > 1e8:
            continue  # the correctors cannot make this bump (a phase advance near a multiple of pi)
        delta = np.linalg.solve(matrix, wanted)
        use = max(abs(d) / _half(k) for d, k in zip(delta, chosen, strict=True))
        if best is None or use < best[0]:
            best = (use, chosen, delta, matrix)
    if best is None:
        raise ValueError(f"no set of {count} correctors around {label} can make the bump")
    _, chosen, delta, matrix = best

    # The bump in the full model, its amplitude made exact by one more step if needed.
    def measure(values: dict[str, float]):
        result = model.orbit_result(values)
        pos = np.asarray(getattr(result, plane)) - np.asarray(getattr(base, plane))
        ang = np.asarray(getattr(result, "p" + plane)) - np.asarray(getattr(base, "p" + plane))
        dlt = np.asarray(result.delta) - np.asarray(base.delta)
        return result, pos, ang, dlt

    now = {k.channel: model.value(k.channel) for k in chosen}
    last = chosen[-1].occurrences[-1]
    # Each condition is judged by the orbit it moves: an offset as it is, an angle over ten metres (a
    # betatron length), the energy through the ring's largest dispersion; met within 1e-4 of the bump.
    # A further step is kept only if it brings the bump closer: where the full model departs from the
    # response (the sextupoles, or an energy residual the correctors barely move), stepping on would
    # chase it with ever larger kicks until the beam is lost.
    reach = max(1e-9, 1e-4 * abs(amplitude))
    dispersion = max(1e-3, float(np.max(np.abs(model.optics().eta_x)))) if energy else 1.0
    scale = np.array([1.0, *([] if angle is None else [10.0]), 1.0, 10.0, *([dispersion] if energy else [])])

    def miss(values: dict[str, float]):
        measured = measure(values)
        error = np.array(wanted) - np.array(conditions(*measured[1:], last))
        return measured, error, float(np.max(np.abs(error * scale)))

    values = {k.channel: now[k.channel] + d for k, d in zip(chosen, delta, strict=True)}
    (_, pos, ang, dlt), error, worst = miss(values)
    for _ in range(3):
        if worst <= reach:
            break
        step = delta + np.linalg.solve(matrix, error)
        trial = {k.channel: now[k.channel] + d for k, d in zip(chosen, step, strict=True)}
        try:
            measured, trial_error, trial_worst = miss(trial)
        except Exception:  # noqa: BLE001 - the engine lost the beam: the step went too far
            break
        if trial_worst >= worst:
            break  # no closer: the values measured before are the ones proposed
        delta, values, error, worst = step, trial, trial_error, trial_worst
        _, pos, ang, dlt = measured

    changes = [
        ChannelChange(
            channel=k.channel,
            device=k.device,
            unit=k.unit,
            before=now[k.channel],
            after=float(now[k.channel] + d),
            limits=k.limits,
        )
        for k, d in zip(chosen, delta, strict=True)
    ]
    bpm_change = model.orbit(values)[plane] - model.orbit()[plane]
    first_s, last_s = chosen[0].s, model.position[last]
    inside = np.array([_between(m.s, first_s, last_s, model) for m in model.bpms])
    leak = float(np.max(np.abs(bpm_change[~inside]))) if (~inside).any() else 0.0
    tolerance = max(1e-6, 0.01 * abs(amplitude))
    checks = [
        Check(
            name="amplitude reached",
            ok=abs(pos[index[point]] - amplitude) <= tolerance,
            value=float(pos[index[point]] * 1e3),
            limit=amplitude * 1e3,
            note=f"mm at {label}",
        ),
        Check(
            name="closed: no orbit change outside",
            ok=leak <= tolerance,
            value=leak * 1e3,
            limit=tolerance * 1e3,
            note=f"largest orbit change at the {int((~inside).sum())} BPMs outside the bump, mm",
        ),
        headroom(changes),
    ]
    if angle is not None:
        checks.insert(
            1,
            Check(
                name="angle reached",
                ok=abs(ang[index[point]] - angle) <= max(1e-7, 0.01 * abs(angle)),
                value=float(ang[index[point]] * 1e3),
                limit=angle * 1e3,
                note=f"mrad at {label}",
            ),
        )
    plane_word = "horizontal" if plane == "x" else "vertical"
    summary = (
        f"A closed {plane_word} bump of {amplitude * 1e3:+.3f} mm"
        + ("" if angle is None else f" and {angle * 1e3:+.3f} mrad")
        + f" at {label} in {model.name}, with {count} correctors: "
        + ", ".join(f"{c.device} {c.change:+.4g} {c.unit}" for c in changes)
    )
    return Proposal(
        operation="local_bump",
        machine=model.name,
        summary=summary,
        changes=changes,
        predicted={
            "plane": plane,
            "at": label,
            "unit": "mm",
            "amplitude": float(pos[index[point]] * 1e3),
            "angle_mrad": float(ang[index[point]] * 1e3),
            "from": chosen[0].device,
            "to": chosen[-1].device,
            "leak": leak * 1e3,
            "bpms": [m.name for m in model.bpms],
            "change": [float(v * 1e3) for v in bpm_change],
            "inside": [bool(b) for b in inside],
        },
        checks=checks,
        method=(
            f"closed {count}-corrector bump solved on the model's orbit response: the offset"
            + ("" if angle is None else " and angle")
            + " at the point, zero orbit and angle after the last corrector"
            + (
                "; the beam's energy unchanged (the RF holds it: a longer path would change it)"
                if energy
                else ""
            )
            + "; of the correctors around the point, the set that uses the least of their range"
        ),
        parameters={"correctors": [k.channel for k in chosen], "count": count},
        engine=model.engine,
        references=REFERENCES,
    )


def _point(model: MachineModel, at: str) -> tuple[str, str]:
    """The occurrence a bump is at, and how to call it: a BPM's name, or an occurrence's."""
    for m in model.bpms:
        if at == m.name:
            return m.occurrence, m.name
    try:
        placed = model.lattice.placement(at, branch=model.branch)
    except KeyError:
        names = ", ".join(m.name for m in model.bpms[:5])
        raise ValueError(f"no BPM or element {at!r} in {model.name}; BPMs are e.g. {names}") from None
    return placed.id, placed.name


def _upstream(knob: Knob, s0: float, model: MachineModel) -> bool:
    """A corrector before the point (in a ring: in the half turn before it)."""
    if not model.periodic:
        return knob.s <= s0
    return 0 < (s0 - knob.s) % model.layout.length <= model.layout.length / 2 or knob.s == s0


def _between(s: float, first: float, last: float, model: MachineModel) -> bool:
    if not model.periodic or first <= last:
        return first <= s <= last
    return s >= first or s <= last  # the bump crosses the ring's start


def _half(knob: Knob) -> float:
    return (knob.limits[1] - knob.limits[0]) / 2 if knob.limits else 1.0
