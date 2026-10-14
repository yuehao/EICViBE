"""Tune moves and corrections with quadrupole knobs, disturbing the optics as little as they can.

A quadrupole's strength change Δk over a length l moves the tunes by
ΔQ ≈ ±βΔk·l/4π. With two families the 2×2 tune response is inverted; with more
quadrupole channels (strings and trims) there are many ways to the same tune
change, and the one chosen beats the beta functions and moves the dispersion
least: the tune response is met exactly while the beta-beating (Δβ/β, both
planes, at every element) and the dispersion change are least squares, as
trim-quadrupole tune knobs are built (the LHC's arc trims change the tunes
"with a minimum impact on the beta-beating"). The responses come from the model
of the machine as it is set, by finite differences; a few Newton steps in the
full model make the tunes exact. The chromaticity's change is reported.

References: J. Wenninger, CAS (arXiv:2004.14001), eqs. 29-35; W. Herr et al.
on LHC tune knobs (arXiv:1410.7055); M. Minty, F. Zimmermann, "Measurement and
Control of Charged Particle Beams" (Springer, 2003), ch. 2.
"""

from collections.abc import Sequence

import numpy as np

from .model import MachineModel
from .proposal import ChannelChange, Check, Proposal, headroom

REFERENCES = [
    "J. Wenninger, Orbit steering and correction / optics control, CAS (arXiv:2004.14001), eqs. 29-35",
    "LHC tune knobs with arc trim quadrupoles, minimum beta-beating (arXiv:1410.7055)",
    "M. Minty, F. Zimmermann, Measurement and Control of Charged Particle Beams, Springer 2003",
]
#: How much a dispersion change weighs against beta-beating: 1 cm like 1 %.
DISPERSION_SCALE = 1.0
#: The beta-beating, and the dispersion change against the ring's largest dispersion, a tune move
#: should stay within.
BEAT_LIMIT, DISPERSION_LIMIT = 0.05, 0.05


def move_tunes(
    model: MachineModel,
    dqx: float,
    dqy: float,
    *,
    quadrupoles: Sequence[str] | None = None,
) -> Proposal:
    """A change of the tunes by (``dqx``, ``dqy``) with the quadrupole channels (all of the ring's,
    or ``quadrupoles``), beating the beta functions and moving the dispersion least."""
    if not model.periodic:
        raise ValueError(f"{model.name} is a line: it has no tunes")
    knobs = model.quadrupoles
    if quadrupoles is not None:
        wanted = set(quadrupoles)
        if unknown := wanted - {k.channel for k in knobs}:
            raise ValueError(f"not quadrupole channels of {model.name}: {sorted(unknown)}")
        knobs = [k for k in knobs if k.channel in wanted]
    if len(knobs) < 2:
        raise ValueError(f"{model.name} has fewer than two quadrupole channels to move both tunes with")
    base = model.optics()
    now = np.array([model.value(k.channel) for k in knobs])
    steps = np.array([k.step(v) for k, v in zip(knobs, now, strict=True)])
    bx, by, eta = (np.asarray(getattr(base, c)) for c in ("beta_x", "beta_y", "eta_x"))

    tunes, rows = [], []
    for knob, v, h in zip(knobs, now, steps, strict=True):
        moved = model.optics({knob.channel: v + h})
        tunes.append([(moved.tune_x - base.tune_x) / h, (moved.tune_y - base.tune_y) / h])
        rows.append(
            np.concatenate(
                [
                    (np.asarray(moved.beta_x) / bx - 1) / h,
                    (np.asarray(moved.beta_y) / by - 1) / h,
                    (np.asarray(moved.eta_x) - eta) / DISPERSION_SCALE / h,
                ]
            )
        )
    jacobian = np.array(tunes).T  # 2 × n, tune per channel unit
    disturb = np.array(rows).T / np.sqrt(len(bx))  # rms over the ring
    scale = steps  # in units of each channel's step, so that the problem is well conditioned

    def solve(dq: np.ndarray) -> np.ndarray:
        j, m = jacobian * scale, disturb * scale
        if len(knobs) == 2:
            z = np.linalg.solve(j, dq)
        else:
            hessian = m.T @ m + 1e-9 * np.trace(m.T @ m) / len(knobs) * np.eye(len(knobs))
            kkt = np.block([[hessian, j.T], [j, np.zeros((2, 2))]])
            z = np.linalg.solve(kkt, np.concatenate([np.zeros(len(knobs)), dq]))[: len(knobs)]
        return z * scale

    target = np.array([dqx, dqy])
    delta = solve(target)
    for _ in range(5):  # Newton steps in the full model: the tunes made exact
        changes = {k.channel: float(v + d) for k, v, d in zip(knobs, now, delta, strict=True)}
        after = model.optics(changes)
        reached = np.array([after.tune_x - base.tune_x, after.tune_y - base.tune_y])
        if np.max(np.abs(target - reached)) < 1e-6:
            break
        delta = delta + solve(target - reached)
    changes = {k.channel: float(v + d) for k, v, d in zip(knobs, now, delta, strict=True)}
    after = model.optics(changes)
    reached = np.array([after.tune_x - base.tune_x, after.tune_y - base.tune_y])
    beat_x = float(np.max(np.abs(np.asarray(after.beta_x) / bx - 1)))
    beat_y = float(np.max(np.abs(np.asarray(after.beta_y) / by - 1)))
    deta = float(np.max(np.abs(np.asarray(after.eta_x) - eta)))
    largest = float(np.max(np.abs(eta))) or 1.0
    proposed = [
        ChannelChange(
            channel=k.channel,
            device=k.device,
            unit=k.unit,
            before=float(v),
            after=float(v + d),
            limits=k.limits,
        )
        for k, v, d in zip(knobs, now, delta, strict=True)
        if abs(d) > 1e-12 * max(abs(v), 1.0)
    ]
    checks = [
        Check(
            name="tunes reached",
            ok=bool(np.max(np.abs(target - reached)) <= 1e-4),
            value=float(np.max(np.abs(target - reached))),
            limit=1e-4,
            note="largest difference from the tune change asked for",
        ),
        Check(
            name="beta-beating",
            ok=max(beat_x, beat_y) <= BEAT_LIMIT,
            value=max(beat_x, beat_y),
            limit=BEAT_LIMIT,
            note=f"largest Δβ/β anywhere: x {beat_x:.2%}, y {beat_y:.2%}",
        ),
        Check(
            name="dispersion kept",
            ok=deta <= DISPERSION_LIMIT * largest,
            value=deta / largest,
            limit=DISPERSION_LIMIT,
            note=f"largest change of the horizontal dispersion, {deta * 1e3:.1f} mm, against its largest, "
            f"{largest:.3f} m",
        ),
        headroom(proposed),
    ]
    summary = (
        f"Tunes of {model.name} moved by ({dqx:+.4f}, {dqy:+.4f}) to ({after.tune_x:.4f}, "
        f"{after.tune_y:.4f}) with {len(proposed)} quadrupole channels; beta-beating at most "
        f"{max(beat_x, beat_y):.2%}, "
        f"dispersion changed by at most {deta * 1e3:.2f} mm; chromaticity ({base.chroma_x:+.2f}, "
        f"{base.chroma_y:+.2f}) -> ({after.chroma_x:+.2f}, {after.chroma_y:+.2f})"
    )
    return Proposal(
        operation="tune_move",
        machine=model.name,
        summary=summary,
        changes=proposed,
        predicted={
            "tunes": [[base.tune_x, base.tune_y], [after.tune_x, after.tune_y]],
            "chromaticity": [[base.chroma_x, base.chroma_y], [after.chroma_x, after.chroma_y]],
            "beta_beating": [beat_x, beat_y],
            "dispersion_change_mm": deta * 1e3,
            "tune_response": {
                k.channel: list(map(float, col)) for k, col in zip(knobs, jacobian.T, strict=True)
            },
        },
        checks=checks,
        method=(
            "the two quadrupole families' tune response inverted"
            if len(knobs) == 2
            else "the tune response met exactly with the least beta-beating and dispersion change "
            "(constrained least squares over the quadrupole channels)"
        )
        + ", then Newton steps in the full model",
        parameters={"quadrupoles": [k.channel for k in knobs]},
        engine=model.engine,
        references=REFERENCES,
    )
