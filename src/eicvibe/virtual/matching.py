"""Bunches matched to the optics: a ring's one-turn map, or a line's initial optics.

A ring's bunch must repeat itself turn after turn: its second moments are those
of the ring's periodic solution where it is injected. :func:`ring_matching`
finds them from the ring's own one-turn map, the linear map about the closed
orbit that the engine tracks (finite differences of one turn, so it is exactly
what the run will track: the same engine, fidelity, settings and effects). Its
normal modes give the beta functions, alpha and dispersion, coupling
included, and with RF the synchrotron motion: the bunch length follows from the
momentum spread, as a stored bunch's does. Without RF (or with the
longitudinal effect off) the momentum is fixed turn by turn and the bunch
length is free; transversely the bunch is matched about the off-momentum
closed orbit, through the dispersion.

A single pass has no periodic solution: its bunch is matched to the optics
where it starts (:func:`twiss_bunch`), its line's initial conditions or the
machine's optics there.
"""

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from eicvibe.engines.contracts import Effect, Options

#: Offsets of the finite differences: small enough to be linear, large enough above round-off.
STEP = 1e-8


def one_turn_matrix(session, closed: np.ndarray, options: Options, step: float = STEP) -> np.ndarray:
    """The ring's linear one-turn map about ``closed`` (x, px, y, py, zeta, delta), by central differences."""
    offsets = np.vstack([np.eye(6) * step, -np.eye(6) * step])
    track = session.track_beam(closed + offsets, 1, (), options)
    if np.asarray(track.lost).any():
        raise ValueError("particles near the closed orbit are lost in one turn: the ring holds no beam")
    final = np.asarray(track.final, dtype=float)
    return (final[:6] - final[6:]).T / (2 * step)


@dataclass(frozen=True)
class Matching:
    """How a bunch fits the optics where it is injected.

    ``W`` turns normalized coordinates into ``(x, px, y, py, zeta, delta)``: each
    mode's pair of columns carries its emittance. Without synchrotron motion
    only the transverse modes are columns of ``W`` and ``dispersion`` gives the
    transverse orbit per unit delta.
    """

    closed: np.ndarray
    W: np.ndarray
    tunes: tuple[float, ...]
    longitudinal: bool
    dispersion: np.ndarray = field(default_factory=lambda: np.zeros(4))
    to: str = ""

    def bunch(self, rng: np.random.Generator, beam, centre: np.ndarray | None = None) -> np.ndarray:
        """A Gaussian bunch of ``beam`` (its emittances and momentum spread), centred exactly."""
        n = beam.particles
        u = rng.standard_normal((6, n))
        u = u - u.mean(axis=1, keepdims=True) if n > 1 else np.zeros_like(u)
        centre = self.closed if centre is None else np.asarray(centre, dtype=float)
        if self.longitudinal:
            scale = np.sqrt(np.repeat(self.emittances(beam), 2))
            return centre + (self.W @ (scale[:, None] * u)).T
        delta = beam.sigma_delta * u[4]
        transverse = self.W[:4, :4] @ (
            np.sqrt(np.repeat([beam.emittance_x, beam.emittance_y], 2))[:, None] * u[:4]
        )
        out = np.empty((n, 6))
        out[:, :4] = (transverse + np.outer(self.dispersion, delta)).T
        out[:, 4] = beam.sigma_zeta * u[5]
        out[:, 5] = delta
        return centre + out

    def emittances(self, beam) -> np.ndarray:
        """The three modes' emittances: the beam's transverse ones, and the longitudinal one its momentum
        spread gives (zero without synchrotron motion)."""
        if not self.longitudinal:
            return np.array([beam.emittance_x, beam.emittance_y, 0.0])
        transverse = np.array([beam.emittance_x, beam.emittance_y])
        from_transverse = float(transverse @ (self.W[5, 0:4:2] ** 2 + self.W[5, 1:4:2] ** 2))
        weight = self.W[5, 4] ** 2 + self.W[5, 5] ** 2
        return np.array([*transverse, max(beam.sigma_delta**2 - from_transverse, 0.0) / weight])

    def sigma_zeta(self, beam) -> float:
        """The matched bunch length (m), or the beam's own without synchrotron motion."""
        if not self.longitudinal:
            return beam.sigma_zeta
        e = self.emittances(beam)
        return math.sqrt(float(e @ (self.W[4, 0::2] ** 2 + self.W[4, 1::2] ** 2)))

    def twiss(self) -> dict[str, float]:
        """Beta, alpha and dispersion where the bunch is injected (the modes' x and y projections)."""
        W = self.W
        if self.longitudinal:
            eta = W[:4, 4:6] @ np.linalg.pinv(W[5:6, 4:6])  # the orbit that goes with delta in mode III
            eta = eta[:, 0]
        else:
            eta = self.dispersion
        return {
            "beta_x": W[0, 0] ** 2 + W[0, 1] ** 2,
            "alpha_x": -(W[0, 0] * W[1, 0] + W[0, 1] * W[1, 1]),
            "beta_y": W[2, 2] ** 2 + W[2, 3] ** 2,
            "alpha_y": -(W[2, 2] * W[3, 2] + W[2, 3] * W[3, 3]),
            "eta_x": float(eta[0]),
            "etap_x": float(eta[1]),
            "eta_y": float(eta[2]),
            "etap_y": float(eta[3]),
        }

    def summary(self, beam) -> dict[str, Any]:
        """What the bunch is matched to, for run.json and events."""
        twiss = {k: round(float(v), 9) for k, v in self.twiss().items()}
        out: dict[str, Any] = {"to": self.to, **twiss, "tunes": [round(q, 9) for q in self.tunes[:2]]}
        if (
            self.longitudinal
        ):  # its sense of rotation flips at transition: the tune is the smaller of q, 1 - q
            out["synchrotron_tune"] = round(min(self.tunes[2], 1.0 - self.tunes[2]), 9)
        out["sigma_zeta"] = self.sigma_zeta(beam)
        out["emittances"] = self.emittances(beam).tolist()
        return out


def _modes(M: np.ndarray) -> tuple[np.ndarray, list[float]]:
    """Normal modes of a stable symplectic map: W with W^T S W = S, ordered by plane, and their tunes."""
    n = M.shape[0]
    S = np.kron(np.eye(n // 2), np.array([[0.0, 1.0], [-1.0, 0.0]]))
    values, vectors = np.linalg.eig(M)
    if np.any(np.abs(np.abs(values) - 1.0) > 1e-6) or np.any(np.abs(values.imag) < 1e-12):
        raise ValueError(f"the one-turn map is not stable: eigenvalues {np.round(values, 6).tolist()}")
    columns, tunes, planes = [], [], []
    for value, vector in zip(values, vectors.T, strict=True):
        if value.imag <= 0:
            continue
        a, b = vector.real, vector.imag
        omega = a @ S @ b
        q = math.atan2(value.imag, value.real) / (2 * math.pi)
        if omega < 0:  # the other of the pair turns the right way
            b, q = -b, 1.0 - q
        a, b = a / math.sqrt(abs(omega)), b / math.sqrt(abs(omega))
        weight = [
            (a[2 * k] ** 2 + a[2 * k + 1] ** 2 + b[2 * k] ** 2 + b[2 * k + 1] ** 2) for k in range(n // 2)
        ]
        columns.append((a, b))
        tunes.append(q % 1.0)
        planes.append(int(np.argmax(weight)))
    if sorted(planes) != list(range(n // 2)):
        raise ValueError("the one-turn map's modes cannot be told apart by plane")
    order = np.argsort(planes)
    W = np.column_stack([c for k in order for c in columns[k]])
    return W, [tunes[k] for k in order]


def ring_matching(session, closed: np.ndarray, options: Options, where: str = "") -> Matching:
    """The bunch matching of a ring where its turns start, from its one-turn map about ``closed``.

    With the longitudinal effect and RF on its stable side, from the 6D map: three normal modes.
    Otherwise (no RF, the effect off, or RF that holds no bunch) from the map with RF off: two
    transverse modes and the dispersion. A 6D map's transverse block alone is not a map (the RF
    couples x, through the path length, to the momentum), so it is never used by itself.
    """
    to = f"the one-turn map at {where}".strip()
    note = ", 4D"
    if Effect.LONGITUDINAL in options.effects:
        M = one_turn_matrix(session, closed, options)
        # Synchrotron motion turns the longitudinal pair: 2 cos(2 pi Qs) < 2. The margin keeps the
        # noise of finite differences (1e-8) from making a coasting beam's (1, 1) look like one.
        if abs(M[4, 4] + M[5, 5]) < 2.0 - 1e-6:
            try:
                W, tunes = _modes(M)
                return Matching(closed, W, tuple(tunes), True, to=f"{to}, 6D")
            except ValueError:  # simulated as configured all the same (ADR 0007)
                note = ", 4D: the RF holds no bunch"
    flat = Options(**(options.model_dump() | {"effects": options.effects - {Effect.LONGITUDINAL}}))
    M = one_turn_matrix(session, closed, flat)
    W4, tunes = _modes(M[:4, :4])  # a ring without stable transverse optics holds no beam: this raises
    dispersion = np.linalg.solve(np.eye(4) - M[:4, :4], M[:4, 5])
    W = np.eye(6)
    W[:4, :4] = W4
    return Matching(closed, W, tuple(tunes), False, dispersion, to + note)


def twiss_bunch(rng: np.random.Generator, beam, optics, centre: np.ndarray) -> np.ndarray:
    """A Gaussian bunch matched to Twiss values (a TwissInit) about ``centre`` (x, px, y, py, zeta, delta).

    Centred, so its centroid is ``centre`` exactly; one particle is the centroid alone.
    """
    n = beam.particles
    u = rng.standard_normal((6, n))
    u = u - u.mean(axis=1, keepdims=True) if n > 1 else np.zeros_like(u)
    t = optics
    delta = beam.sigma_delta * u[4]
    ex, ey = math.sqrt(beam.emittance_x), math.sqrt(beam.emittance_y)
    x = ex * math.sqrt(t.beta_x) * u[0] + t.eta_x * delta
    px = ex / math.sqrt(t.beta_x) * (u[1] - t.alpha_x * u[0]) + t.etap_x * delta
    y = ey * math.sqrt(t.beta_y) * u[2] + t.eta_y * delta
    py = ey / math.sqrt(t.beta_y) * (u[3] - t.alpha_y * u[2]) + t.etap_y * delta
    zeta = beam.sigma_zeta * u[5]
    return np.asarray(centre, dtype=float) + np.column_stack([x, px, y, py, zeta, delta])
