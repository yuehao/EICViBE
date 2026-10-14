"""What every engine must do (ADR 0006): one standard, the same for Xsuite, TrackPad and any engine to come.

Each test runs for every registered engine (``ENGINES``) that is installed here, against
references that do not come from an engine: a ring and a line of FODO cells of thick
quadrupoles and drifts, whose optics the transfer matrices give exactly, and the
contract's own conventions (a positive Kn0L deflects toward -x, a positive Ks0L toward
+y; a session updated in place is the session opened with those values; a ramp leaves
the settings at its end). An engine joins ``ENGINES`` when it passes. Comparisons of
engines with each other, and with MAD-X, are in the engines' own tests.
"""

import math

import numpy as np
import pytest

from eicvibe.diagnostics.analysis import tune_of
from eicvibe.domain.lattice import LatticeBuilder, SequentialIds
from eicvibe.engines import ENGINES, EngineAdapter, EngineSession, engine
from eicvibe.engines.contracts import (
    EngineUnavailable,
    Fidelity,
    OpticsRequest,
    Options,
    OrbitRequest,
    Ramp,
    TrackRequest,
    TwissInit,
    UnsupportedRequest,
)
from eicvibe.engines.model import SETTINGS

PRECISE = Options(fidelity=Fidelity.PRECISE)
CELLS = 6
L_QUAD, K1, L_DRIFT = 0.5, 0.7, 2.0
KICK = 1e-5


# -- the reference: FODO cells, exactly ------------------------------------------------------------


def quad(k: float, length: float) -> np.ndarray:
    if k > 0:
        w = math.sqrt(k)
        return np.array(
            [
                [math.cos(w * length), math.sin(w * length) / w],
                [-w * math.sin(w * length), math.cos(w * length)],
            ]
        )
    w = math.sqrt(-k)
    return np.array(
        [
            [math.cosh(w * length), math.sinh(w * length) / w],
            [w * math.sinh(w * length), math.cosh(w * length)],
        ]
    )


def drift(length: float) -> np.ndarray:
    return np.array([[1.0, length], [0.0, 1.0]])


def cell(plane: str) -> np.ndarray:
    """A cell's matrix, QF first: horizontally focusing then defocusing, vertically the other way."""
    k = K1 if plane == "x" else -K1
    return drift(L_DRIFT) @ quad(-k, L_QUAD) @ drift(L_DRIFT) @ quad(k, L_QUAD)


def periodic(plane: str) -> tuple[float, float, float]:
    """A cell's phase advance (rad), and beta and alpha where it starts."""
    m = cell(plane)
    mu = math.acos(0.5 * np.trace(m))
    if m[0, 1] < 0:
        mu = 2 * math.pi - mu
    return mu, m[0, 1] / math.sin(mu), (m[0, 0] - m[1, 1]) / (2 * math.sin(mu))


@pytest.fixture(scope="module")
def fodo():
    """A ring and a line of the same cells; a corrector and a marker in the first cell."""
    b = LatticeBuilder("conformance", ids=SequentialIds())
    b.define("BEG", "BeginningEle", ReferenceP={"species_ref": "electron", "pc_ref": 1e9})
    b.define("QF", "Quadrupole", length=L_QUAD, MagneticMultipoleP={"Kn1": K1})
    b.define("QD", "Quadrupole", length=L_QUAD, MagneticMultipoleP={"Kn1": -K1})
    b.define("D", "Drift", length=L_DRIFT)
    b.define("COR", "Kicker")
    b.define("MK", "Marker")
    first = ["QF", "D", "COR", "QD", "D", "MK"]
    rest = ["QF", "D", "QD", "D"] * (CELLS - 1)
    b.add_branch("ring", "periodic", ["BEG", *first, *rest])
    b.add_branch("line", "linac", ["BEG", *first, *rest])
    return b.build()


@pytest.fixture(scope="module", params=list(ENGINES))
def name(request):
    candidate = engine(request.param)
    if not candidate.available():
        pytest.skip(f"{request.param} is not installed here")
    return request.param


@pytest.fixture(scope="module")
def ring(fodo, name):
    try:
        session = engine(name).open(fodo, fodo.layout("ring").id, PRECISE)
    except EngineUnavailable as error:
        pytest.skip(str(error))
    yield session
    session.close()


def occurrence(lattice, branch: str, name: str) -> str:
    return lattice.placement(name, branch=branch).id


# -- what an engine is -----------------------------------------------------------------------------


def test_an_engine_is_an_adapter_whose_sessions_check_requests_first(name, ring):
    adapter = engine(name)
    assert isinstance(adapter, EngineAdapter) and isinstance(ring, EngineSession)
    capabilities = adapter.capabilities()
    assert capabilities.engine == adapter.name == ring.engine == name
    assert capabilities.updatable <= SETTINGS  # it changes in place only what the contract allows


# -- the physics, against the matrices --------------------------------------------------------------


def test_a_ring_has_the_tunes_and_optics_the_matrices_give(fodo, ring):
    result = ring.optics(OpticsRequest(options=PRECISE))
    (mux, bx, ax), (muy, by, ay) = periodic("x"), periodic("y")
    assert (result.tune_x, result.tune_y) == pytest.approx(
        (CELLS * mux / (2 * math.pi), CELLS * muy / (2 * math.pi)), abs=1e-7
    )
    assert (result.start.beta_x, result.start.beta_y) == pytest.approx((bx, by), rel=1e-7)
    assert (result.start.alpha_x, result.start.alpha_y) == pytest.approx((ax, ay), abs=1e-7)
    assert max(abs(e) for e in result.eta_x) < 1e-9  # no bends, no dispersion
    assert result.s[-1] == pytest.approx(CELLS * 2 * (L_QUAD + L_DRIFT))


def test_a_line_carries_its_initial_optics_cell_after_cell(fodo, name):
    session = engine(name).open(fodo, fodo.layout("line").id, PRECISE)
    try:
        (mux, bx, ax), (muy, by, ay) = periodic("x"), periodic("y")
        initial = TwissInit(beta_x=bx, alpha_x=ax, beta_y=by, alpha_y=ay)
        result = session.optics(OpticsRequest(options=PRECISE, initial=initial))
        ends = [i for i, o in enumerate(result.occurrences) if fodo.placement(o).name.startswith("D_")][1::2]
        assert len(ends) == CELLS  # the second drift ends each cell
        for k, i in enumerate(ends, start=1):  # periodic optics come back at the end of every cell
            assert (result.beta_x[i], result.beta_y[i]) == pytest.approx((bx, by), rel=1e-7)
            assert result.mu_x[i] == pytest.approx(k * mux / (2 * math.pi), abs=1e-7)
            assert result.mu_y[i] == pytest.approx(k * muy / (2 * math.pi), abs=1e-7)
    finally:
        session.close()


def test_a_kick_moves_the_closed_orbit_as_linear_optics_predict(fodo, name, ring):
    """A positive Kn0L deflects toward -x, a positive Ks0L toward +y; the closed orbit follows
    x(s) = theta sqrt(beta beta_k) cos(|mu - mu_k| - pi Q) / (2 sin pi Q)."""
    cor = occurrence(fodo, "ring", "COR_1")
    optics = ring.optics(OpticsRequest(options=PRECISE))
    kicked = engine(name).open(
        fodo,
        fodo.layout("ring").id,
        PRECISE,
        {(cor, "MagneticMultipoleP.Kn0L"): KICK, (cor, "MagneticMultipoleP.Ks0L"): KICK},
    )
    try:
        orbit = kicked.orbit(OrbitRequest(options=PRECISE))
    finally:
        kicked.close()
    k = optics.occurrences.index(cor)
    for plane, theta in (("x", -KICK), ("y", +KICK)):
        beta, mu = (
            np.asarray(getattr(optics, f"beta_{plane}")),
            2 * math.pi * np.asarray(getattr(optics, f"mu_{plane}")),
        )
        q = getattr(optics, f"tune_{plane}")
        expected = (
            theta
            * np.sqrt(beta * beta[k])
            * np.cos(np.abs(mu - mu[k]) - math.pi * q)
            / (2 * math.sin(math.pi * q))
        )
        got = np.asarray(getattr(orbit, plane))
        assert np.max(np.abs(got - expected)) < 1e-4 * np.max(np.abs(expected)), plane


def test_a_misaligned_quadrupole_moves_the_beam_as_its_offset_pitch_and_yaw_predict(fodo, name):
    """BodyShiftP is about the body's centre, rotations right-handed (PALS): a yaw y_rot turns the
    exit towards +x, a pitch x_rot towards -y. In the body's frame the beam enters off its axis by
    the offset and half a length of the turned axis, at minus the axis's angle; the quadrupole's
    matrix takes it through, and the same, reversed, brings it back. Every setting changes in place."""
    qf = occurrence(fodo, "line", "QF_1")
    dx, dy, yaw, pitch = 2e-4, -1e-4, 3e-4, -2e-4
    half = L_QUAD / 2
    x = quad(K1, L_QUAD) @ np.array([-dx + yaw * half, -yaw]) + np.array([dx + yaw * half, yaw])
    y = quad(-K1, L_QUAD) @ np.array([-dy - pitch * half, pitch]) + np.array([dy - pitch * half, -pitch])
    values = {
        (qf, "BodyShiftP.x_offset"): dx,
        (qf, "BodyShiftP.y_offset"): dy,
        (qf, "BodyShiftP.y_rot"): yaw,
        (qf, "BodyShiftP.x_rot"): pitch,
    }
    request = OrbitRequest(options=PRECISE, initial=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    for opened in (values, {}):
        session = engine(name).open(fodo, fodo.layout("line").id, PRECISE, opened)
        try:
            if not opened:
                session.update(values)
            orbit = session.orbit(request)
        finally:
            session.close()
        k = orbit.occurrences.index(qf)
        got = np.array([orbit.x[k], orbit.px[k], orbit.y[k], orbit.py[k]])
        # First order in the angles: their squares (1e-7 of them) are what an exact rotation adds.
        assert got == pytest.approx(np.concatenate([x, y]), abs=1e-9), opened or "updated"


def test_an_update_in_place_is_the_session_opened_with_the_values(fodo, name, ring):
    qf = occurrence(fodo, "ring", "QF_1")
    values = {(qf, "MagneticMultipoleP.Kn1"): 1.05 * K1}
    fresh = engine(name).open(fodo, fodo.layout("ring").id, PRECISE, values)
    try:
        ring.update(values)
        updated, opened = (
            ring.optics(OpticsRequest(options=PRECISE)),
            fresh.optics(OpticsRequest(options=PRECISE)),
        )
        assert (updated.tune_x, updated.tune_y) == pytest.approx((opened.tune_x, opened.tune_y), abs=1e-12)
        assert updated.beta_x == pytest.approx(opened.beta_x, rel=1e-12)
    finally:
        ring.update({(qf, "MagneticMultipoleP.Kn1"): K1})
        fresh.close()


def test_tracking_turn_by_turn_shows_the_tune(fodo, ring):
    mk = occurrence(fodo, "ring", "MK_1")
    track = ring.track(
        TrackRequest(options=PRECISE, particles=((1e-4, 0, 1e-4, 0, 0, 0),), turns=512, observe=(mk,))
    )
    optics = ring.optics(OpticsRequest(options=PRECISE))
    for plane in ("x", "y"):
        signal = np.array([turn[0] for turn in getattr(track, plane)[0]])
        q = getattr(optics, f"tune_{plane}") % 1.0
        assert tune_of(signal) == pytest.approx(min(q, 1 - q), abs=2e-4)


def test_a_bunchs_centroids_are_its_particles_mean(fodo, ring):
    mk = occurrence(fodo, "ring", "MK_1")
    rng = np.random.default_rng(3)
    bunch = np.column_stack([rng.normal(0, 1e-4, 16) for _ in range(4)] + [np.zeros(16), np.zeros(16)])
    beam = ring.track_beam(bunch, 3, [mk], PRECISE)
    each = ring.track(
        TrackRequest(options=PRECISE, particles=tuple(map(tuple, bunch)), turns=3, observe=(mk,))
    )
    for plane in ("x", "y"):
        means = [np.mean(turn) for turn in getattr(each, plane)[0]]
        assert getattr(beam, plane)[0] == pytest.approx(means, abs=1e-12)
    assert list(beam.count[0]) == [16, 16, 16]


def test_a_ramp_moves_a_setting_turn_by_turn_and_leaves_it_at_its_end(fodo, name, ring):
    cor = occurrence(fodo, "ring", "COR_1")
    mk = occurrence(fodo, "ring", "MK_1")
    turns = 64
    ramp = Ramp(
        {(cor, "MagneticMultipoleP.Kn0L"): 0.0},
        {(cor, "MagneticMultipoleP.Kn0L"): KICK},
        np.linspace(0, 1, turns),
    )
    beam = ring.track_beam(np.zeros((1, 6)), turns, [mk], PRECISE, ramps=[ramp])
    assert (
        beam.x[0][0] == pytest.approx(0.0, abs=1e-15) and abs(beam.x[0][-1]) > 1e-7
    )  # the orbit moved as it ramped
    end = engine(name).open(fodo, fodo.layout("ring").id, PRECISE, {(cor, "MagneticMultipoleP.Kn0L"): KICK})
    try:
        assert ring.orbit(OrbitRequest(options=PRECISE)).x == pytest.approx(
            end.orbit(OrbitRequest(options=PRECISE)).x, abs=1e-15
        )
    finally:
        ring.update({(cor, "MagneticMultipoleP.Kn0L"): 0.0})
        end.close()


# -- what every engine refuses, and why --------------------------------------------------------------


def test_every_engine_refuses_the_same_requests_for_the_same_reasons(fodo, name, ring):
    line = engine(name).open(fodo, fodo.layout("line").id, PRECISE)
    qf = occurrence(fodo, "ring", "QF_1")
    elsewhere = occurrence(fodo, "line", "QF_1")
    mk_line = occurrence(fodo, "line", "MK_1")
    first, cor = occurrence(fodo, "line", "QF_1"), occurrence(fodo, "line", "COR_1")
    refusals = {
        "cannot change in place": lambda: ring.update({(qf, "BendP.angle_ref"): 0.1}),
        "not in this branch": lambda: ring.update({(elsewhere, "MagneticMultipoleP.Kn1"): K1}),
        "initial conditions": lambda: line.optics(OpticsRequest(options=PRECISE)),
        "passed once; turns > 1": lambda: line.track(
            TrackRequest(options=PRECISE, particles=((0, 0, 0, 0, 0, 0),), turns=2)
        ),
        "ramp while a ring turns": lambda: line.track_beam(
            np.zeros((1, 6)),
            1,
            [],
            PRECISE,
            ramps=[
                Ramp({(cor, "MagneticMultipoleP.Kn0L"): 0.0}, {(cor, "MagneticMultipoleP.Kn0L"): 1e-6}, [0.0])
            ],
        ),
        "outside the part tracked": lambda: line.track_beam(
            np.zeros((1, 6)), 1, [mk_line], PRECISE, start=first, end=cor
        ),
    }
    try:
        for reason, request in refusals.items():
            with pytest.raises(UnsupportedRequest) as refused:
                request()
            assert refused.value.engine == name and len(refused.value.reasons) == 1, reason
            assert reason in refused.value.reasons[0], refused.value.reasons
    finally:
        line.close()
