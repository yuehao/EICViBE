"""Bunches matched to the optics: a ring's periodic solution, a single pass's initial optics."""

import math

import numpy as np
import pytest

pytest.importorskip("xtrack")

from eicvibe.api.demo import demo_lattice  # noqa: E402
from eicvibe.domain.lattice import ElementKind, LatticeBuilder, SequentialIds, route_through  # noqa: E402
from eicvibe.engines import engine, machine_optics, route_start  # noqa: E402
from eicvibe.engines.contracts import Effect, Fidelity, OpticsRequest, Options, OrbitRequest  # noqa: E402
from eicvibe.engines.model import initial_conditions  # noqa: E402
from eicvibe.virtual import Beam, BpmHistory, RingRun, one_turn_matrix, ring_matching  # noqa: E402

SIX = Options(fidelity=Fidelity.FAST, effects=frozenset({Effect.LONGITUDINAL}))
FOUR = Options(fidelity=Fidelity.FAST)


@pytest.fixture(scope="module")
def ring():
    lattice = demo_lattice()
    cavity = next(p.id for p in lattice.layout("ring").placements if p.kind is ElementKind.RFCAVITY)
    stable = {(cavity, "RFP.phase"): math.pi / 2}
    return lattice, stable


def matched(lattice, values, options):
    session = engine("xsuite").open(lattice, "ring", options, values)
    orbit = session.orbit(OrbitRequest(options=options))
    closed = np.array([orbit.x[-1], orbit.px[-1], orbit.y[-1], orbit.py[-1], orbit.zeta[-1], orbit.delta[-1]])
    return session, closed, ring_matching(session, closed, options, "ring's start")


def test_a_rings_bunch_without_rf_is_matched_to_its_periodic_optics(ring):
    lattice, stable = ring
    session, _, matching = matched(lattice, stable, FOUR)
    assert not matching.longitudinal and matching.to.endswith("4D")
    optics = session.optics(OpticsRequest(options=FOUR)).start
    twiss = matching.twiss()
    for key in ("beta_x", "alpha_x", "beta_y", "alpha_y", "eta_x", "etap_x"):
        assert twiss[key] == pytest.approx(getattr(optics, key), rel=1e-6, abs=1e-8), key
    beam = Beam(particles=4000, sigma_zeta=0.01)
    bunch = matching.bunch(np.random.default_rng(0), beam)
    assert np.std(bunch[:, 0]) == pytest.approx(
        math.sqrt(beam.emittance_x * optics.beta_x + (optics.eta_x * beam.sigma_delta) ** 2), rel=0.05
    )
    assert np.std(bunch[:, 4]) == pytest.approx(0.01, rel=0.05)  # nothing matches a coasting beam's length


def test_with_rf_the_bunch_length_is_matched_and_stays(ring):
    lattice, stable = ring
    session, closed, matching = matched(lattice, stable, SIX)
    assert matching.longitudinal and matching.to.endswith("6D")
    R = np.linalg.inv(matching.W) @ one_turn_matrix(session, closed, SIX) @ matching.W
    for k in range(3):  # the normal modes: the map turns each in its own plane
        R[2 * k : 2 * k + 2, 2 * k : 2 * k + 2] = 0.0
    assert np.abs(R).max() < 1e-9
    beam = Beam(particles=2000)
    summary = matching.summary(beam)
    assert 0 < summary["synchrotron_tune"] < 0.05

    def lengths(bunch):
        out = [np.std(bunch[:, 4])]
        for _ in range(4):
            bunch = np.asarray(session.track_beam(bunch, 20, (), SIX).final)
            out.append(np.std(bunch[:, 4]))
        return np.array(out)

    bunch = matching.bunch(np.random.default_rng(1), beam)
    assert np.std(bunch[:, 4]) == pytest.approx(summary["sigma_zeta"], rel=0.03)
    kept = lengths(bunch)
    assert kept / kept[0] == pytest.approx(1.0, abs=0.03)  # a quarter synchrotron period and more
    longer = bunch.copy()
    longer[:, 4] = closed[4] + 3 * (bunch[:, 4] - closed[4])
    breathing = lengths(longer)
    assert breathing.min() / breathing[0] < 0.5  # a mismatched bunch breathes


def test_rf_that_holds_no_bunch_is_matched_transversely(ring):
    lattice, stable = ring
    _, _, matching = matched(lattice, {key: 0.0 for key in stable}, SIX)  # the crest, above transition
    assert not matching.longitudinal and "the RF holds no bunch" in matching.to


def test_a_ring_run_says_what_its_bunch_is_matched_to(ring, tmp_path):
    lattice, stable = ring
    run = RingRun(
        lattice,
        "ring",
        tmp_path / "run",
        engine="xsuite",
        fidelity=Fidelity.FAST,
        beam=Beam(particles=8),
        chunk=256,
        values=stable,
        turns=256,
    )
    run.start()
    run.wait()
    run.stop()
    history = BpmHistory.open(tmp_path / "run")
    matched_to = history.info.beam["matched"]
    assert matched_to["to"] == "the one-turn map at ring's start, 6D" and matched_to["sigma_zeta"] > 0
    (inject,) = [e for e in history.events() if e["kind"] == "inject"]
    assert inject["sigma_zeta"] == pytest.approx(matched_to["sigma_zeta"])


def test_a_single_pass_starts_from_its_lines_initial_optics_and_orbit():
    b = LatticeBuilder("line", ids=SequentialIds())
    reference = {"species_ref": "electron", "pc_ref": 1e9}
    b.define(
        "BEG",
        "BeginningEle",
        ReferenceP=reference,
        TwissP={"beta_a": 8.0, "beta_b": 4.0, "alpha_a": 0.5, "eta_x": 0.2},
        ParticleP={"x": 1e-3, "px": -2e-4},
    )
    b.define("D", "Drift", length=1.0)
    b.add_branch("line", "linac", ["BEG", "D"])
    lattice = b.build()
    start, origin = route_start(engine("xsuite"), lattice, route_through(lattice, "line", ["line"]))
    assert origin == "line's initial conditions"
    assert (start.beta_x, start.alpha_x, start.eta_x, start.x, start.px) == (8.0, 0.5, 0.2, 1e-3, -2e-4)
    assert initial_conditions(lattice, "line") == start


def test_a_single_pass_from_a_fork_starts_from_the_machines_optics_there(ring):
    lattice, _ = ring
    route = route_through(lattice, "out", ["extraction"])
    start, origin = route_start(engine("xsuite"), lattice, route)
    assert origin == "the optics of extraction at X_START_1"
    whole = machine_optics(engine("xsuite"), lattice)
    assert start.beta_x == pytest.approx(whole[lattice.layout("extraction").id].start.beta_x)
