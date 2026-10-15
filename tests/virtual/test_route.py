"""Shots along a route: every BPM's trajectory reading, shot after shot, as a real line records them."""

import math

import numpy as np
import pytest

pytest.importorskip("xtrack")

from eicvibe.api.demo import ROUTES, demo_lattice  # noqa: E402
from eicvibe.diagnostics import NO_BEAM  # noqa: E402
from eicvibe.domain.lattice import ElementKind, route_through  # noqa: E402
from eicvibe.domain.machine import NamingArea, generate_machine_registry  # noqa: E402
from eicvibe.engines import engine, route_optics  # noqa: E402
from eicvibe.engines.contracts import Fidelity  # noqa: E402
from eicvibe.virtual import Beam, BpmHistory, BpmModel, Jitter, RouteRun, route_bpms  # noqa: E402

QUIET = {
    "jitter": Jitter(x=0, px=0, y=0, py=0, delta=0),
    "bpm": BpmModel(resolution=0, offset_rms=0, gain_rms=0),
}


@pytest.fixture(scope="module")
def demo():
    lattice = demo_lattice()
    areas = {
        "ring": (NamingArea(prefix="dr", start=0.0, end=lattice.layout("ring").length + 1.0),),
        "injection": None,
        "extraction": (NamingArea(prefix="dx", start=0.0, end=lattice.layout("extraction").length + 1.0),),
    }
    registry = generate_machine_registry(lattice, areas)
    return lattice, registry, route_through(lattice, "transfer", ROUTES["transfer"])


def shots(demo, folder, **options) -> RouteRun:
    lattice, registry, route = demo
    options = {
        "engine": "xsuite",
        "fidelity": Fidelity.FAST,
        "beam": Beam(particles=16),
        "rate": 1000.0,
    } | options
    return RouteRun(lattice, route, folder, registry=registry, **options)


def test_a_route_run_records_every_shot_as_bpms_read_it(demo, tmp_path):
    lattice, registry, route = demo
    run = shots(demo, tmp_path / "run", shots=12, chunk=4).start()
    run.wait()
    run.stop()
    assert run.error is None
    history = BpmHistory.open(tmp_path / "run")
    info = history.info
    assert (info.kind, info.unit, info.branch, info.branch_id) == (
        "route",
        "shot",
        "transfer",
        "route:transfer",
    )
    assert history.span() == (0, 12) and history.state()["state"] == "done"
    bpms = route_bpms(lattice, route, registry)
    assert info.bpms == [b[2] for b in bpms] and info.s == pytest.approx([b[3] for b in bpms])
    assert info.bpms[0].startswith("injection-b") and info.bpms[-1].startswith("dx-b")  # every branch's
    x = history.turn_by_turn(0, 12).x.position
    assert np.isfinite(x).all()
    assert (np.std(x, axis=1) > 0.005).all()  # shot-to-shot jitter, beyond the BPMs' resolution


def test_a_corrector_moves_the_trajectory_downstream_as_linear_optics_predict(demo, tmp_path):
    lattice, _, route = demo
    corrector = next(p for p in lattice.layout("ring").placements if p.kind is ElementKind.KICKER)
    run = shots(demo, tmp_path / "kick", beam=Beam(particles=1), chunk=1, **QUIET).start()
    run.wait(3)
    run.update({(corrector.id, "MagneticMultipoleP.Kn0L"): 1e-5})  # 1e-5 toward -x
    run.wait(6)
    run.stop()
    history = BpmHistory.open(tmp_path / "kick")
    (event,) = [e for e in history.events() if e["kind"] == "settings"]
    x = history.turn_by_turn(0, 6).x.position
    before, after = x[:, event["turn"] - 1], x[:, event["turn"]]
    optics = route_optics(engine("xsuite"), lattice, route)
    k = optics.occurrences.index(corrector.id)
    at = [optics.occurrences.index(o) for o in history.info.occurrences]
    beta, mu = np.array(optics.beta_x), np.array(optics.mu_x)
    phase = 2 * math.pi * (mu[at] - mu[k])
    expected = np.where(np.array(at) > k, -1e-2 * np.sqrt(beta[at] * beta[k]) * np.sin(phase), 0.0)  # mm
    # A thick kick, not a point: compared on the trajectory's scale, as near a node of the sine
    scale = np.abs(expected).max()
    assert after - before == pytest.approx(expected, rel=3e-2, abs=2e-2 * scale)


def test_a_lost_beam_reads_nothing_downstream(demo, tmp_path):
    lattice, _, _ = demo
    corrector = next(p for p in lattice.layout("ring").placements if p.kind is ElementKind.KICKER)
    run = shots(
        demo, tmp_path / "lost", chunk=2, shots=4, values={(corrector.id, "MagneticMultipoleP.Kn0L"): 0.5}
    )
    run.start()
    run.wait()
    run.stop()
    history = BpmHistory.open(tmp_path / "lost")
    tbt = history.turn_by_turn(0, 4)
    last = tbt.x.position[-1]
    assert np.isnan(last).all() and (tbt.x.status[-1] == NO_BEAM).all()
    assert np.isfinite(tbt.x.position[0]).all()  # the line before the ring still sees it
    assert "loss" in [e["kind"] for e in history.events()]


def test_an_acceleration_shrinks_the_angles_and_momentum_spread_and_keeps_positions():
    from eicvibe.virtual.route import accelerated

    bunch = np.array([[1e-3, 2e-4, -1e-3, -2e-4, 0.01, 1e-3]])
    faster = accelerated(bunch, 0.75 / 10.0)  # from 0.75 to 10 GeV/c
    assert faster[0] == pytest.approx([1e-3, 1.5e-5, -1e-3, -1.5e-5, 0.01, 7.5e-5])
    assert bunch[0, 1] == 2e-4  # the bunch given is left as it was


def test_a_bpm_read_on_two_passes_has_one_offset_and_gain():
    from eicvibe.virtual.errors import BpmModel, VirtualBpms

    bpms = VirtualBpms(BpmModel(), 4, 50, 1e-9, names=["b1", "b2", "b1", "b3"])
    assert bpms.offsets[:, 0] == pytest.approx(bpms.offsets[:, 2])
    assert bpms.gains[:, 0] == pytest.approx(bpms.gains[:, 2])
    assert bpms.offsets[0, 0] != bpms.offsets[0, 1]


def test_a_route_from_where_a_beam_is_delivered_shoots_that_beam(demo, tmp_path):
    """The beam an injector code delivered at a branch's start (ADR 0012): each shot is a sample of its
    particles, kept as machine files, at its own charge, rather than a Gaussian matched there."""
    from eicvibe.store import DeliveredBeam, read_beams, write_beam

    lattice, registry, route = demo
    first = lattice.branch(route.segments[0].branch).name
    rng = np.random.default_rng(3)
    particles = rng.normal(0.0, [2e-4, 2e-5, 1e-4, 1e-5, 1e-3, 1e-3], (500, 6))
    pc = lattice.initial_conditions(route.segments[0].branch).reference.pc_ref
    readings = {}
    for shift in (+5e-4, -5e-4):  # the same beam, its centroid 0.5 mm one way or the other
        moved = particles + [shift, 0, 0, 0, 0, 0]
        folder = tmp_path / f"machine{shift}"
        write_beam(folder, DeliveredBeam(first, moved, 3e-9, pc, {"code": "test"}))
        beam = read_beams(folder)[first]
        assert beam.particles == pytest.approx(moved) and beam.charge == 3e-9
        run = shots(demo, tmp_path / f"run{shift}", shots=2, chunk=2, delivered=beam, **QUIET)
        run.start()
        run.wait()
        run.stop()
        history = BpmHistory.open(tmp_path / f"run{shift}")
        assert history.info.beam["delivered"]["charge"] == 3e-9
        tbt = history.turn_by_turn(0, 2)
        assert tbt.x.intensity[0, 0] == pytest.approx(3.0)  # nC: the delivered beam's charge
        readings[shift] = tbt.x.position[:, 0]
    # The same sample, one centroid 1 mm from the other: every BPM reads the difference linear optics carry.
    difference = readings[5e-4] - readings[-5e-4]
    assert np.abs(difference).max() > 0.2 and not np.allclose(difference, 0)
    with pytest.raises(ValueError, match="does not start where the beam is delivered"):
        shots(demo, tmp_path / "other", delivered=DeliveredBeam("ring", particles, 1e-9, pc))
