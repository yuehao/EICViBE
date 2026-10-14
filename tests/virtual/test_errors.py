"""A virtual machine's errors (ADR 0011): drawn once from a seed, in the engine only, and physical."""

import json
import math

import numpy as np
import pytest

pytest.importorskip("xtrack")

from eicvibe.api.demo import ROUTES, demo_lattice  # noqa: E402
from eicvibe.domain.lattice import ElementKind, route_through  # noqa: E402
from eicvibe.domain.machine import NamingArea, generate_machine_registry  # noqa: E402
from eicvibe.domain.machine.power import design_value  # noqa: E402
from eicvibe.engines import engine  # noqa: E402
from eicvibe.engines.contracts import Fidelity, OpticsRequest, Options, OrbitRequest  # noqa: E402
from eicvibe.virtual import (  # noqa: E402
    Beam,
    BpmHistory,
    BpmModel,
    ErrorSpec,
    Jitter,
    MachineErrors,
    MagnetErrors,
    RingRun,
    Ripple,
    RouteRun,
    Selection,
    SupplyNoise,
)
from eicvibe.virtual.errors import VirtualBpms  # noqa: E402

PERFECT = BpmModel(resolution=0, orbit_resolution=0, offset_rms=0, gain_rms=0)
FAST = Options(fidelity=Fidelity.FAST, effects=frozenset())


@pytest.fixture(scope="module")
def demo():
    lattice = demo_lattice()
    areas = {"ring": (NamingArea(prefix="dr", start=0.0, end=lattice.layout("ring").length + 1.0),)}
    registry = generate_machine_registry(
        lattice, areas, series={"ring": (ElementKind.BEND, ElementKind.QUADRUPOLE)}
    )
    return lattice, registry


@pytest.fixture(params=["xsuite", "trackpad"])
def tracker(request):
    if not engine(request.param).available():
        pytest.skip(f"{request.param} is not installed here")
    return request.param


def quads(lattice) -> list[str]:
    return [p.id for p in lattice.layout("ring").placements if p.kind is ElementKind.QUADRUPOLE]


def test_errors_are_drawn_once_from_the_seed_for_the_magnets_chosen(demo):
    lattice, registry = demo
    spec = ErrorSpec(
        seed=11,
        magnets=(
            MagnetErrors(
                select=Selection(kinds=(ElementKind.QUADRUPOLE,)), x_offset=1e-4, pitch=2e-4, field=1e-3
            ),
            MagnetErrors(select=Selection(names=("qf*",)), x_offset=1e-4, cut=1.0),
        ),
    )
    one, two = (
        MachineErrors(spec, lattice, ["ring"], registry),
        MachineErrors(spec, lattice, ["ring"], registry),
    )
    assert one.shifts == two.shifts and one.factors == two.factors  # the same machine every time
    other = MachineErrors(spec.replaced(seed=12), lattice, ["ring"], registry)
    assert other.shifts != one.shifts

    ring = lattice.layout("ring").placements
    chosen = {p.id for p in ring if p.kind is ElementKind.QUADRUPOLE}
    assert {o for o, path in one.shifts if path == "BodyShiftP.x_rot"} == chosen
    assert {o for o, _ in one.factors} == chosen
    assert all(path == "MagneticMultipoleP.Kn1" for _, path in one.factors)
    pitches = [v for (_, path), v in one.shifts.items() if path == "BodyShiftP.x_rot"]
    assert max(abs(v) for v in pitches) <= 2.5 * 2e-4  # cut at 2.5 sigma, by default
    # A magnet chosen twice has both errors: QF's offsets add the second draw, cut at one sigma.
    qf = [p for p in ring if p.name.lower().startswith("qf")]
    qd = [p for p in ring if p.kind is ElementKind.QUADRUPOLE and not p.name.lower().startswith("qf")]
    only_first = MachineErrors(
        spec.replaced(magnets=spec.model_dump()["magnets"][:1]), lattice, ["ring"], registry
    )
    assert all(
        one.shifts[(p.id, "BodyShiftP.x_offset")] == only_first.shifts[(p.id, "BodyShiftP.x_offset")]
        for p in qd
    )
    second = [
        one.shifts[(p.id, "BodyShiftP.x_offset")] - only_first.shifts[(p.id, "BodyShiftP.x_offset")]
        for p in qf
    ]
    assert max(abs(v) for v in second) <= 1e-4 and any(second)
    # Nothing is drawn for what the spec does not name, nor outside the branches asked for.
    assert MachineErrors(ErrorSpec(seed=1), lattice, ["ring"], registry).empty
    line = MachineErrors(spec, lattice, ["injection"], registry)
    assert {o for o, _ in line.shifts} == {
        p.id for p in lattice.layout("injection").placements if p.kind is ElementKind.QUADRUPOLE
    }


def test_settings_reach_the_engine_through_the_errors(demo):
    lattice, registry = demo
    spec = ErrorSpec(
        seed=3,
        magnets=(
            MagnetErrors(select=Selection(kinds=(ElementKind.BEND,)), field=1e-3, roll=1e-4),
            MagnetErrors(select=Selection(kinds=(ElementKind.QUADRUPOLE,)), y_offset=1e-4),
        ),
    )
    errors = MachineErrors(spec, lattice, ["ring"], registry)
    bend = next(p for p in lattice.layout("ring").placements if p.name.startswith("B_"))  # not the septum
    field = (bend.id, "MagneticMultipoleP.Kn0")
    factor = errors.factors[field]
    # Opened: every error, on the lattice's values (a bend's field from its angle, where implied).
    opened = errors.engine_values({}, complete=True)
    assert opened[field] == pytest.approx(design_value(bend, field[1]) * factor) and design_value(
        bend, field[1]
    )
    assert opened[(bend.id, "BodyShiftP.z_rot")] == errors.shifts[(bend.id, "BodyShiftP.z_rot")]
    # An update: the setting given, scaled by its magnet's error; nothing else.
    assert errors.engine_values({field: 0.5}) == {field: pytest.approx(0.5 * factor)}
    quad = quads(lattice)[0]
    unrelated = {(quad, "MagneticMultipoleP.Kn1"): 0.7}
    assert errors.engine_values(unrelated) == unrelated  # quadrupoles have offsets here, no field error


def test_a_ring_with_errors_reads_the_orbit_they_make_and_keeps_them_hidden(demo, tracker, tmp_path):
    lattice, registry = demo
    spec = ErrorSpec(
        seed=5,
        magnets=(
            MagnetErrors(
                select=Selection(kinds=(ElementKind.QUADRUPOLE,)), x_offset=2e-4, y_offset=1e-4, yaw=3e-4
            ),
        ),
    )
    errors = MachineErrors(spec, lattice, ["ring"], registry)
    run = RingRun(
        lattice,
        "ring",
        tmp_path / "run",
        engine=tracker,
        fidelity=Fidelity.FAST,
        effects=frozenset(),
        beam=Beam(particles=1),
        bpm=PERFECT,
        registry=registry,
        turns=256,
        errors=errors,
    )
    run.start()
    run.wait()
    run.stop()
    assert run.error is None
    session = engine(tracker).open(
        lattice, lattice.layout("ring").id, FAST, errors.engine_values({}, complete=True)
    )
    try:
        closed = session.orbit(OrbitRequest(options=FAST))
    finally:
        session.close()
    history = BpmHistory.open(tmp_path / "run")
    readings = history.turn_by_turn(0, 256)
    at = [closed.occurrences.index(o) for o in history.info.occurrences]
    # One particle on the machine's closed orbit: every BPM reads it, a few mm of it, turn after turn.
    expected = 1e3 * np.asarray(closed.x)[at]
    assert np.abs(expected).max() > 0.2
    assert readings.x.position.mean(axis=1) == pytest.approx(expected, abs=1e-6)
    # The settings, the start and the log know nothing of the errors; hidden.json keeps them.
    assert history.info.values == {} and run.values == {}
    assert not any("BodyShiftP" in json.dumps(e) for e in history.events())
    hidden = json.loads((tmp_path / "run" / "hidden.json").read_text())
    assert hidden["errors"]["seed"] == 5
    assert {e["name"] for e in hidden["errors"]["magnets"].values()} == {
        p.name for p in lattice.layout("ring").placements if p.kind is ElementKind.QUADRUPOLE
    }


def test_a_dipole_supplys_ripple_moves_the_orbit_by_the_dispersion(demo, tracker, tmp_path):
    """Every bend on one supply: a slow relative change eta of its current bends the beam as a
    momentum change of -eta would, so without RF the orbit is -D eta where the dispersion is D."""
    lattice, registry = demo
    ring = lattice.layout("ring")
    session = engine("xsuite").open(lattice, ring.id, FAST)
    try:
        optics = session.optics(OpticsRequest(options=FAST))
    finally:
        session.close()
    turns = 6000
    period = ring.length / 299_792_458.0  # an ultra-relativistic electron's
    ripple = Ripple(frequency=1.0 / (3000 * period), amplitude=1e-4)  # a period of 3000 turns: quasi-static
    spec = ErrorSpec(
        seed=2,
        supplies=(
            SupplyNoise(select=Selection(kinds=(ElementKind.BEND,), names=("b_*",)), ripple=(ripple,)),
        ),
    )
    errors = MachineErrors(spec, lattice, ["ring"], registry)
    assert (
        len(errors.supplies) == 1 and len(errors.supplies[0].keys) == 22
    )  # one string: the bends, not the septum
    run = RingRun(
        lattice,
        "ring",
        tmp_path / "ripple",
        engine=tracker,
        fidelity=Fidelity.FAST,
        effects=frozenset(),
        beam=Beam(particles=1),
        bpm=PERFECT,
        registry=registry,
        turns=turns,
        buffer_turns=turns,
        ramp_turns=0,
        errors=errors,
    )
    run.start()
    run.wait()
    run.stop()
    assert run.error is None
    history = BpmHistory.open(tmp_path / "ripple")
    x = history.turn_by_turn(0, turns).x.position
    [(_, eta)] = errors.fluctuations(run.revolution_period).turns(0, turns)
    assert np.ptp(eta) == pytest.approx(2e-4, rel=0.01)
    for k, occurrence in enumerate(history.info.occurrences):
        d = optics.at(occurrence).eta_x
        slope = np.polyfit(eta, x[k], 1)[0]
        assert slope == pytest.approx(-1e3 * d, rel=0.05), k


def test_a_bpms_roll_mixes_the_planes_and_its_closed_orbit_has_noise_of_its_own(demo, tmp_path):
    rolled = VirtualBpms(BpmModel(offset_rms=0, gain_rms=0, roll_rms=1e-2, resolution=0), 3, 10, 1e-9)
    y = np.full((3, 1), 1e-3)
    rx, ry = rolled.systematic(np.zeros((3, 1)), y, np.full((3, 1), 10))
    assert rx[:, 0] == pytest.approx(np.sin(rolled.rolls))  # mm: 1 mm of y seen in x as the roll's sine
    assert ry[:, 0] == pytest.approx(np.cos(rolled.rolls))
    assert np.abs(rolled.rolls).max() > 1e-3

    lattice, registry = demo
    for orbit in (0.0, 0.01):
        run = RingRun(
            lattice,
            "ring",
            tmp_path / f"orbit{orbit}",
            engine="xsuite",
            fidelity=Fidelity.FAST,
            effects=frozenset(),
            beam=Beam(particles=1),
            bpm=BpmModel(resolution=1.0, orbit_resolution=orbit, offset_rms=0, gain_rms=0),
            registry=registry,
            turns=4096,
            slow_period=0.0,  # a sample per engine call
        )
        run.start()
        run.wait()
        run.stop()
        history = BpmHistory.open(tmp_path / f"orbit{orbit}")
        assert history.info.orbit_resolution == orbit
        samples = history.orbit()
        assert len(samples.turn) > 3
        spread = np.std(samples.x, axis=0).max()  # on the closed orbit, at rest: only the noise moves
        turnwise = history.turn_by_turn(0, 4096).x.position.std(axis=1).min()
        assert turnwise == pytest.approx(1.0, rel=0.1)  # turn by turn: its own 1 mm noise
        if orbit:
            assert spread == pytest.approx(orbit, rel=0.6)
        else:
            assert spread < 1e-9  # no average of the turn-by-turn noise


def test_a_routes_supplies_jitter_shot_to_shot(tmp_path):
    lattice = demo_lattice()
    route = route_through(lattice, "transfer", ROUTES["transfer"])
    branches = list(dict.fromkeys(s.branch for s in route.segments))
    spec = ErrorSpec(seed=4, supplies=(SupplyNoise(select=Selection(kinds=(ElementKind.BEND,)), rms=1e-3),))
    trajectories = {}
    for name, errors in (("quiet", None), ("noisy", MachineErrors(spec, lattice, branches))):
        run = RouteRun(
            lattice,
            route,
            tmp_path / name,
            engine="xsuite",
            fidelity=Fidelity.FAST,
            beam=Beam(particles=1),
            jitter=Jitter(x=0, px=0, y=0, py=0, delta=0),
            bpm=PERFECT,
            rate=1000.0,
            shots=8,
            chunk=4,
            errors=errors,
        )
        run.start()
        run.wait()
        run.stop()
        assert run.error is None
        trajectories[name] = BpmHistory.open(tmp_path / name).turn_by_turn(0, 8).x.position
    assert np.nanstd(trajectories["quiet"], axis=1).max() < 1e-9  # every shot the same
    jitter = np.nanstd(trajectories["noisy"], axis=1)
    assert np.nanmax(jitter) > 1e-3 and not math.isnan(np.nanmax(jitter))  # each shot its own currents


def test_a_row_that_chooses_nothing_is_said_and_a_ring_without_a_closed_orbit_loses_its_beam(demo, tmp_path):
    lattice, registry = demo
    nothing = ErrorSpec(seed=1, magnets=(MagnetErrors(select=Selection(names=("any",)), x_offset=1e-3),))
    drawn = MachineErrors(nothing, lattice, ["ring"], registry)
    assert drawn.empty and drawn.chosen == {"magnets": [0], "supplies": []}
    assert drawn.problems == ["magnet errors, row 1: no magnet named any here"]

    # Quadrupoles 20 % off: no stable optics. The injector still sends the design's bunch; it is lost.
    broken = ErrorSpec(
        seed=1, magnets=(MagnetErrors(select=Selection(kinds=(ElementKind.QUADRUPOLE,)), field=0.2),)
    )
    run = RingRun(
        lattice,
        "ring",
        tmp_path / "broken",
        engine="xsuite",
        fidelity=Fidelity.FAST,
        beam=Beam(particles=4),
        bpm=PERFECT,
        registry=registry,
        turns=512,
        errors=MachineErrors(broken, lattice, ["ring"], registry),
    )
    run.start()
    run.wait()
    run.stop()
    assert run.error is None
    assert any("no closed orbit or no stable optics with its errors" in w for w in run.warnings)
    history = BpmHistory.open(tmp_path / "broken")
    assert any(e["kind"] == "loss" for e in history.events())
    assert np.isnan(history.turn_by_turn(448, 64).x.position).all()  # nothing left to read
