"""The Xsuite engine against MAD-X: the same lattice gives the same physics."""

import math
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("xtrack")
madx_module = pytest.importorskip("cpymad.madx")

from eicvibe.engines import machine_optics  # noqa: E402
from eicvibe.engines.contracts import (  # noqa: E402
    Effect,
    Fidelity,
    OpticsRequest,
    Options,
    OrbitRequest,
    Ramp,
    TrackRequest,
    UnsupportedRequest,
)
from eicvibe.engines.model import Program, quasi_static  # noqa: E402
from eicvibe.engines.xsuite import XsuiteEngine  # noqa: E402
from eicvibe.importers import import_cpymad  # noqa: E402
from tests.importers.test_madx import ELEMENTS, RING  # noqa: E402

ENGINE = XsuiteEngine()
EIN = Path(os.environ.get("EICVIBE_EIN_FOLDER", "/Users/haoyue/Research/EIC/virtual_accelerator/EIN"))

# The parts of ELEMENTS the comparison leaves out are importer questions, not engine ones:
# MAD-X ignores knl on a QUADRUPOLE (the importer keeps it), and a DIPEDGE becomes a marker.
LINE = (
    ELEMENTS.replace("crab: crabcavity, l=1, volt=2, freq=394, tilt=pi/2;", "crab: drift, l=1;")
    .replace(", knl={0, 0, 0.03}", "")
    .replace("de: dipedge, h=0.05, e1=0.02;", "de: marker;")
)


@pytest.fixture(scope="module")
def ring():
    """The matched FODO ring of the importer tests, with MAD-X's TWISS of it."""
    with tempfile.TemporaryDirectory() as folder:
        (Path(folder) / "optics").mkdir()
        (Path(folder) / "optics" / "strengths.madx").write_text("kqf = 0.5;\nkqd = -0.5;\n")
        with madx_module.Madx(stdout=False, command_log=None) as madx:
            madx.chdir(folder)
            madx.input(RING.replace("stop;", ""))
            lattice = import_cpymad(madx, ["ring"], kind="periodic", name="ring").lattice
            twiss = madx.twiss(sequence="ring", chrom=True)  # chromaticity by finite differences
            table = {k: np.array(twiss[k]) for k in ("s", "betx", "bety", "dx", "mux", "x")}
            table["q"] = (twiss.summary.q1, twiss.summary.q2, twiss.summary.dq1, twiss.summary.dq2)
    return lattice, table


def at_positions(result, table, column, ours):
    """Our column at MAD-X's rows, matched by s (optics are functions of s)."""
    s = np.array(result.s)
    index = np.searchsorted(s, table["s"] - 1e-9)
    return np.array(getattr(result, ours))[np.clip(index, 0, len(s) - 1)], table[column]


def test_every_element_matches_xsuites_own_madx_import():
    xt = pytest.importorskip("xtrack")
    start = (1e-3, 2e-4, -5e-4, 1e-4, 0.0, 1e-3)
    with madx_module.Madx(stdout=False, command_log=None) as madx:
        madx.input(LINE)
        lattice = import_cpymad(madx, ["line"], kind="linac", name="el").lattice
        reference = xt.Line.from_madx_sequence(madx.sequence["line"], deferred_expressions=False)
    reference.particle_ref = xt.Particles(p0c=18e9, mass0=xt.ELECTRON_MASS_EV)
    reference.configure_bend_model(edge="full")  # the standard fidelity's edges
    particle = reference.build_particles(
        x=[start[0]], px=[start[1]], y=[start[2]], py=[start[3]], zeta=[0.0], delta=[start[5]]
    )
    reference.track(particle)
    six = Options(effects=frozenset({Effect.LONGITUDINAL}))  # the line has a cavity
    orbit = ENGINE.open(lattice, "line", six).orbit(OrbitRequest(initial=start, options=six))
    ours = (orbit.x[-1], orbit.px[-1], orbit.y[-1], orbit.py[-1])
    theirs = (particle.x[0], particle.px[0], particle.y[0], particle.py[0])
    # Tilted bend with fringe and gradient, rbend, tilted quadrupole, multipole to order 11,
    # kicker signs, RF, solenoid, monitor, and SROTATION, YROTATION and TRANSLATION patches.
    assert ours == pytest.approx(theirs, abs=1e-13)


def test_ring_optics_match_madx(ring):
    # Xsuite's exact bends differ from MAD-X's thick maps by a few 1e-6 in tune here, as for
    # Xsuite's own MAD-X import; the conversion itself is checked exactly against that import.
    # Bends of rho = 5 m: the chromaticity needs the chromatic terms of the edges.
    lattice, table = ring
    optics = ENGINE.open(lattice, "ring").optics(OpticsRequest())
    assert (optics.tune_x, optics.tune_y) == pytest.approx(table["q"][:2], abs=1e-5)
    assert (optics.chroma_x, optics.chroma_y) == pytest.approx(table["q"][2:], rel=1e-3)
    for ours, theirs in ("beta_x", "betx"), ("beta_y", "bety"), ("eta_x", "dx"), ("mu_x", "mux"):
        mine, madx = at_positions(optics, table, theirs, ours)
        assert mine == pytest.approx(madx, rel=1e-4, abs=1e-6), ours


def test_ring_optics_equal_xsuites_own_madx_import(ring):
    xt = pytest.importorskip("xtrack")
    lattice, _ = ring
    with tempfile.TemporaryDirectory() as folder:
        (Path(folder) / "optics").mkdir()
        (Path(folder) / "optics" / "strengths.madx").write_text("kqf = 0.5;\nkqd = -0.5;\n")
        with madx_module.Madx(stdout=False, command_log=None) as madx:
            madx.chdir(folder)
            madx.input(RING.replace("stop;", ""))
            reference = xt.Line.from_madx_sequence(madx.sequence["ring"], deferred_expressions=False)
    reference.particle_ref = xt.Particles(p0c=100e9, mass0=xt.PROTON_MASS_EV)
    reference.configure_bend_model(edge="full")  # the standard fidelity's edges
    theirs = reference.twiss(method="4d")
    ours = ENGINE.open(lattice, "ring").optics(OpticsRequest())
    assert (ours.tune_x, ours.tune_y) == pytest.approx((theirs.qx, theirs.qy), abs=1e-10)
    assert (ours.chroma_x, ours.chroma_y) == pytest.approx((theirs.dqx, theirs.dqy), rel=1e-6)


def test_a_corrector_kicks_the_closed_orbit_as_linear_optics_predict(ring):
    lattice, _ = ring
    session = ENGINE.open(lattice, "ring")
    optics = session.optics(OpticsRequest())
    corrector = next(p for p in lattice.layout("ring").placements if p.kind.value == "Kicker")
    before = session.orbit(OrbitRequest())
    kick = corrector.element.MagneticMultipoleP.Kn0L
    session.update({(corrector.id, "MagneticMultipoleP.Kn0L"): kick + 1e-5})  # 1e-5 more toward -x
    after = session.orbit(OrbitRequest())
    dx = np.array(after.x) - np.array(before.x)
    k = optics.occurrences.index(corrector.id)
    q, beta, mu = optics.tune_x, np.array(optics.beta_x), np.array(optics.mu_x)
    phase = np.abs(mu - mu[k]) * 2 * math.pi
    expected = -1e-5 * np.sqrt(beta * beta[k]) * np.cos(phase - math.pi * q) / (2 * math.sin(math.pi * q))
    assert dx == pytest.approx(expected, rel=2e-2, abs=1e-9)  # thick-kick and second-order differences


def test_an_update_equals_a_new_session(ring):
    lattice, _ = ring
    quad = next(p for p in lattice.layout("ring").placements if p.kind.value == "Quadrupole")
    change = {(quad.id, "MagneticMultipoleP.Kn1"): quad.element.MagneticMultipoleP.Kn1 * 1.01}
    session = ENGINE.open(lattice, "ring")
    session.update(change)
    updated = session.optics(OpticsRequest())
    fresh = ENGINE.open(lattice, "ring", values=change).optics(OpticsRequest())
    assert updated.tune_x == pytest.approx(fresh.tune_x, abs=1e-12) and updated.beta_x == pytest.approx(
        fresh.beta_x
    )
    with pytest.raises(UnsupportedRequest, match="cannot change in place"):
        session.update({(quad.id, "BendP.angle_ref"): 0.1})


def test_fidelities_and_effects(ring):
    lattice, _ = ring
    standard = ENGINE.open(lattice, "ring").optics(OpticsRequest())
    for fidelity in (Fidelity.FAST, Fidelity.PRECISE):
        other = ENGINE.open(lattice, "ring", Options(fidelity=fidelity)).optics(
            OpticsRequest(options=Options(fidelity=fidelity))
        )
        assert other.fidelity is fidelity and other.tune_x == pytest.approx(standard.tune_x, abs=1e-3)
    six = Options(effects=frozenset({Effect.LONGITUDINAL}))
    longitudinal = ENGINE.open(lattice, "ring", six).optics(OpticsRequest(options=six))
    assert longitudinal.tune_x == pytest.approx(standard.tune_x, abs=1e-4)
    with pytest.raises(UnsupportedRequest, match="effect space_charge: none .in Xsuite .xfields.; not yet"):
        ENGINE.open(lattice, "ring", Options(effects=frozenset({Effect.SPACE_CHARGE})))
    with pytest.raises(UnsupportedRequest, match="no linear fidelity"):
        ENGINE.open(lattice, "ring", Options(fidelity=Fidelity.LINEAR))


def test_turn_by_turn_tracking_oscillates_at_the_tune(ring):
    lattice, _ = ring
    session = ENGINE.open(lattice, "ring")
    tune = session.optics(OpticsRequest()).tune_x % 1
    bpm = next(p.id for p in lattice.layout("ring").placements if p.kind.value == "Instrument")
    result = session.track(TrackRequest(particles=((1e-4, 0, 0, 0, 0, 0),), turns=256, observe=(bpm,)))
    x = np.array(result.x[0])[:, 0]
    x -= x.mean()  # about the closed orbit, which the ring's corrector offsets
    spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    assert np.argmax(spectrum) / len(x) == pytest.approx(min(tune, 1 - tune), abs=2 / len(x))
    assert result.lost == (False,)


def ramped_and_turn_by_turn(engine, lattice, options):
    """A corrector and a quadrupole ramped while a bunch turns: tracked in one call with a ``Ramp``,
    and by setting each turn's values and tracking that turn. Returns both, the one-call session last."""
    placements = lattice.layout("ring").placements
    corrector = next(p for p in placements if p.kind.value == "Kicker")
    quad = next(p for p in placements if p.kind.value == "Quadrupole")
    kick, k1 = (corrector.id, "MagneticMultipoleP.Kn0L"), (quad.id, "MagneticMultipoleP.Kn1")
    start = {kick: corrector.element.MagneticMultipoleP.Kn0L, k1: quad.element.MagneticMultipoleP.Kn1}
    end = {kick: start[kick] + 1e-4, k1: start[k1] * 1.01}
    turns = 24
    ramp = Ramp(start, end, np.clip((np.arange(turns) - 4) / 16, 0, 1))  # held, ramping, held
    bpms = tuple(p.id for p in placements if p.kind.value == "Instrument")
    bunch = np.random.default_rng(3).normal(0, 1e-4, (20, 6)) * [1, 0.1, 1, 0.1, 0, 1]
    session = engine.open(lattice, "ring", options)
    ramped = session.track_beam(bunch, turns, bpms, options, ramps=[ramp])
    stepped, coordinates, x = engine.open(lattice, "ring", options), bunch, []
    for t in range(turns):
        stepped.update(ramp.at(t))
        one = stepped.track_beam(coordinates, 1, bpms, options)
        x.append(one.x[:, 0])
        coordinates = one.final
    return ramped, np.column_stack(x), coordinates, session, stepped


def test_settings_ramp_turn_by_turn_within_one_call(ring):
    lattice, _ = ring
    ramped, x, final, session, stepped = ramped_and_turn_by_turn(ENGINE, lattice, Options())
    assert np.ptp(x, axis=1).max() > 1e-5  # the bunch felt the ramp
    assert ramped.x == pytest.approx(x, rel=1e-12, abs=1e-15)
    # The turn-by-turn tracking restarts the bunch from its coordinates every turn, which rounds.
    assert ramped.final == pytest.approx(final, abs=1e-11)
    # The session is left where the ramp ended.
    assert session.orbit(OrbitRequest()).x == pytest.approx(stepped.orbit(OrbitRequest()).x, abs=1e-15)
    part = lattice.layout("ring").placements[3].id
    with pytest.raises(UnsupportedRequest, match="settings ramp while a ring turns"):
        ENGINE.open(lattice, "ring").track_beam(
            final, 1, (), Options(), end=part, ramps=[Ramp({}, {}, [0.0])]
        )


def test_a_slow_program_is_followed_by_snapshots(ring):
    lattice, _ = ring
    quad = next(p for p in lattice.layout("ring").placements if p.kind.value == "Quadrupole")
    k1 = quad.element.MagneticMultipoleP.Kn1
    program = Program({(quad.id, "MagneticMultipoleP.Kn1"): [(0.0, k1), (1.0, k1 * 1.02)]})
    session = ENGINE.open(lattice, "ring")
    tunes = [r.tune_x for _, r in quasi_static(session, program, (0.0, 0.5, 1.0))]
    assert tunes[0] < tunes[1] < tunes[2] or tunes[0] > tunes[1] > tunes[2]  # monotonic along the ramp
    assert program.at(0.5) == {(quad.id, "MagneticMultipoleP.Kn1"): pytest.approx(k1 * 1.01)}


@pytest.fixture(scope="module")
def ein():
    from eicvibe.store import is_machine, load_lattice

    if not is_machine(EIN / "machine"):
        pytest.skip("EIN machine folder not found")
    return load_lattice(EIN / "machine")


@pytest.mark.parametrize(("branch", "source"), [("bar", "BAR.madx"), ("rcs", "RCS9GeV40CellV1.madx")])
def test_the_ein_rings_match_madx(ein, branch, source):
    with tempfile.TemporaryDirectory() as folder, madx_module.Madx(stdout=False, command_log=None) as madx:
        madx.chdir(folder)
        madx.call(str(EIN / "EICVIBE_lattice" / "madx" / source))
        madx.use(sequence="ring")
        twiss = madx.twiss()
        table = {k: np.array(twiss[k]) for k in ("s", "betx", "bety", "dx")}
        tunes = (twiss.summary.q1, twiss.summary.q2)
    optics = ENGINE.open(ein, branch).optics(OpticsRequest())
    assert (optics.tune_x, optics.tune_y) == pytest.approx(tunes, abs=1e-6)
    for ours, theirs in ("beta_x", "betx"), ("beta_y", "bety"), ("eta_x", "dx"):
        mine, madx_values = at_positions(optics, table, theirs, ours)
        # Near zeros (dispersion-free straights) compare on the column's scale: 2e-5 of 0.93 m in the RCS.
        assert mine == pytest.approx(madx_values, rel=1e-5, abs=1e-4 * np.abs(madx_values).max()), ours


def test_every_ein_branch_gets_optics_through_its_forks(ein):
    results = machine_optics(ENGINE, ein)
    names = {ein.branch(b).name for b in results}
    assert names == {"ltr", "itl", "bar", "etl", "rcs", "rte"}
    itl, ltr = (results[ein.layout(n).id] for n in ("itl", "ltr"))
    fork = ein.placement("fork_ltr_to_itl", branch="ltr").id
    assert itl.start.beta_x == pytest.approx(ltr.at(fork).beta_x)  # the ITL starts where the LTR forks
    assert all(b > 0 for r in results.values() for b in r.beta_x)


@pytest.mark.parametrize(
    ("zero_phase", "phase"),
    [("ACCELERATING", 0.0), ("BELOW_TRANSITION", math.pi / 2), ("ABOVE_TRANSITION", -math.pi / 2)],
)
def test_every_rf_phase_reference_is_converted(zero_phase, phase):
    from eicvibe.domain.lattice import LatticeBuilder, SequentialIds

    b = LatticeBuilder("rf", ids=SequentialIds())
    b.define("BEG", "BeginningEle", ReferenceP={"species_ref": "electron", "pc_ref": 1e9})
    rf = {"voltage": 1e6, "frequency": 500e6, "zero_phase": zero_phase, "phase": phase}
    b.define("CAV", "RFCavity", length=0.0, RFP=rf)
    b.add_branch("line", "linac", ["BEG", "CAV"])
    six = Options(effects=frozenset({Effect.LONGITUDINAL}))
    orbit = ENGINE.open(b.build(), "line", six).orbit(OrbitRequest(initial=(0, 0, 0, 0, 0, 0), options=six))
    assert orbit.delta[-1] == pytest.approx(
        1e6 / 1e9, rel=1e-6
    )  # the crest: the full voltage, for electrons too


def test_a_bunch_is_seen_as_bpms_see_it(ring):
    lattice, _ = ring
    session = ENGINE.open(lattice, "ring")
    bpms = tuple(p.id for p in lattice.layout("ring").placements if p.kind.value == "Instrument")
    start = np.array([[1e-3, 0.0, 5e-4, 0.0, 0.0, 0.0]])
    one = session.track_beam(start, 64, bpms, Options())
    single = session.track(TrackRequest(particles=(tuple(start[0]),), turns=64, observe=bpms))
    assert one.x == pytest.approx(np.array(single.x)[:, :, 0], abs=1e-15)  # one particle: its own centroid
    assert one.final[0] == pytest.approx(start[0], abs=1e-2) and not one.lost.any()
    spread = start + np.random.default_rng(0).normal(0, 1e-4, (40, 6)) * [1, 0.1, 1, 0.1, 0, 0]
    bunch = session.track_beam(spread, 64, bpms, Options())
    assert (bunch.count == 40).all() and bunch.final.shape == (40, 6)
    assert bunch.x[0, 0] != pytest.approx(one.x[0, 0], abs=1e-9)  # the mean of 40, not of one


def test_a_route_carries_the_beams_optics_across_its_joins():
    from eicvibe.api.demo import ROUTES, demo_lattice
    from eicvibe.domain.lattice import route_length, route_through
    from eicvibe.engines import route_optics

    lattice = demo_lattice()
    route = route_through(lattice, "transfer", ROUTES["transfer"])
    optics = route_optics(ENGINE, lattice, route)
    assert optics.branch == "route:transfer" and not optics.periodic
    assert (optics.start.beta_x, optics.start.beta_y) == pytest.approx((8.0, 4.0))  # the line's own start
    assert np.all(np.diff(optics.s) >= 0) and optics.s[-1] == pytest.approx(route_length(lattice, route))
    from eicvibe.engines.model import initial_conditions

    start = initial_conditions(lattice, "injection")
    injection = ENGINE.open(lattice, "injection").optics(OpticsRequest(initial=start))
    n = len(injection.occurrences)
    assert optics.beta_x[:n] == pytest.approx(injection.beta_x, rel=1e-9)  # the line, as it is alone
    # The ring's part starts from where the line ends: one metre of field-free septum further on.
    b, a = injection.beta_x[-1], injection.alpha_x[-1]
    assert optics.beta_x[n] == pytest.approx(b - 2 * a + (1 + a * a) / b, rel=1e-9)
    ring = ENGINE.open(lattice, "ring").optics(OpticsRequest())
    assert max(optics.beta_x[n:]) != pytest.approx(max(ring.beta_x), rel=1e-3)  # the beam's, not periodic
