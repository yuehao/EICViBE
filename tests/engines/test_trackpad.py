"""The TrackPad engine against Xsuite: one lattice, two engines, the same physics.

Xsuite's conversion is checked against MAD-X (test_xsuite.py), so agreement
here checks TrackPad's conversion. Both run at the precise fidelity, where
both integrate exactly; what still differs is stated where it is tested.
"""

import math
import os
import tempfile
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pytest

if find_spec("juliacall") is None:  # importing juliacall starts Julia: the engine does that itself
    pytest.skip("juliacall is not installed", allow_module_level=True)
pytest.importorskip("xtrack")
madx_module = pytest.importorskip("cpymad.madx")

from eicvibe.engines import machine_optics  # noqa: E402
from eicvibe.engines.contracts import (  # noqa: E402
    Effect,
    EngineUnavailable,
    Fidelity,
    OpticsRequest,
    Options,
    OrbitRequest,
    TrackRequest,
    UnsupportedRequest,
)
from eicvibe.engines.trackpad import TrackPadEngine, _julia  # noqa: E402
from eicvibe.engines.xsuite import XsuiteEngine  # noqa: E402
from eicvibe.importers import import_cpymad  # noqa: E402
from tests.importers.test_madx import RING  # noqa: E402

TRACKPAD, XSUITE = TrackPadEngine(), XsuiteEngine()
PRECISE = Options(fidelity=Fidelity.PRECISE)
EIN = Path(os.environ.get("EICVIBE_EIN_FOLDER", "/Users/haoyue/Research/EIC/virtual_accelerator/EIN"))


@pytest.fixture(scope="module", autouse=True)
def julia():
    try:
        _julia()
    except EngineUnavailable as error:
        pytest.skip(str(error))


def line_of(definition: str, length: float):
    """A line holding one MAD-X element at its start."""
    name = definition.partition(":")[0]
    with madx_module.Madx(stdout=False, command_log=None) as madx:
        madx.input(
            f"{definition}\nline: sequence, l={length + 0.1}, refer=entry; {name}, at=0; endsequence;\n"
            "beam, sequence=line, particle=electron, energy=18;\nuse, sequence=line;"
        )
        return import_cpymad(madx, ["line"], kind="linac", name="one").lattice


SIX = Options(fidelity=Fidelity.PRECISE, effects=frozenset({Effect.LONGITUDINAL}))


def end_of(engine, lattice, start):
    orbit = engine.open(lattice, "line", SIX).orbit(OrbitRequest(initial=start, options=SIX))
    return np.array([orbit.x[-1], orbit.px[-1], orbit.y[-1], orbit.py[-1], orbit.zeta[-1], orbit.delta[-1]])


START = (1e-3, 2e-4, -5e-4, 1e-4, 2e-3, 1e-3)

# Thin kicks, frame changes and RF: the same maps in both engines, to round-off.
EXACT = {
    "multipole": ("mp: multipole, knl={0, 0.01, 0.2, 0.5}, ksl={0, 0.005, 0, 0.1};", 0.0),
    "hkicker": ("hc: hkicker, l=0.1, kick=1e-4;", 0.1),
    "vkicker": ("vc: vkicker, kick=2e-4, tilt=0.1;", 0.0),
    "kicker": ("kk: kicker, hkick=3e-4, vkick=-4e-4;", 0.0),
    "monitor": ("bpm: monitor, l=0.1;", 0.1),
    "srotation": ("sr: srotation, angle=0.01;", 0.0),
    "yrotation": ("yr: yrotation, angle=0.02;", 0.0),
    "xrotation": ("xr: xrotation, angle=0.015;", 0.0),
    "translation": ("tr: translation, dx=0.001, dy=-0.0005, ds=0.002;", 0.0),
    "rfcavity": ("cav: rfcavity, l=1, volt=5, lag=0.1, freq=591;", 1.0),
}

# Thick magnets: TrackPad integrates in 40 fourth-order steps, Xsuite with its own maps
# (exact bend bodies, thick quadrupoles, kicks in sextupoles), within 1e-3 of the change.
THICK = {
    "sbend": ("b: sbend, l=2, angle=0.1, e1=0.02, fint=0.5, hgap=0.03, k1=0.01, tilt=0.2;", 2.0),
    "rbend": ("rb: rbend, l=2, angle=0.1;", 2.0),
    "quadrupole": ("q: quadrupole, l=0.5, k1=1.2, tilt=0.1;", 0.5),
    "sextupole": ("s: sextupole, l=0.3, k2=3.0;", 0.3),
    "octupole": ("o: octupole, l=0.3, k3=50.0, tilt=0.05;", 0.3),
    "solenoid": ("sol: solenoid, l=2, ks=0.1;", 2.0),
}


@pytest.mark.parametrize("case", EXACT)
def test_thin_kicks_frames_and_rf_equal_xsuites(case):
    lattice = line_of(*EXACT[case])
    assert end_of(TRACKPAD, lattice, START) == pytest.approx(end_of(XSUITE, lattice, START), abs=1e-15)


@pytest.mark.parametrize("case", THICK)
def test_thick_magnets_agree_with_xsuite(case):
    lattice = line_of(*THICK[case])
    ours, theirs = end_of(TRACKPAD, lattice, START), end_of(XSUITE, lattice, START)
    change = np.max(np.abs(theirs - np.array(START)))
    assert np.max(np.abs(ours - theirs)) < 1e-3 * change


@pytest.mark.parametrize("zero_phase", ["ACCELERATING", "BELOW_TRANSITION", "ABOVE_TRANSITION"])
def test_every_rf_phase_reference_equals_xsuites(zero_phase):
    from eicvibe.domain.lattice import LatticeBuilder, SequentialIds

    b = LatticeBuilder("rf", ids=SequentialIds())
    b.define("BEG", "BeginningEle", ReferenceP={"species_ref": "electron", "pc_ref": 1e9})
    rf = {"voltage": 1e6, "frequency": 500e6, "zero_phase": zero_phase, "phase": 0.3}
    b.define("CAV", "RFCavity", length=0.5, RFP=rf)
    b.add_branch("line", "linac", ["BEG", "CAV"])
    lattice = b.build()
    assert end_of(TRACKPAD, lattice, START) == pytest.approx(end_of(XSUITE, lattice, START), abs=1e-15)


def test_a_combined_function_bend_agrees_with_xsuite():
    # A BAR bend: 45 degrees at rho = 1.9 m with a gradient. Both engines kick with the
    # curved-frame field, which a straight-frame kick misses by 1e-3 of the deflection.
    start = (1e-3, 0.0, 1e-3, 0.0, 0.0, 1e-3)
    lattice = line_of("b: sbend, l=1.5, angle=0.7854, k1=-0.5;", 1.5)
    ours, theirs = end_of(TRACKPAD, lattice, start), end_of(XSUITE, lattice, start)
    assert np.max(np.abs(ours - theirs)[:4]) < 1e-6 * np.max(np.abs(theirs[:4] - np.array(start[:4])))


@pytest.fixture(scope="module")
def ring():
    with tempfile.TemporaryDirectory() as folder:
        (Path(folder) / "optics").mkdir()
        (Path(folder) / "optics" / "strengths.madx").write_text("kqf = 0.5;\nkqd = -0.5;\n")
        with madx_module.Madx(stdout=False, command_log=None) as madx:
            madx.chdir(folder)
            madx.input(RING.replace("stop;", ""))
            return import_cpymad(madx, ["ring"], kind="periodic", name="ring").lattice


def compare_optics(ours, theirs, *, beta=1e-5, chroma=1e-3):
    assert (ours.tune_x, ours.tune_y) == pytest.approx((theirs.tune_x, theirs.tune_y), abs=1e-6)
    for column in ("beta_x", "beta_y"):
        assert getattr(ours, column) == pytest.approx(getattr(theirs, column), rel=beta), column
    scale = np.max(np.abs(theirs.eta_x))
    assert ours.eta_x == pytest.approx(theirs.eta_x, abs=beta * scale)
    if chroma is not None:
        assert (ours.chroma_x, ours.chroma_y) == pytest.approx((theirs.chroma_x, theirs.chroma_y), rel=chroma)


def test_ring_optics_equal_xsuites(ring):
    ours = TRACKPAD.open(ring, "ring", PRECISE).optics(OpticsRequest(options=PRECISE))
    theirs = XSUITE.open(ring, "ring", PRECISE).optics(OpticsRequest(options=PRECISE))
    compare_optics(ours, theirs)  # chromaticity -1.0207, -1.4102, as MAD-X's
    assert ours.momentum_compaction == pytest.approx(theirs.momentum_compaction, rel=1e-6)
    # The corrector's 4D closed orbit, 5e-5 m: the exact bends' edge maps part at 2e-5 of it.
    assert ours.x == pytest.approx(theirs.x, abs=1e-4 * np.max(np.abs(theirs.x)))


def test_the_fast_fidelity_is_approximate(ring):
    fast = Options(fidelity=Fidelity.FAST)
    ours = TRACKPAD.open(ring, "ring", fast).optics(OpticsRequest(options=fast))
    theirs = XSUITE.open(ring, "ring", PRECISE).optics(OpticsRequest(options=PRECISE))
    assert ours.tune_x == pytest.approx(theirs.tune_x, abs=1e-3)
    # The AT-style bend lacks the curvature terms that carry half this ring's chromaticity.
    assert ours.chroma_x == pytest.approx(2 * theirs.chroma_x, rel=0.05)


@pytest.fixture(scope="module")
def ein():
    from eicvibe.store import is_machine, load_lattice

    if not is_machine(EIN / "machine"):
        pytest.skip("EIN machine folder not found")
    return load_lattice(EIN / "machine")


@pytest.mark.parametrize("branch", ["bar", "rcs"])
def test_the_ein_rings_agree_with_xsuite(ein, branch):
    ours = TRACKPAD.open(ein, branch, PRECISE).optics(OpticsRequest(options=PRECISE))
    theirs = XSUITE.open(ein, branch, PRECISE).optics(OpticsRequest(options=PRECISE))
    # The BAR's chromaticity is compared with MAD-X below: Xsuite's edge model moves its vertical one.
    compare_optics(ours, theirs, chroma=1e-2 if branch == "rcs" else None)


def test_the_bar_chromaticity_is_madxs(ein):
    with tempfile.TemporaryDirectory() as folder, madx_module.Madx(stdout=False, command_log=None) as madx:
        madx.chdir(folder)
        madx.call(str(EIN / "EICVIBE_lattice" / "madx" / "BAR.madx"))
        madx.use(sequence="ring")
        summary = madx.twiss(chrom=True).summary
    ours = TRACKPAD.open(ein, "bar", PRECISE).optics(OpticsRequest(options=PRECISE))
    # MAD-X (-0.0612, -0.0843); Xsuite gives (-0.0557, +0.0794) through its edges.
    assert (ours.chroma_x, ours.chroma_y) == pytest.approx((summary.dq1, summary.dq2), abs=1e-3)


def test_every_ein_branch_agrees_with_xsuite(ein):
    ours = machine_optics(TRACKPAD, ein, options=PRECISE)
    theirs = machine_optics(XSUITE, ein, options=PRECISE)
    assert set(ours) == set(theirs) == {b.id for b in ein.branches}
    for branch in ours:
        # The RTE runs 4 cm off axis after its extraction patch, where the bend models part by 1e-3.
        rel = 3e-3 if ein.branch(branch).name == "rte" else 1e-4
        assert ours[branch].beta_x == pytest.approx(theirs[branch].beta_x, rel=rel)
        assert ours[branch].beta_y == pytest.approx(theirs[branch].beta_y, rel=rel)


@pytest.mark.parametrize("fidelity", [Fidelity.FAST, Fidelity.PRECISE])
def test_sampling_inside_magnets_adds_points_and_changes_nothing(ein, fidelity):
    options = Options(fidelity=fidelity)
    session = TRACKPAD.open(ein, "bar", options)
    plain = session.optics(OpticsRequest(options=options))
    sampled = session.optics(OpticsRequest(options=options, inside=True))
    assert plain.profile is None and sampled.profile is not None
    for column in ("beta_x", "beta_y", "alpha_x", "eta_x", "mu_x", "mu_y"):
        assert getattr(sampled, column) == pytest.approx(getattr(plain, column), abs=1e-10), column
    profile = sampled.profile
    steps = {Fidelity.FAST: 2, Fidelity.PRECISE: 40}[fidelity]
    quadrupoles = [i for i, p in enumerate(ein.layout("bar").placements) if p.kind.value == "Quadrupole"]
    assert all(profile.index.count(i) == steps for i in quadrupoles)  # one point per integrator step
    assert list(profile.index) == sorted(profile.index) and np.all(np.diff(profile.s) >= 0)
    ends = np.searchsorted(profile.index, np.arange(len(sampled.occurrences)), side="right")
    assert np.array(profile.beta_x)[ends] == pytest.approx(np.array(sampled.beta_x), abs=1e-12)


def test_a_corrector_moves_the_closed_orbit_as_in_xsuite(ring):
    corrector = next(p for p in ring.layout("ring").placements if p.kind.value == "Kicker")
    change = {(corrector.id, "MagneticMultipoleP.Kn0L"): corrector.element.MagneticMultipoleP.Kn0L + 1e-5}
    moved = {}
    for engine in (TRACKPAD, XSUITE):
        session = engine.open(ring, "ring", PRECISE)
        before = np.array(session.orbit(OrbitRequest(options=PRECISE)).x)
        session.update(change)
        moved[engine.name] = np.array(session.orbit(OrbitRequest(options=PRECISE)).x) - before
    assert moved["trackpad"] == pytest.approx(moved["xsuite"], rel=1e-5, abs=1e-12)


def test_an_update_equals_a_new_session(ring):
    quad = next(p for p in ring.layout("ring").placements if p.kind.value == "Quadrupole")
    change = {(quad.id, "MagneticMultipoleP.Kn1"): quad.element.MagneticMultipoleP.Kn1 * 1.01}
    session = TRACKPAD.open(ring, "ring")
    session.update(change)
    updated = session.optics(OpticsRequest())
    fresh = TRACKPAD.open(ring, "ring", values=change).optics(OpticsRequest())
    assert updated.tune_x == pytest.approx(fresh.tune_x, abs=1e-12)
    with pytest.raises(UnsupportedRequest, match="cannot change in place"):
        session.update({(quad.id, "BendP.angle_ref"): 0.1})


def test_turn_by_turn_tracking_oscillates_at_the_tune_and_loses_what_it_should(ring):
    session = TRACKPAD.open(ring, "ring")
    tune = session.optics(OpticsRequest()).tune_x % 1
    bpm = next(p.id for p in ring.layout("ring").placements if p.kind.value == "Instrument")
    particles = ((1e-4, 0, 0, 0, 0, 0), (1.5, 0, 0, 0, 0, 0))  # beyond TrackPad's 1 m coordinate limit
    result = session.track(TrackRequest(particles=particles, turns=256, observe=(bpm,)))
    x = np.array(result.x[0])[:, 0]
    x -= x.mean()
    spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    assert np.argmax(spectrum) / len(x) == pytest.approx(min(tune, 1 - tune), abs=2 / len(x))
    assert result.lost == (False, True)
    assert math.isnan(result.x[0][-1][1])


def test_the_longitudinal_motion_is_in_orbits_and_tracking_not_in_optics(ring):
    six = SIX
    session = TRACKPAD.open(ring, "ring", six)
    ours = session.orbit(OrbitRequest(options=six))
    theirs = XSUITE.open(ring, "ring", six).orbit(OrbitRequest(options=six))
    flat = session.orbit(OrbitRequest(options=PRECISE))
    # The corrector lengthens the orbit; the RF lowers the momentum to keep it synchronous,
    # and the dispersion takes a fifth off the orbit (MAD-X's TWISS: 4.4585e-5 m here).
    assert ours.delta[0] == pytest.approx(-1.99e-6, rel=0.01) and flat.delta[0] == 0.0
    assert ours.x[0] == pytest.approx(0.84 * flat.x[0], rel=0.01)
    # Xsuite's 6D orbit is 1e-4 off MAD-X's (TrackPad's within 1e-8): compare on the orbit's scale.
    assert ours.x == pytest.approx(theirs.x, abs=1e-3 * np.max(np.abs(theirs.x)))
    assert ours.delta == pytest.approx(theirs.delta, rel=1e-3)
    with pytest.raises(UnsupportedRequest, match="effect longitudinal in optics: none .TrackPad.s Twiss"):
        session.optics(OpticsRequest(options=six))
    with pytest.raises(UnsupportedRequest, match="effect radiation_damping: none"):
        TRACKPAD.open(ring, "ring", Options(effects=frozenset({Effect.RADIATION_DAMPING})))
    with pytest.raises(UnsupportedRequest, match="no linear fidelity"):
        TRACKPAD.open(ring, "ring", Options(fidelity=Fidelity.LINEAR))
