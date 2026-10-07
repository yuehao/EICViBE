import math

import pytest

madx_module = pytest.importorskip("cpymad.madx")

from eicvibe.domain.lattice import BranchKind, ElementKind, diff  # noqa: E402
from eicvibe.importers import Severity, UnsupportedElementError, import_cpymad, import_madx  # noqa: E402

RING = """
! FODO ring built in a loop and matched to its tunes.
call, file="optics/strengths.madx";
setq(newqx, newqy): macro = { qx_target = newqx; qy_target = newqy; };
exec, setq(2.1, 2.05);

mq: quadrupole, l=0.5;
qf: mq, k1:=kqf;
qd: mq, k1:=kqd;
b: sbend, l=2.0, angle=2*pi/16;
hc: hkicker, kick=1e-5;
bpm: monitor;
cav: rfcavity, l=0.5, volt=1, lag=0.5, harmon=10;

ring: sequence, l=80, refer=entry;
endsequence;
seqedit, sequence=ring;
n = 0;
while (n < 8) {
  install, element=qf, at=10*n;
  install, element=b, at=10*n + 1.5;
  install, element=qd, at=10*n + 5;
  install, element=bpm, at=10*n + 5.5;
  install, element=b, at=10*n + 6.5;
  n = n + 1;
}
install, element=hc, at=9.5;
install, element=cav, at=79;
endedit;

beam, particle=proton, energy=100;
use, sequence=ring;
match, sequence=ring;
  vary, name=kqf, step=1e-6;
  vary, name=kqd, step=1e-6;
  global, q1=qx_target, q2=qy_target;
  lmdif, calls=500, tolerance=1e-14;
endmatch;
stop;
"""

ELEMENTS = """
b: sbend, l=2, angle=0.1, e1=0.02, fint=0.5, hgap=0.03, k1=0.01, tilt=0.2;
rb: rbend, l=2, angle=0.1;
q: quadrupole, l=0.5, k1=1.2, tilt=0.1, knl={0, 0, 0.03};
mp: multipole, knl={0, 0.01, 0.2, 0, 0, 0, 7, 0, 0, 0, 0.5, 9}, ksl={0, 0.005};
hc: hkicker, l=0.1, kick=1e-4;
vc: vkicker, kick=2e-4;
kk: kicker, hkick=3e-4, vkick=-4e-4;
cav: rfcavity, l=1, volt=5, lag=0.25, harmon=7560;
crab: crabcavity, l=1, volt=2, freq=394, tilt=pi/2;
sol: solenoid, l=2, ks=0.1;
bpm: monitor, l=0.1;
col: rcollimator, l=0.5, apertype=rectangle, aperture={0.01, 0.02};
sr: srotation, angle=0.01;
yr: yrotation, angle=0.02;
tr: translation, dx=0.001, ds=0.002;
de: dipedge, h=0.05, e1=0.02;
screen: marker, apertype=rectellipse, aperture={0.02, 0.02, 0.025, 0.025}, aper_offset={0.001, 0};
line: sequence, l=30, refer=entry;
  b, at=0; rb, at=3; q, at=6; mp, at=7; hc, at=8; vc, at=9; kk, at=10; cav, at=11; crab, at=13;
  sol, at=15; bpm, at=18; col, at=19; sr, at=20; yr, at=20; tr, at=20; de, at=21; screen, at=22;
endsequence;
beam, sequence=line, particle=electron, energy=18;
use, sequence=line;
"""


@pytest.fixture
def ring_script(tmp_path):
    (tmp_path / "optics").mkdir()
    (tmp_path / "optics" / "strengths.madx").write_text("kqf = 0.5;\nkqd = -0.5;\n")
    script = tmp_path / "ring.madx"
    script.write_text(RING)
    return script


@pytest.fixture
def elements():
    with madx_module.Madx(stdout=False) as madx:
        madx.input(ELEMENTS)
        yield import_cpymad(madx, kind="linac", name="elements")


def matched(script) -> tuple[float, float]:
    with madx_module.Madx(stdout=False) as madx:
        with madx.chdir(str(script.parent)):
            madx.call(script.name)
        return madx.globals.kqf, madx.globals.kqd


def test_script_runs_with_calls_macros_loops_and_matching(ring_script):
    result = import_madx(ring_script)
    lattice = result.lattice
    assert result.warnings == ()
    kqf, kqd = matched(ring_script)
    assert kqf != 0.5  # MATCH changed the strengths
    assert lattice.element("qf").MagneticMultipoleP.Kn1 == kqf
    assert lattice.element("qd").MagneticMultipoleP.Kn1 == kqd
    assert lattice.element("qf").tags == ("mq",)

    layout = lattice.layout("ring")
    assert lattice.name == "ring" and layout.periodic and layout.length == pytest.approx(80.0)
    beginning = layout.placements[0].element
    assert beginning.kind is ElementKind.BEGINNING_ELE
    assert beginning.ReferenceP.species_ref == "proton"
    assert beginning.ReferenceP.pc_ref == pytest.approx(math.sqrt(100e9**2 - 938.272e6**2), rel=1e-6)
    names = [p.name for p in layout.placements]
    assert names.count("qf_1") == 1 and "qf_8" in names and "hc" in names
    assert sum(p.kind is ElementKind.BEND for p in layout.placements) == 16


def madx_rows(table) -> list[tuple[float, float]]:
    """(s at exit, length) of each row of a MAD-X table that the importer places."""
    rows = zip(table.name, table.keyword, table.s, table.l, strict=True)
    return [
        (s, length)
        for name, kind, s, length in rows
        if "$" not in name and not (kind == "drift" and length == 0)
    ]


@pytest.mark.parametrize(
    ("script", "sequence", "initial"),
    [(RING, "ring", {}), (ELEMENTS, "line", {"betx": 10.0, "bety": 10.0})],
    ids=["ring", "elements"],
)
def test_lengths_match_madx_twiss_and_survey(tmp_path, script, sequence, initial):
    (tmp_path / "optics").mkdir()
    (tmp_path / "optics" / "strengths.madx").write_text("kqf = 0.5;\nkqd = -0.5;\n")
    (tmp_path / "lattice.madx").write_text(script)
    with madx_module.Madx(stdout=False) as madx:
        with madx.chdir(str(tmp_path)):
            madx.call("lattice.madx")
        layout = import_cpymad(madx, [sequence]).lattice.layout(sequence)
        ours = [(p.s_exit, p.length) for p in layout.placements[1:]]  # placements[0] stands for $start
        twiss, survey = madx.twiss(sequence=sequence, **initial), madx.survey(sequence=sequence)
        for table in (twiss, survey):
            assert ours == pytest.approx(madx_rows(table), rel=1e-12, abs=1e-12)
        assert layout.length == pytest.approx(survey.s[-1], rel=1e-12)
        assert layout.length == pytest.approx(twiss.summary.length, rel=1e-9)  # the summary is rounded


def test_reimport_keeps_ids_so_diff_shows_only_new_strengths(ring_script):
    first = import_madx(ring_script).lattice
    assert import_madx(ring_script).lattice.digest == first.digest
    ring_script.write_text(RING.replace("exec, setq(2.1, 2.05);", "exec, setq(2.15, 2.1);"))
    changes = diff(first, import_madx(ring_script).lattice)
    assert {(c.entity, c.id, c.field) for c in changes} == {
        ("element", first.element("qf").id, "MagneticMultipoleP.Kn1"),
        ("element", first.element("qd").id, "MagneticMultipoleP.Kn1"),
    }


def test_magnets_and_bends(elements):
    lattice = elements.lattice
    bend = lattice.element("b")
    assert (bend.BendP.angle_ref, bend.BendP.e1, bend.BendP.tilt_ref) == (0.1, 0.02, 0.2)
    assert bend.BendP.edge1_int == bend.BendP.edge2_int == pytest.approx(0.015)
    assert bend.MagneticMultipoleP.Kn1 == 0.01 and bend.MagneticMultipoleP.tilt1 == 0.0

    rbend = lattice.element("rb")
    assert rbend.kind is ElementKind.BEND and rbend.length == pytest.approx(2 * 0.05 / math.sin(0.05))
    assert rbend.BendP.e1 == rbend.BendP.e2 == pytest.approx(0.05)

    quad = lattice.element("q").MagneticMultipoleP
    assert (quad.Kn1, quad.Kn2L, quad.tilt1, quad.tilt2) == (1.2, 0.03, 0.1, 0.1)

    thin = lattice.element("mp")
    assert thin.kind is ElementKind.MULTIPOLE and thin.length == 0.0
    multipoles = thin.MagneticMultipoleP
    assert (multipoles.Kn1L, multipoles.Kn2L, multipoles.Ks1L) == (0.01, 0.2, 0.005)
    assert (multipoles.Kn6L, multipoles.Kn10L) == (7.0, 0.5)
    [dropped] = [n for n in elements.warnings if n.subject == "mp"]
    assert dropped.message.startswith("KNL[11] not imported")

    assert lattice.element("sol").SolenoidP.Ksol == 0.1


def test_kickers_take_the_multipole_sign(elements):
    lattice = elements.lattice
    assert lattice.element("hc").MagneticMultipoleP.Kn0L == -1e-4
    assert lattice.element("hc").tags == ("HCOR",)
    assert lattice.element("vc").MagneticMultipoleP.Ks0L == 2e-4
    kicker = lattice.element("kk").MagneticMultipoleP
    assert (kicker.Kn0L, kicker.Ks0L) == (-3e-4, -4e-4)
    assert any(n.code == "kick-sign" for n in elements.notes)


def test_rf_units_and_phase(elements):
    cavity = elements.lattice.element("cav").RFP
    assert (cavity.voltage, cavity.harmon, cavity.frequency) == (5e6, 7560, None)
    assert cavity.phase == pytest.approx(math.pi / 2) and cavity.zero_phase == "BELOW_TRANSITION"
    crab = elements.lattice.element("crab")
    assert crab.kind is ElementKind.CRABCAVITY and crab.RFP.frequency == pytest.approx(394e6)
    assert any(n.subject == "crab" and "TILT" in n.message for n in elements.warnings)


def test_frame_changes_become_patches(elements):
    lattice = elements.lattice
    assert lattice.element("sr").PatchP.z_rot == 0.01
    assert lattice.element("yr").PatchP.y_rot == 0.02
    shift = lattice.element("tr").PatchP
    assert (shift.x_offset, shift.y_offset, shift.z_offset) == (0.001, 0.0, 0.002)


def test_apertures(elements):
    collimator = elements.lattice.element("col")
    assert collimator.kind is ElementKind.DRIFT and collimator.length == 0.5
    aperture = collimator.ApertureP
    assert (aperture.shape, aperture.x_max, aperture.y_min) == ("RECTANGULAR", 0.01, -0.02)

    screen = elements.lattice.element("screen").ApertureP
    assert screen.shape == "VERTICES" and screen.vertices.center == (0.001, 0.0)
    points = [v.point for v in screen.vertices.list]
    assert (
        len(points) == 8
        and points[0] == pytest.approx((0.02, -0.015))
        and points[2] == pytest.approx((0.015, 0.02))
    )
    assert screen.vertices.list[0].radius == (0.025, 0.025) and screen.vertices.list[1].radius is None


def test_beam_positions_and_unsupported_elements(elements):
    lattice = elements.lattice
    layout = lattice.layout("line")
    assert layout.branch.kind is BranchKind.LINAC and layout.length == pytest.approx(30.0)
    assert layout.placements[0].element.ReferenceP.species_ref == "electron"
    assert not any(n.code in ("position", "length") for n in elements.warnings)
    assert lattice.element("de").kind is ElementKind.MARKER
    assert [n.subject for n in elements.warnings if n.code == "unsupported"] == ["de"]
    assert any(n.code == "no-twiss" and n.severity is Severity.INFO for n in elements.notes)


def test_session_without_sequences():
    with madx_module.Madx(stdout=False) as madx, pytest.raises(ValueError, match="no sequence"):
        import_cpymad(madx)


def test_strict_mode_rejects_unsupported_elements():
    with madx_module.Madx(stdout=False) as madx:
        madx.input(ELEMENTS)
        with pytest.raises(UnsupportedElementError, match="DIPEDGE 'de'"):
            import_cpymad(madx, strict=True)


def test_several_sequences_share_definitions_and_ions_are_matched_by_mass():
    with madx_module.Madx(stdout=False) as madx:
        madx.input("""
        q: quadrupole, l=0.5, k1=0.3;
        hsr: sequence, l=10; q, at=1; endsequence;
        tl: sequence, l=4; q, at=2; endsequence;
        beam, sequence=hsr, particle=ion, mass=183.4335, charge=79, energy=197*110;
        """)
        with pytest.raises(KeyError, match="esr"):
            import_cpymad(madx, ["esr"])
        result = import_cpymad(madx, ["HSR", "tl"], kind={"tl": "linac"})
    lattice = result.lattice
    assert lattice.branch(lattice.root_branch).name == "hsr"
    assert [b.kind for b in lattice.branches] == [BranchKind.PERIODIC, BranchKind.LINAC]
    assert lattice.placement("q", branch="hsr").definition is lattice.placement("q", branch="tl").definition
    # MAD-X numbers implicit drifts per sequence; each sequence keeps its own.
    assert lattice.layout("tl").length == pytest.approx(4.0)
    assert lattice.layout("hsr").placements[0].element.ReferenceP.species_ref == "gold197"
    assert [(n.code, n.subject) for n in result.warnings] == [("no-beam", "tl")]


def test_imported_ring_gets_a_registry_whose_bindings_resolve(ring_script):
    from eicvibe.domain.machine import NamingArea, generate_registry

    lattice = import_madx(ring_script).lattice
    registry = generate_registry(lattice, areas=[NamingArea(prefix="rcb1", start=0.0, end=80.0)])
    assert registry.check_against(lattice) == []
    names = {c.name for c in registry.channels}
    assert {
        "rcb1-q1-ps:sp",
        "rcb1-q6-ps:sp",
        "rcb1-th10-ps:sp",
        "rcb1-cav80:sp:volt",
        "rcb-btun-rdx",
    } <= names
    assert sum(name.endswith(":rdx") and "-b" in name for name in names) == 8
