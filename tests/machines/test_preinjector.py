"""The EIN preinjector (ADR 0012), from its designer's survey and PARMELA deck in the EIN folder."""

import math
import os
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from eicvibe.domain.lattice import ElementKind
from eicvibe.domain.machine.naming import LATTICE_NAME
from eicvibe.geometry.floor import floor_geometry
from eicvibe.machines.preinjector import build_preinjector, preinjector_beam

FOLDER = Path(os.environ.get("EICVIBE_EIN_FOLDER", "/Users/haoyue/Research/EIC/virtual_accelerator/EIN"))
pytestmark = pytest.mark.skipif(
    not (FOLDER / "einlattices/Preinjector/parmela/rr6_sep.inp").exists(), reason="EIN folder not found"
)


@pytest.fixture(scope="module")
def pre():
    return build_preinjector(FOLDER)


def test_the_main_line_is_the_survey_with_the_decks_strengths(pre):
    lattice = pre.lattice
    main = lattice.layout("prj")
    kinds = Counter(p.kind for p in main.placements)
    assert main.length == pytest.approx(76.76491)  # the cathode to where the survey connects the LTR
    assert (kinds[ElementKind.SOLENOID], kinds[ElementKind.QUADRUPOLE], kinds[ElementKind.RFCAVITY]) == (
        16,
        27,
        17,
    )
    placed = {p.name: p for p in main.placements}
    # The gun, from the cathode through the survey's first drift: the deck's map scaled to 319.9 kV, DC.
    gun = placed["prj-gun30"]
    assert gun.kind is ElementKind.EGUN and (gun.s_enter, gun.length) == pytest.approx((0.0, 0.307))
    assert (gun.element.RFP.voltage, gun.element.RFP.frequency) == (pytest.approx(319.9e3), None)
    assert main.placements[1].name == "prj-cc0" and main.placements[2] is gun
    # Fields as designed: a quadrupole's gradient (PARMELA's focusing sign reversed for electrons) ...
    assert placed["prj-q1154"].element.MagneticMultipoleP.Bn1 == pytest.approx(-1.72)
    # ... the bucking pairs of coils, each on its own supply ...
    assert [placed[f"prj-sol175{s}"].element.SolenoidP.Bsol for s in "ab"] == pytest.approx(
        [0.011723, -0.011723]
    )
    # ... and the gun dipole, bending to +x (a negative PALS angle) at the gun's 320 keV.
    dipole = placed["prj-d144"].element.BendP
    assert (dipole.angle_ref, dipole.e1) == pytest.approx((-math.radians(16), -math.radians(4)))
    assert pre.gun_energy == pytest.approx(320.37e3)
    cavities = [p.element.RFP for p in main.placements if p.kind is ElementKind.RFCAVITY]
    assert sorted({round(rf.frequency / 1e6, 3) for rf in cavities}) == [197.055, 1300.352, 2856.0]
    structures = [rf for rf in cavities if rf.frequency == 2856e6]
    assert len(structures) == 14 and all(
        rf.voltage == pytest.approx(17.8e6 * 2.95546, rel=1e-4) for rf in structures
    )
    assert all(-math.pi < rf.phase <= math.pi for rf in cavities)
    assert all(not p.element.MagneticMultipoleP for p in main.placements if p.kind is ElementKind.KICKER)


def test_names_are_eic_names_and_each_is_used_once(pre):
    names = list(pre.names.values())
    assert all(LATTICE_NAME.fullmatch(n) for n in names) and len(set(names)) == len(names)
    assert pre.names[("prj", "prj-sol175a")] == "prj-sol175:1"  # coils of one solenoid: append numbers
    assert pre.names[("prj_gun", "prj-thv144")] == "prj-thv144:2"  # the gun line's own steerer
    assert ("prj_gun", "prj-d144") not in pre.names  # one magnet: named at its main-line traversal


def test_the_floor_is_the_surveys(pre):
    lattice = pre.lattice
    geometry = floor_geometry(lattice)

    def end(branch: str) -> tuple[float, float]:
        frame = geometry.branches[lattice.layout(branch).id].frames[-1].exit
        return 100 * frame.r[0], 100 * frame.r[2]  # cm, from the cathode

    assert end("prj") == pytest.approx((2078.79, 7384.34), abs=0.2)  # where it meets the LTR
    assert end("prj_gun") == pytest.approx((-41.35, 278.99), abs=0.2)  # the Faraday cup
    # The Mott polarimeter: 121.12 cm aside and 327.07 cm on from the solenoid the spin line starts at,
    # which is at (179.48, 760.64) cm on the main line, heading 16 degrees.
    c, s = math.cos(math.radians(16)), math.sin(math.radians(16))
    mott = (179.475 + 121.124 * c + 327.071 * s, 760.638 - 121.124 * s + 327.071 * c)
    assert end("prj_spin") == pytest.approx(mott, abs=0.2)
    assert geometry.closures == ()


def test_the_lattice_engines_leave_it_to_an_injector_code(pre):
    from eicvibe.engines import engine, machine_optics
    from eicvibe.engines.contracts import UnsupportedRequest

    with pytest.raises(UnsupportedRequest, match="electron gun") as refused:
        engine("xsuite").open(pre.lattice, pre.lattice.layout("prj").id)
    assert "given as fields" in str(refused.value)
    notes = {}
    machine_optics(engine("xsuite"), pre.lattice, notes=notes)
    assert notes[pre.lattice.layout("prj").id].startswith("not for a lattice engine: prj has an electron gun")


def test_it_delivers_parmelas_beam_to_the_ltr():
    pc = 749999825.92  # the LTR's reference
    p = preinjector_beam(FOLDER, pc)
    assert p.shape == (19998, 6)

    def twiss(u, up):
        u, up = u - u.mean(), up - up.mean()
        e = math.sqrt((u * u).mean() * (up * up).mean() - (u * up).mean() ** 2)
        return e, (u * u).mean() / e, -(u * up).mean() / e

    # PARMELA's interface (linac_interface.txt), which the LTR starts from: emittances 0.0529 and
    # 0.0315 mm mrad, beta 9.07 and 20.39 m, alpha -0.588 and -0.748; 0.913 mm long.
    assert twiss(p[:, 0], p[:, 1]) == pytest.approx((5.29e-8, 9.07, -0.588), rel=0.01)
    assert twiss(p[:, 2], p[:, 3]) == pytest.approx((3.15e-8, 20.39, -0.748), rel=0.01)
    assert p[:, 4].std() == pytest.approx(0.913e-3, rel=1e-3)
    assert abs(p[:, 5].mean()) < 1e-6 and p[:, 5].std() == pytest.approx(1.72 / 750.0, rel=0.01)


@pytest.mark.skipif(
    shutil.which("madx") is None and not (FOLDER / "EICVIBE_lattice/madx").exists(), reason="no MAD-X files"
)
def test_it_joins_a_machine_folder_and_leaves_the_rest_as_it_was(tmp_path):
    pytest.importorskip("cpymad.madx")
    from eicvibe.machines.ein import add_ein_preinjector, build_ein, write_ein_machine
    from eicvibe.store import check_machine, load_machine

    folder = tmp_path / "machine"
    write_ein_machine(build_ein(FOLDER, preinjector=None), folder)
    before = {p.relative_to(folder): p.read_text() for p in folder.rglob("*.yaml")}
    add_ein_preinjector(folder, FOLDER)
    after = {p.relative_to(folder): p.read_text() for p in folder.rglob("*.yaml")}
    for path, text in before.items():
        if path.parts[0] == "lattice" and path.name != "shared.yaml":
            assert after[path] == text, path  # every branch file the folder had, unchanged
        elif path.name in ("power.yaml", "diagnostics.yaml", "rf.yaml"):
            assert after[path].startswith(text.rstrip("\n").rsplit("\n", 1)[0]), path  # only added to
    machine = load_machine(folder)
    assert {"prj", "prj_gun", "prj_spin"} <= {b.name for b in machine.lattice.branches}
    assert machine.beams["ltr"].particles.shape == (19998, 6)
    assert check_machine(folder) == []
    powered = {o for c in machine.registry.circuits for m in c.magnets for o in m.occurrences}
    q1154 = machine.lattice.placement("prj-q1154", branch="prj").id
    assert q1154 in powered and np.isclose(
        machine.lattice.placement(q1154).element.MagneticMultipoleP.Bn1, -1.72
    )
