"""The EIN assembly against the EIN folder of the virtual accelerator project (not in this repository)."""

import json
import os
from pathlib import Path

import pytest

pytest.importorskip("cpymad.madx")

from eicvibe.domain.lattice import Lattice  # noqa: E402
from eicvibe.domain.machine import MachineRegistry  # noqa: E402
from eicvibe.machines.ein import (  # noqa: E402
    RCSV5S1,
    TEMPORARY_AREAS,
    EinSources,
    build_ein,
    ein_registry,
    ein_repository,
    save_ein,
)

FOLDER = Path(os.environ.get("EICVIBE_EIN_FOLDER", "/Users/haoyue/Research/EIC/virtual_accelerator/EIN"))
pytestmark = pytest.mark.skipif(not (FOLDER / "EICVIBE_lattice/madx").exists(), reason="EIN folder not found")


@pytest.fixture(scope="module")
def ein():
    return build_ein(FOLDER)


def connections(lattice):
    return {
        (lattice.branch(e.from_branch).name, lattice.branch(e.to_branch).name): (
            e.to_occurrence and lattice.placement(e.to_occurrence).name,
            e.to_side.value,
        )
        for e in lattice.forks
    }


def test_branches_and_connections(ein):
    lattice = ein.lattice
    kinds = {b.name: b.kind.value for b in lattice.branches}
    assert kinds == {
        "ltr": "linac",
        "itl": "linac",
        "bar": "periodic",
        "etl": "linac",
        "rcs": "periodic",
        "rte": "linac",
    }
    assert lattice.branch(lattice.root_branch).name == "ltr"
    assert connections(lattice) == {
        ("ltr", "itl"): (None, "entrance"),
        ("itl", "bar"): ("septum", "exit"),
        ("bar", "etl"): ("eseptum", "entrance"),
        ("etl", "ltr"): ("ltr_mark_p2", "exit"),
        ("ltr", "rcs"): ("mar.sep_2", "exit"),  # the staged RCS spells its injection point MAR.SEP
        ("rcs", "rte"): (None, "entrance"),
    }
    [septum] = lattice.shared_components
    roles = {t.role: lattice.placement(t.occurrence).name for t in septum.traversals}
    assert roles == {"injection": "iseptum", "circulation": "septum", "extraction": "eseptum"}
    assert lattice.diagnose() == ()


def test_the_40_cell_rcs_is_the_default_pinned_at_injection(ein):
    rcs = ein.lattice.layout("rcs")
    assert rcs.length == pytest.approx(1506.933944, abs=1e-6)
    assert rcs.placements[0].element.ReferenceP.pc_ref == pytest.approx(0.75e9, rel=1e-6)
    voltages = [p.element.RFP.voltage for p in rcs.placements if p.kind.value == "RFCavity"]
    assert sum(voltages) == pytest.approx(1.69758e6)
    extraction = next(e for e in ein.lattice.forks if ein.lattice.branch(e.to_branch).name == "rte")
    assert extraction.s_from == pytest.approx(843.2502, abs=1e-4)  # the marker after ES.PIP03


def test_the_rcs_file_energy_can_be_kept_and_rcsv5s1_can_replace_it():
    kept = build_ein(FOLDER, rcs_energy=None, rcs_rf_voltage=None).lattice.layout("rcs")
    assert kept.placements[0].element.ReferenceP.pc_ref == pytest.approx(9e9, rel=1e-8)
    lattice = build_ein(FOLDER, sources=EinSources(**RCSV5S1)).lattice
    injection = next(e for e in lattice.forks if lattice.branch(e.to_branch).name == "rcs")
    assert injection.s_to == pytest.approx(2.518)  # the marker after MQA_US, spelled MAR_SEP in this file
    assert lattice.layout("rcs").length == pytest.approx(1507.014835, abs=1e-6)


def test_duplicated_sections_are_cut(ein):
    ltr, rte = ein.lattice.layout("ltr"), ein.lattice.layout("rte")
    assert [p.name for p in ltr.placements[-2:]] == ["mar.inj", "fork_ltr_to_rcs"]
    assert ltr.length == pytest.approx(63.684163, abs=1e-6)
    assert "mar.sep" not in {p.name for p in rte.placements}
    assert rte.placements[1].name.startswith("el.pat__t")


def test_ltr_starts_with_the_preinjector_beam(ein):
    start = ein.lattice.layout("ltr").placements[0].element
    assert (start.TwissP.beta_a, start.TwissP.beta_b, start.TwissP.etap_x) == (9.06696, 20.3877, -0.011)
    assert start.FloorP.theta == pytest.approx(0.157101307179585)
    assert start.ReferenceP.species_ref == "electron"


def test_bar_mode_retunes_fifteen_ltr_quadrupoles(ein):
    lattice = ein.lattice
    targets = {lattice.placement(s.occurrence).name for s in ein.bar_mode}
    assert len(ein.bar_mode) == 15 and all(s.parameter == "MagneticMultipoleP.Kn1" for s in ein.bar_mode)
    assert {"ltr_pip01_l1001", "ltr_pip03_l1016"} <= targets
    assert all(lattice.placement(s.occurrence).branch_id == lattice.layout("ltr").id for s in ein.bar_mode)


def test_registry_covers_every_branch_and_the_septum_once(ein):
    lattice = ein.lattice
    registry = ein_registry(ein)
    assert registry.check_against(lattice) == []
    assert {d.name.split("-")[0] for d in registry.devices} == set(TEMPORARY_AREAS.values())
    traversals = {t.occurrence for t in lattice.shared_components[0].traversals}
    [septum] = [d for d in registry.devices if set(d.occurrences) & traversals]
    assert septum.name == "bar-sptm18-ps" and set(septum.occurrences) == traversals
    # One pulsed supply: the field of the injection and extraction bends, never the stored beam's kicker.
    setting = registry.channel("bar-sptm18-ps:sp")
    assert {lattice.placement(t).name for t in setting.binding.targets} == {"iseptum", "eseptum"}
    for name, branch in (("iseptum", "itl"), ("eseptum", "etl")):
        bend = lattice.placement(name, branch=branch).element
        assert bend.MagneticMultipoleP.Kn0 == pytest.approx(bend.BendP.angle_ref / bend.length)
        assert not bend.BendP.Kn0_from_g_ref  # so that 0 A means no field
    assert lattice.placement("septum", branch="bar").kind.value == "Kicker"


def test_temporary_names_can_change_without_changing_devices(ein):
    temporary = ein_registry(ein)
    renamed = ein_registry(ein, areas={"rcs": "rcb", "bar": "ba"})
    assert [d.id for d in renamed.devices] == [d.id for d in temporary.devices]
    assert {d.name.split("-")[0] for d in renamed.devices} == {"ltr", "itl", "ba", "etl", "rcb", "rte"}
    problems = renamed.naming_problems(ein_repository({"rcs": "rcb", "bar": "ba"}))
    assert problems and all("area" not in p for p in problems)  # only signal tokens awaiting approval remain


def test_save_writes_the_lattice_and_its_companions(ein, tmp_path):
    paths = save_ein(ein, tmp_path)
    assert Lattice.from_json(paths["lattice"].read_text()) == ein.lattice
    assert MachineRegistry.model_validate_json(paths["registry"].read_text()).check_against(ein.lattice) == []
    manifest = json.loads(paths["manifest"].read_text())
    assert manifest["lattice_digest"] == ein.lattice.digest and len(manifest["sources"]) == 9
    assert len(json.loads(paths["bar_mode"].read_text())["settings"]) == 15


def test_assembly_is_reproducible(ein):
    assert build_ein(FOLDER).lattice.digest == ein.lattice.digest


def test_the_floor_plan_closes_through_the_one_septum(ein):
    from eicvibe.geometry import floor_geometry

    lattice = ein.lattice
    geometry = floor_geometry(lattice)
    assert all(b.placed_by or b.branch == lattice.root_branch for b in geometry.branches.values())
    [closure] = geometry.closures  # the loop LTR -> ITL -> BAR -> ETL -> LTR
    assert closure.position_error < 1e-9 and closure.angle_error < 1e-9

    def frames(name, branch):
        placed = lattice.placement(name, branch=branch)
        return geometry.branches[placed.branch_id].frames[placed.index]

    # One rectangular magnet: a bend for the injected and extracted beams, a pulsed-off kicker
    # for the stored beam, which passes along its axis. The injected beam leaves it 1.75 mrad
    # (its e2) off the stored orbit, and the extracted beam enters 1.75 mrad off.
    injection, circulation, extraction = (
        frames("iseptum", "itl"),
        frames("septum", "bar"),
        frames("eseptum", "etl"),
    )
    assert circulation.exit.r == pytest.approx(injection.exit.r, abs=1e-12)
    assert circulation.exit.angles[0] - injection.exit.angles[0] == pytest.approx(1.75e-3)
    assert extraction.entrance.r == pytest.approx(circulation.entrance.r, abs=1e-12)
    assert extraction.entrance.angles[0] - circulation.entrance.angles[0] == pytest.approx(1.75e-3)
