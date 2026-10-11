"""The EIN assembly against the EIN folder of the virtual accelerator project (not in this repository)."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cpymad.madx")

from eicvibe.domain.lattice import Lattice  # noqa: E402
from eicvibe.machines.ein import (  # noqa: E402
    RCSV5S1,
    TEMPORARY_AREAS,
    EinSources,
    build_ein,
    ein_registry,
    ein_repository,
    update_ein_machine,
    write_ein_machine,
)
from eicvibe.store import (  # noqa: E402
    build_machine,
    check_machine,
    load_lattice,
    load_machine,
    name_ids,
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


def test_the_ltr_keeps_its_names_its_bpms_and_its_injection_angle(ein):
    # Converted from Bmad with its writer's losses corrected (eicvibe.importers.bmad).
    ltr = ein.lattice.layout("ltr").placements
    kinds = {p.name: p.kind.value for p in ltr}
    assert kinds["ltr_qua01"] == "Quadrupole" and kinds["ltr_cor12"] == "Kicker"
    assert [p.name for p in ltr if p.kind.value == "Instrument"] == [f"ltr_bpm{i:02d}" for i in range(1, 16)]
    assert not [p.name for p in ltr if "_l10" in p.name]  # no names of Bmad's superposition slaves
    # A particle launched on the LTR's axis meets the RCS where Bmad's design has it: 30.9 mm inside,
    # heading out at 4.39 mrad, for the injection kicker to put on the axis.
    xsuite = pytest.importorskip("eicvibe.engines.xsuite")
    from eicvibe.engines.contracts import Options

    final = (
        xsuite.XsuiteEngine().open(ein.lattice, "ltr").track_beam(np.zeros((1, 6)), 1, (), Options()).final
    )
    assert final[0, :2] == pytest.approx(
        [-3.090030e-2, 4.391291e-3], abs=1e-8
    )  # Tao: -3.090030e-2, 4.391291e-3


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
    [magnet] = registry.circuit(setting.binding.circuit).magnets  # one magnet, one transfer function
    assert {lattice.placement(t).name for t in magnet.occurrences} == {"iseptum", "eseptum"}
    for name, branch in (("iseptum", "itl"), ("eseptum", "etl")):
        bend = lattice.placement(name, branch=branch).element
        assert bend.MagneticMultipoleP.Kn0 == pytest.approx(bend.BendP.angle_ref / bend.length)
        assert not bend.BendP.Kn0_from_g_ref  # so that 0 A means no field
    assert lattice.placement("septum", branch="bar").kind.value == "Kicker"


def test_every_ein_element_has_an_eic_name(ein):
    lattice = ein.lattice
    registry = ein_registry(ein)
    unnamed = {
        p.kind.value
        for b in lattice.branches
        for p in lattice.layout(b.id).placements
        if not registry.slot_of(p.id)
    }
    assert unnamed == {"BeginningEle", "Fork", "Patch"}  # bookkeeping only
    names = {
        p.name: registry.slot_of(p.id).name
        for p in lattice.layout("ltr").placements
        if registry.slot_of(p.id)
    }
    assert names["ltr_qua01"] == "ltr-q1:1" and names["ltr_qua02"] == "ltr-q1:2"  # one metre, two quads
    assert names["ltr_qua08"] == "ltr-q6"
    assert names["ltr_bpm01"] == "ltr-b1" and names["ltr_bpm05"] == "ltr-b18"
    assert registry.slot_of(lattice.placement("iseptum", branch="itl").id).name == "bar-sptm18"  # one magnet
    assert not [p for p in registry.naming_problems(ein_repository()) if "signal" not in p]


def test_temporary_names_can_change_without_changing_devices(ein):
    temporary = ein_registry(ein)
    renamed = ein_registry(ein, areas={"rcs": "rcb", "bar": "ba"})
    assert [d.id for d in renamed.devices] == [d.id for d in temporary.devices]
    assert {d.name.split("-")[0] for d in renamed.devices} == {"ltr", "itl", "ba", "etl", "rcb", "rte"}
    problems = renamed.naming_problems(ein_repository({"rcs": "rcb", "bar": "ba"}))
    assert problems and all("area" not in p for p in problems)  # only signal tokens awaiting approval remain


def test_a_new_machine_folder_holds_the_lattice_devices_and_routes(ein, tmp_path):
    folder = tmp_path / "machine"
    write_ein_machine(ein, folder)
    machine = load_machine(folder)
    assert machine.lattice == name_ids(ein.lattice)  # definitions and branches go by name in the files
    registry = ein_registry(ein)
    supplies = [(d.id, d.name) for d in registry.devices if d.type.value == "power_supply"]
    assert [(d.id, d.name) for d in machine.registry.devices if d.type.value == "power_supply"] == supplies
    assert [d.name for d in machine.registry.devices] == [d.name for d in registry.devices]
    assert [c.name for c in machine.registry.channels] == [c.name for c in registry.channels]
    ltr = (folder / "lattice" / "ltr.yaml").read_text()
    assert "{name: ltr_qua01, eic: ltr-q1:1, element: ltr.ltr_qua01, id: occ_" in ltr
    assert machine.source["madx"].startswith("MAD-X") and len(machine.source["files"]) == 7
    assert not (folder / "settings").exists() or not list((folder / "settings").iterdir())  # no BAR mode
    assert {r.name: len(r.stored) for r in machine.routes} == {"direct": 0, "bar": 1}
    assert check_machine(folder) == []
    paths = build_machine(folder)
    assert Lattice.from_json(paths["lattice"].read_text()) == name_ids(ein.lattice)
    assert json.loads(paths["manifest"].read_text())["lattice_digest"] == name_ids(ein.lattice).digest


def test_importing_again_keeps_identities_and_shows_what_changed(ein, tmp_path):
    folder = tmp_path / "machine"
    write_ein_machine(ein, folder)
    assert update_ein_machine(ein, folder) == ()  # the design files have not changed
    ltr = folder / "lattice" / "ltr.yaml"
    text = ltr.read_text()
    old = "ltr.ltr_qua01: {kind: Quadrupole, length: 0.3, MagneticMultipoleP: {Kn1: "
    assert old in text
    ltr.write_text(text.replace(old, old + "1"))  # someone edited a strength in the lattice files
    changes = update_ein_machine(ein, folder)
    assert [(c.entity, c.field) for c in changes] == [("element", "MagneticMultipoleP.Kn1")]
    power = (folder / "devices" / "power.yaml").read_text()
    update_ein_machine(ein, folder, write=True)  # the design files win
    assert load_machine(folder).lattice == name_ids(ein.lattice)
    assert "eic: ltr-q1:1" in ltr.read_text()  # the import writes the EIC names too
    assert (folder / "devices" / "power.yaml").read_text() == power  # devices are not touched


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


def test_ring_magnets_are_strung_by_family_and_line_magnets_powered_alone(ein):
    lattice = ein.lattice
    registry = ein_registry(ein)
    assert registry.power_problems(lattice) == []
    powered = {o for c in registry.circuits for m in c.magnets for o in m.occurrences}
    magnets = {"Bend", "Quadrupole", "Sextupole", "Kicker"}
    unpowered = {
        p.name
        for b in lattice.branches
        for p in lattice.layout(b.id).placements
        if p.kind.value in magnets and p.id not in powered
    }
    assert unpowered == {"septum"}  # bends included; only the septum's stored-beam traversal is never set
    dipoles = next(c for c in registry.circuits if c.name == "rcs.b0")
    assert len(dipoles.magnets) == 156 and registry.setting_of(dipoles.id).name == "rcs-dh111-ps:sp"
    # Strings join magnets of one length and design strength; in the BAR and RCS these are the lattice
    # families: the main, weak and "orig" dipoles differ in strength, so they are three strings.
    strings = sorted((c.name, len(c.magnets)) for c in registry.circuits if c.series)
    assert ("rcs.b0_weak", 4) in strings and ("rcs.b0_orig", 2) in strings and ("bar.bd", 8) in strings
    assert len(strings) == 20 and not any("+" in name for name, _ in strings)
    by_branch = {}
    for circuit in registry.circuits:
        branch = lattice.branch(lattice.placement(circuit.magnets[0].occurrences[0]).branch_id).name
        by_branch.setdefault(branch, set()).add(circuit.series)
    assert by_branch["ltr"] == by_branch["rte"] == {False} and by_branch["rcs"] == {False, True}
    alone = ein_registry(ein, series={})  # everything on its own supply
    assert not any(c.series for c in alone.circuits)


def test_once_matched_the_bar_is_transparent_and_both_routes_land_matched(ein, tmp_path, capsys):
    pytest.importorskip("scipy")
    pytest.importorskip("xtrack")
    from eicvibe.engines import engine, machine_optics, route_optics
    from eicvibe.machines.ein import main, match_ein

    folder = tmp_path / "machine"
    write_ein_machine(ein, folder)  # the design files' strengths
    files = {b: folder / "lattice" / f"{b}.yaml" for b in ("ltr", "itl", "etl")}
    before = {b: f.read_text().splitlines() for b, f in files.items()}
    assert main([str(FOLDER), "--machine", str(folder), "--match"]) == 0
    report = capsys.readouterr().out
    assert "NOT MATCHED" not in report and "left at BAR injection" in report
    for branch, path in files.items():
        changed = [b for b, a in zip(before[branch], path.read_text().splitlines(), strict=True) if a != b]
        assert changed and all("kind: Quadrupole" in line for line in changed)  # only quadrupoles
    assert check_machine(folder) == []
    lattice = load_lattice(folder)

    # Transparent: from P2 on the beam has the same optics through the BAR as past it, and it lands
    # with the RCS's own optics either way.
    xsuite = engine("xsuite")
    routes = {r.name: r for r in load_machine(folder).routes}
    direct, bar = (route_optics(xsuite, lattice, routes[n]) for n in ("direct", "bar"))
    merge = lattice.placement("ltr_qua11", branch="ltr").id
    for key, tolerance in (
        ("beta_x", 1e-2),
        ("alpha_x", 1e-2),
        ("beta_y", 1e-2),
        ("alpha_y", 1e-2),
        ("eta_x", 1e-4),
    ):
        assert getattr(bar.at(merge), key) == pytest.approx(
            getattr(direct.at(merge), key), rel=1e-2, abs=tolerance
        )
    periodic = machine_optics(xsuite, lattice)[lattice.layout("rcs").id]
    inside = {p.id for p in lattice.layout("rcs").placements}
    for optics in (direct, bar):
        for plane in ("beta_x", "beta_y"):
            peak = max(
                b for o, b in zip(optics.occurrences, getattr(optics, plane), strict=True) if o in inside
            )
            assert peak == pytest.approx(max(getattr(periodic, plane)), rel=1e-2)  # no beating round the RCS

    # The ITL matches beta and alpha into the BAR; the dispersion it cannot match is reported.
    again = match_ein(folder)
    assert again.success and all(m.values == m.start for m in again.steps.values())  # nothing to change
    assert again.steps["itl"].stretches["itl"].bmag_before == pytest.approx((1.0, 1.0), abs=1e-3)
    assert again.injection[2] > 0.01  # H: the ITL's bends need beta ~16 m at D1, the bypass gives more
