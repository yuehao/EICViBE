"""Machine folders: lattice, devices and settings in files people read and edit."""

import pytest

from eicvibe.api.demo import demo_lattice
from eicvibe.domain.lattice import ElementKind, LatticeBuilder, SequentialIds
from eicvibe.domain.machine import TableTransfer, assemble_registry, describe_machine
from eicvibe.store import (
    FileError,
    Setting,
    build_machine,
    check_machine,
    load_lattice,
    load_machine,
    name_ids,
    read_settings,
    update_names,
    write_machine,
    write_settings,
)
from eicvibe.store.cli import main as cli
from eicvibe.store.device_files import add_bpms, rename_references


def described(lattice):
    return describe_machine(
        lattice,
        {"ring": None, "injection": None, "extraction": None},
        series={"ring": (ElementKind.BEND, ElementKind.QUADRUPOLE)},
    )


@pytest.fixture
def folder(tmp_path):
    lattice = demo_lattice()
    write_machine(tmp_path / "demo", lattice, described(lattice), source={"importer": "test"})
    return tmp_path / "demo"


def test_a_machine_folder_reads_back_exactly(folder):
    lattice = name_ids(demo_lattice())  # definitions, branches and components go by name in the files
    machine = load_machine(folder)
    assert machine.lattice == lattice and machine.lattice.digest == lattice.digest
    assert machine.description == described(lattice)
    assert machine.registry == assemble_registry(lattice, described(lattice))
    assert machine.source == {"importer": "test"}
    names = sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*.yaml"))
    assert names == [
        "devices/diagnostics.yaml",
        "devices/power.yaml",
        "devices/rf.yaml",
        "lattice/extraction.yaml",
        "lattice/injection.yaml",
        "lattice/ring.yaml",
        "lattice/shared.yaml",
        "machine.yaml",
        "naming.yaml",
    ]
    assert check_machine(folder) == []
    ring = (folder / "lattice" / "ring.yaml").read_text()
    assert "id: ele_" not in ring and "id: br_" not in ring  # only occurrences keep ids
    assert "  - {name: QF_1, eic: ring-q4, element: QF, id: occ_4}" in ring  # its EIC name beside it
    drifts = [line for line in ring.splitlines() if "element: D}" in line or "element: DL}" in line]
    assert drifts and all("id:" not in line and "eic: ring-dt" in line for line in drifts)  # by name


def test_eic_names_follow_the_naming_rules_and_build_keeps_them_current(folder):
    ring = folder / "lattice" / "ring.yaml"
    ring.write_text(ring.read_text().replace("elements:", "elements:  # checked by Ann", 1))
    naming = folder / "naming.yaml"
    naming.write_text(naming.read_text().replace("  ring: ring\n", "  ring: dr\n", 1))  # SEG's prefix arrives
    problems = check_machine(folder)
    assert len(problems) == 1 and "EIC names in the lattice files differ" in problems[0]
    changes = update_names(folder, write=False)
    assert "eic ring/QF_1: ring-q4 -> dr-q4" in changes
    build_machine(folder)
    text = ring.read_text()
    assert "{name: QF_1, eic: dr-q4, element: QF, id: occ_4}" in text and "# checked by Ann" in text
    assert check_machine(folder) == []
    assert load_lattice(folder).placement("QF_1", branch="ring").id == "occ_4"  # the same magnet


def test_ids_other_than_occurrences_are_dropped_by_build(folder):
    ring = folder / "lattice" / "ring.yaml"
    old = ring.read_text()
    ring.write_text(
        old.replace("kind: periodic\n", "kind: periodic\nid: br_old\n", 1).replace(
            "DL: {kind: Drift, length: 2.0}", "DL: {kind: Drift, length: 2.0, id: ele_old}", 1
        )
    )
    assert load_lattice(folder).branch("ring").id == "br_old"
    assert update_names(folder) == ["ids lattice/ring.yaml: removed 2 (definitions and branches go by name)"]
    assert ring.read_text() == old
    assert load_lattice(folder) == name_ids(demo_lattice())


def test_overrides_tags_forks_and_shared_definitions_round_trip(tmp_path):
    b = LatticeBuilder("odd", ids=SequentialIds())
    b.define("BEG", "BeginningEle", ReferenceP={"species_ref": "electron", "pc_ref": 1e9})
    b.define("Q", "Quadrupole", length=0.5, tags=["family"], MagneticMultipoleP={"Kn1": 0.5})
    b.define("D", "Drift", length=1.0)
    b.define("UNUSED", "Marker")
    b.add_branch("a", "linac", ["BEG", "Q", "D", "Q"])
    b.add_branch("b", "linac", ["D", "Q"])  # D and Q are a's definitions
    b.set_overrides(("a", "Q_2"), MagneticMultipoleP={"Kn1": 0.55})
    b.update_occurrence(("a", "D_1"), tags=["girder-3"])
    b.add_fork("a", "b", after="D_1", to="Q_1", to_side="exit", propagate_reference=True)
    lattice = b.build()
    write_machine(tmp_path, lattice)
    assert load_lattice(tmp_path) == name_ids(lattice)
    text = (tmp_path / "lattice" / "a.yaml").read_text()
    assert "overrides: {MagneticMultipoleP: {Kn1: 0.55}}" in text
    assert "fork: {to: b, at: Q_1, side: exit, propagate_reference: true}" in text
    assert "UNUSED" in text and "  Q:" not in (tmp_path / "lattice" / "b.yaml").read_text()


def test_a_new_entry_gets_a_derived_id_and_errors_name_the_file(folder):
    ring = folder / "lattice" / "ring.yaml"
    text = ring.read_text()
    first = next(line for line in text.splitlines() if line.startswith("  - {name: SEPTUM_1"))
    ring.write_text(text.replace(first, first.split(", id:")[0] + "}"))  # the occurrence loses its id
    lattice = load_lattice(folder)
    assert lattice.placement("SEPTUM_1", branch="ring").id.startswith("occ_")
    assert load_lattice(folder) == lattice  # the derived id is the same every time
    ring.write_text(text.replace("element: QF,", "element: QFX,", 1))
    with pytest.raises(FileError, match=r"ring\.yaml: line item \d+ \(QF_1\): no element 'QFX'"):
        load_lattice(folder)


def test_a_measured_curve_replaces_a_placeholder_and_a_string_mismatch_is_reported(folder):
    power = folder / "devices" / "power.yaml"
    curve = folder / "devices" / "transfer" / "qf-017.csv"
    curve.parent.mkdir(parents=True, exist_ok=True)
    curve.write_text("# QF-017, measured 2026-09-30\ncurrent [A],gradient [T/m]\n0,0\n100,10.5\n200,20.6\n")
    text = power.read_text()
    text = text.replace(
        "transfer:\n",
        "transfer:\n  qf-017: {kind: table, file: transfer/qf-017.csv, field_unit: T/m, source: measured}\n",
        1,
    )
    lines = text.splitlines()
    string = lines.index(next(line for line in lines if line.startswith("  QF:")))
    first = next(i for i in range(string, len(lines)) if lines[i].startswith("      - ring/QF_"))
    name = lines[first].split("- ")[1]
    lines[first] = f"      - {{magnet: {name}, transfer: qf-017}}"
    power.write_text("\n".join(lines) + "\n")
    machine = load_machine(folder)
    circuit = next(c for c in machine.registry.circuits if c.name == "QF")
    measured = next(m for m in circuit.magnets if isinstance(m.transfer, TableTransfer))
    assert measured.transfer.currents == (0.0, 100.0, 200.0) and measured.transfer.source.value == "measured"
    assert any("QF: its magnets need" in p for p in check_machine(folder))  # 5 % more field than its string


def test_a_service_transfer_function_with_a_fallback(folder):
    power = folder / "devices" / "power.yaml"
    text = power.read_text().replace(
        "transfer:\n",
        "transfer:\n  b-svc: {kind: service, service: magnet-tf, magnet: B-001, hysteresis: true,"
        " fallback: dipole-placeholder, field_unit: T}\n",
        1,
    )
    lines = text.splitlines()
    string = lines.index(next(line for line in lines if line.startswith("  B:")))
    transfer = next(i for i in range(string, len(lines)) if lines[i].startswith("    transfer:"))
    lines[transfer] = "    transfer: b-svc"
    power.write_text("\n".join(lines) + "\n")
    circuit = next(c for c in load_machine(folder).registry.circuits if c.name == "B")
    assert {m.transfer.kind for m in circuit.magnets} == {"service"}
    assert circuit.magnets[0].transfer.fallback.coefficients == (0.0, 0.001)


def test_renaming_an_area_renames_everything_and_keeps_identities(folder):
    before = load_machine(folder).registry
    naming = folder / "naming.yaml"
    assert "  ring: ring\n" in naming.read_text()
    naming.write_text(naming.read_text().replace("  ring: ring\n", "  ring: rcb\n"))
    after = load_machine(folder).registry
    assert sorted(d.id for d in after.devices) == sorted(d.id for d in before.devices)
    assert any(d.name.startswith("rcb-") for d in after.devices)
    assert not any(d.name.startswith("ring-") for d in after.devices)


def test_settings_files(folder):
    lattice = load_lattice(folder)
    quad = lattice.placement("QF_1", branch="ring")
    write_settings(
        folder, "low-tune", [Setting(quad.id, "MagneticMultipoleP.Kn1", 0.79)], lattice, "A test mode."
    )
    assert "ring/QF_1: {MagneticMultipoleP.Kn1: 0.79}" in (folder / "settings" / "low-tune.yaml").read_text()
    assert read_settings(folder, "low-tune", lattice) == [Setting(quad.id, "MagneticMultipoleP.Kn1", 0.79)]


def test_the_command_line_checks_and_builds(folder, capsys):
    assert cli(["check", str(folder)]) == 0
    assert "demo: 3 branches" in capsys.readouterr().out
    assert cli(["build", str(folder)]) == 0
    assert {p.name for p in (folder / "build").iterdir()} == {
        "lattice.json",
        "registry.json",
        "manifest.json",
    }
    (folder / "devices" / "rf.yaml").write_text("cavities:\n  ring/NOPE: {voltage: [0, 1]}\n")
    assert cli(["check", str(folder)]) == 1
    assert "rf.yaml: cavity ring/NOPE: no occurrence 'ring/NOPE'" in capsys.readouterr().err


def test_routes_are_read_by_branch_names_and_never_rewritten(folder):
    from eicvibe.store.machine import ROUTES_FILE, write_routes

    assert load_machine(folder).routes == ()
    assert write_routes(folder, {"transfer": ["injection", "ring", "extraction"]}) is not None
    (route,) = load_machine(folder).routes
    assert route.name == "transfer" and route.branches == ("injection", "ring", "extraction")
    edited = (folder / ROUTES_FILE).read_text().replace("transfer:", "out:")
    (folder / ROUTES_FILE).write_text(edited)
    assert write_routes(folder, {"transfer": ["injection", "ring"]}) is None  # people's file stays
    assert [r.name for r in load_machine(folder).routes] == ["out"]
    (folder / ROUTES_FILE).write_text("routes:\n  back: [extraction, ring]\n")
    with pytest.raises(FileError, match="routes.yaml.*back: no fork from extraction"):
        load_machine(folder)


def test_device_references_follow_an_occurrence_renamed_in_place(folder):
    before = load_machine(folder).registry
    ring = folder / "lattice" / "ring.yaml"
    ring.write_text(ring.read_text().replace("{name: COR_1,", "{name: HCOR_1,"))
    power = folder / "devices" / "power.yaml"
    text = power.read_text()
    assert rename_references(folder, {"ring/COR_1": "ring/HCOR_1"}) == [power]
    assert power.read_text() == text.replace("ring/COR_1:", "ring/HCOR_1:").replace(
        "[ring/COR_1]", "[ring/HCOR_1]"
    )  # COR_10 and COR_11 are other magnets; comments and everything else stay
    assert check_machine(folder) == []
    after = load_machine(folder).registry  # the supply keyed by the magnet takes its new key, and a new id
    assert [d.name for d in after.devices] == [d.name for d in before.devices]
    assert [c.name for c in after.channels] == [c.name for c in before.channels]


def test_bpms_are_added_after_their_branchs(folder):
    path = folder / "devices" / "diagnostics.yaml"
    original = path.read_text()
    path.write_text(original.replace("  - ring/BPM_3\n", "").replace("  - injection/BPM_1\n", ""))
    branches = ["ring", "injection", "extraction"]
    assert add_bpms(folder, ["injection/BPM_1", "ring/BPM_3"], branches) == path
    lines = path.read_text().splitlines()
    bpms = [line for line in lines if line.startswith("  - ")]
    assert bpms[-4:] == ["  - ring/BPM_11", "  - ring/BPM_3", "  - injection/BPM_1", "  - extraction/BPM_1"]
    assert lines[0] == original.splitlines()[0]  # the header comment stays
    assert add_bpms(folder, ["ring/BPM_3"], branches) is None  # listed already
    assert check_machine(folder) == []
