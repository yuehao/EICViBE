import copy
import json
import pickle

import pytest
from pydantic import ValidationError

from eicvibe.domain.lattice import ElementKind, Lattice, LatticeBuilder, SequentialIds


def document(lattice: Lattice) -> dict:
    return copy.deepcopy(lattice.to_dict())


def rejects(data: dict, code: str) -> None:
    with pytest.raises(ValidationError, match=rf"\[{code}\]"):
        Lattice.from_dict(data)


# -- structure ------------------------------------------------------------


def test_duplicate_ids_are_rejected(fodo):
    data = document(fodo)
    data["branches"][0]["occurrences"][1]["id"] = data["branches"][0]["occurrences"][0]["id"]
    rejects(data, "duplicate-id")


def test_occurrence_names_are_unique_within_a_branch(fodo):
    data = document(fodo)
    data["branches"][0]["occurrences"][1]["name"] = "QF_1"
    rejects(data, "duplicate-name")


def test_unknown_element_reference(fodo):
    data = document(fodo)
    data["branches"][0]["occurrences"][0]["element"] = "ele_missing"
    rejects(data, "unknown-element")


def test_overrides_must_apply_to_the_element(fodo):
    data = document(fodo)
    data["branches"][0]["occurrences"][1]["overrides"] = [
        {"group": "MagneticMultipoleP", "parameter": "Kn1", "value": 1.0}  # on a drift
    ]
    rejects(data, "invalid-override")


def test_override_values_are_type_checked_on_their_own(fodo):
    data = document(fodo)
    data["branches"][0]["occurrences"][0]["overrides"] = [
        {"group": "ApertureP", "parameter": "thickness", "value": -1.0}
    ]
    with pytest.raises(ValidationError, match="thickness"):
        Lattice.from_dict(data)


def test_errors_are_reported_together(fodo):
    data = document(fodo)
    data["root_branch"] = "br_missing"
    data["branches"][0]["occurrences"][0]["element"] = "ele_missing"
    with pytest.raises(ValidationError) as excinfo:
        Lattice.from_dict(data)
    assert "[unknown-branch]" in str(excinfo.value) and "[unknown-element]" in str(excinfo.value)


def test_beginning_element_must_lead_its_branch():
    b = LatticeBuilder("x", ids=SequentialIds())
    b.define("BEG", "BeginningEle")
    b.define("D", "Drift", length=1.0)
    b.add_branch("line", "linac", ["D", "BEG"])
    with pytest.raises(ValidationError, match=r"\[beginning-position\]"):
        b.build()


def test_fork_links_must_resolve(machine):
    data = document(machine)
    fork = next(
        o
        for b in data["branches"]
        for o in b["occurrences"]
        if "fork" in o and o["fork"].get("to_occurrence")
    )
    fork["fork"]["to_occurrence"] = data["branches"][0]["occurrences"][0]["id"]  # wrong branch
    rejects(data, "fork-target")
    del fork["fork"]
    rejects(data, "fork-link")


def test_shared_component_rules(machine):
    data = document(machine)
    component = data["shared_components"][0]
    marker_id = next(
        o["id"]
        for b in data["branches"]
        if b["name"] == "ring"
        for o in b["occurrences"]
        if o["name"] == "EXT_1"
    )
    component["traversals"][0]["occurrence"] = marker_id
    rejects(data, "component-traversal")
    with pytest.raises(ValidationError):
        component["traversals"] = component["traversals"][:1]
        Lattice.from_dict(data)


def test_shared_traversals_must_have_compatible_lengths(machine):
    b = LatticeBuilder.from_lattice(machine)
    b.update_element("SEPT_INJ", length=1.5)
    with pytest.raises(ValidationError, match=r"\[component-length\]"):
        b.build()


# -- immutability, equality, serialization -------------------------------------


def test_lattice_is_immutable(fodo):
    with pytest.raises(ValidationError):
        fodo.name = "other"
    with pytest.raises(TypeError, match="LatticeBuilder"):
        fodo.model_copy(update={"name": "other"})


def test_equal_content_is_equal_regardless_of_declaration_order(machine):
    data = document(machine)
    for key in ("elements", "branches"):
        data[key].reverse()
    data["elements"][0]["tags"] = ["b", "a", "a"]
    reordered = Lattice.from_dict(data)
    data["elements"][0]["tags"] = ["a", "b"]
    assert reordered == Lattice.from_dict(data)
    data["elements"][0].pop("tags")
    assert Lattice.from_dict(data) == machine and Lattice.from_dict(data).digest == machine.digest
    assert [e.name for e in machine.elements] == sorted(e.name for e in machine.elements)


def test_canonical_form_omits_defaults(fodo):
    data = fodo.to_dict()
    drift = next(e for e in data["elements"] if e["name"] == "D")
    assert drift == {"id": "ele_3", "name": "D", "kind": "Drift", "length": 2.0}
    assert "root_branch" in data and "shared_components" not in data
    occurrence = data["branches"][0]["occurrences"][0]
    assert set(occurrence) == {"id", "name", "element"}


def test_golden_digest_pins_the_canonical_format(fodo):
    # Changing this value means every stored revision hash changes: bump
    # FORMAT_VERSION and write a migration instead of editing the constant.
    assert fodo.to_json() == (
        '{"branches":[{"id":"br_1","kind":"periodic","name":"cell","occurrences":['
        '{"element":"ele_1","id":"occ_1","name":"QF_1"},{"element":"ele_3","id":"occ_2","name":"D_1"},'
        '{"element":"ele_2","id":"occ_3","name":"QD_1"},{"element":"ele_3","id":"occ_4","name":"D_2"}]}],'
        '"elements":[{"id":"ele_3","kind":"Drift","length":2.0,"name":"D"},'
        '{"MagneticMultipoleP":{"Kn1":-1.2},"id":"ele_2","kind":"Quadrupole","length":0.5,"name":"QD","tags":["QUAD"]},'
        '{"MagneticMultipoleP":{"Kn1":1.2},"id":"ele_1","kind":"Quadrupole","length":0.5,"name":"QF","tags":["QUAD"]}],'
        '"format_version":1,"name":"fodo","root_branch":"br_1"}'
    )
    assert fodo.digest == "sha256:" + __import__("hashlib").sha256(fodo.to_json().encode()).hexdigest()


def test_round_trips(machine):
    assert Lattice.from_json(machine.to_json()) == machine
    assert Lattice.from_dict(json.loads(machine.to_json(indent=2))) == machine
    restored = pickle.loads(pickle.dumps(machine))
    assert restored == machine and restored.digest == machine.digest


def test_digest_tracks_content(fodo):
    b = LatticeBuilder.from_lattice(fodo)
    b.update_element("QF", MagneticMultipoleP={"Kn1": 1.21})
    assert b.build().digest != fodo.digest
    b.update_element("QF", MagneticMultipoleP={"Kn1": 1.2})
    assert b.build().digest == fodo.digest


def test_json_schema_is_available_for_clients():
    schema = Lattice.model_json_schema()
    assert schema["properties"]["format_version"]["const"] == 1
    assert schema["$defs"]["MagneticMultipoleP"]["properties"]["Kn1"]["unit"] == "1/m^2"


# -- lookups and derived data ---------------------------------------------------


def test_positions(fodo):
    layout = fodo.layout("cell")
    assert [(p.name, p.s_enter, p.s_exit) for p in layout.placements] == [
        ("QF_1", 0.0, 0.5),
        ("D_1", 0.5, 2.5),
        ("QD_1", 2.5, 3.0),
        ("D_2", 3.0, 5.0),
    ]
    assert layout.length == 5.0 and layout.periodic


def test_lookup_by_id_or_name(fodo):
    assert fodo.element("QF") is fodo.element("ele_1")
    assert fodo.branch("cell").id == "br_1"
    assert fodo.placement("D_2", branch="cell").id == "occ_4"
    assert fodo.placement("occ_4").name == "D_2"
    with pytest.raises(KeyError):
        fodo.placement("D_2")  # names need a branch
    with pytest.raises(KeyError):
        fodo.element("QX")


def test_a_name_equal_to_another_id_is_ambiguous():
    b = LatticeBuilder("x", ids=SequentialIds())
    b.define("ele_2", "Marker")  # id ele_1, named like the next id
    b.define("M", "Marker")  # id ele_2
    lattice = b.build()
    with pytest.raises(KeyError, match="both"):
        lattice.element("ele_2")


def test_select(fodo):
    assert [p.name for p in fodo.select("cell", kind="Quadrupole")] == ["QF_1", "QD_1"]
    assert [p.name for p in fodo.select("cell", element="D")] == ["D_1", "D_2"]
    assert [p.name for p in fodo.select("cell", tag="QUAD", name="^QF")] == ["QF_1"]
    assert [p.name for p in fodo.select("cell", s_range=(0.6, 2.5))] == ["D_1", "QD_1"]
    with pytest.raises(ValueError):
        fodo.select("cell", s_range=(3.0, 1.0))


def test_forks_and_components(machine):
    injection, extraction = sorted(
        machine.forks, key=lambda f: f.from_branch != machine.branch("injection").id
    )
    septum = machine.shared_component("septum")
    ring_side = machine.placement(septum.traversal("circulation").occurrence)
    assert injection.to_occurrence == ring_side.id and injection.s_to == ring_side.s_exit
    assert machine.component_of(ring_side.id) is septum
    assert extraction.to_occurrence is None and extraction.s_to == 0.0
    assert machine.forks_into("ring") == (injection,)
    assert machine.forks_from("ring") == (extraction,)


def test_initial_conditions(machine):
    conditions = machine.initial_conditions("injection")
    assert conditions.reference.species_ref == "electron" and conditions.twiss.beta_a == 10.0
    assert conditions.particle is None
    assert machine.initial_conditions("ring") is None


def test_effective_element_includes_overrides(fodo):
    b = LatticeBuilder.from_lattice(fodo)
    b.set_overrides(("cell", "QF_1"), BodyShiftP={"x_offset": 1e-4}, MagneticMultipoleP={"Kn1": 1.25})
    lattice = b.build()
    placed = lattice.placement("QF_1", branch="cell")
    assert placed.element.BodyShiftP.x_offset == 1e-4
    assert placed.element.MagneticMultipoleP.Kn1 == 1.25
    assert placed.definition.MagneticMultipoleP.Kn1 == 1.2
    assert placed.kind is ElementKind.QUADRUPOLE
