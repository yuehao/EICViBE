import pytest

from eicvibe.domain.lattice import ForkSide, Lattice, LatticeBuilder, SequentialIds

from .conftest import complex_builder, fodo_builder


def names(lattice: Lattice, branch: str) -> list[str]:
    return [p.name for p in lattice.layout(branch).placements]


def test_placements_share_the_definition(fodo):
    placed = fodo.select("cell", element="D")
    assert {p.definition.id for p in placed} == {fodo.element("D").id}
    b = LatticeBuilder.from_lattice(fodo)
    b.update_element("D", length=2.5)
    assert b.build().layout("cell").length == 6.0  # both drifts changed


def test_default_occurrence_names_fill_the_smallest_free_suffix():
    b = fodo_builder()
    b.remove(("cell", "D_1"))
    assert b.append("cell", "D") and names(b.build(), "cell")[-1] == "D_1"
    assert b.append("cell", "D", name="D_extra")
    with pytest.raises(ValueError, match="already in use"):
        b.append("cell", "D", name="D_extra")


def test_insert_before_and_after():
    b = fodo_builder()
    b.define("BPM", "Instrument")
    b.insert("cell", "BPM", after="QF_1")
    b.insert("cell", "BPM", before="QF_1", name="BPM_up")
    assert names(b.build(), "cell") == ["BPM_up", "QF_1", "BPM_1", "D_1", "QD_1", "D_2"]
    with pytest.raises(ValueError):
        b.insert("cell", "BPM", before="QF_1", after="QD_1")


def test_insertion_keeps_every_existing_identity(fodo):
    b = LatticeBuilder.from_lattice(fodo)
    b.define("M", "Marker")
    b.insert("cell", "M", before="QF_1")
    edited = b.build()
    for old in fodo.layout("cell").placements:
        new = edited.placement(old.id)
        assert new.name == old.name and new.index == old.index + 1


def test_overrides_are_validated_when_set():
    b = fodo_builder()
    with pytest.raises(ValueError, match="overrides do not apply"):
        b.append("cell", "D", MagneticMultipoleP={"Kn1": 1.0})
    b.set_overrides(("cell", "QF_1"), BodyShiftP={"x_offset": 1e-4, "y_offset": 2e-4})
    b.set_overrides(("cell", "QF_1"), BodyShiftP={"x_offset": 3e-4})
    b.clear_overrides(("cell", "QF_1"), "BodyShiftP.y_offset")
    shift = b.build().placement("QF_1", branch="cell").element.BodyShiftP
    assert (shift.x_offset, shift.y_offset) == (3e-4, 0.0)
    b.clear_overrides(("cell", "QF_1"))
    assert b.build().placement("QF_1", branch="cell").occurrence.overrides == ()


def test_definition_changes_are_checked_against_existing_overrides():
    b = LatticeBuilder("x", ids=SequentialIds())
    b.define("K", "Kicker")
    b.add_branch("line", "linac", [])
    b.append("line", "K", MagneticMultipoleP={"Ks0L": 1e-4})
    with pytest.raises(ValueError, match="Kn0/Ks0 and Kn0L/Ks0L"):
        b.update_element("K", MagneticMultipoleP={"Kn0": 1e-3})


def test_removal_refuses_dangling_references():
    b = complex_builder()
    with pytest.raises(ValueError, match="still placed"):
        b.remove_element("D")
    with pytest.raises(ValueError, match="land on"):
        b.remove(("ring", "SEPT_RING_1"))
    with pytest.raises(ValueError, match="traversal of shared component"):
        b.remove(("injection", "SEPT_INJ_1"))
    with pytest.raises(ValueError, match="target of forks"):
        b.remove_branch("extraction")
    with pytest.raises(ValueError, match="root branch"):
        b.remove_branch("ring")
    b.remove(("ring", "fork_ring_to_extraction"))
    b.remove_branch("extraction")
    b.build()


def test_forks_cannot_lose_their_landing_point():
    b = LatticeBuilder("x", ids=SequentialIds())
    b.define("D", "Drift", length=1.0)
    b.add_branch("a", "linac", ["D"])
    b.add_branch("b", "linac", ["D", "D"])
    b.add_fork("a", "b", to="D_2")
    with pytest.raises(ValueError, match="land on"):
        b.remove(("b", "D_2"))


def test_injection_through_a_shared_septum_passes_it_once(machine):
    fork = machine.forks_into("ring")[0]
    septum = machine.shared_component("septum")
    assert {t.role for t in septum.traversals} == {"injection", "circulation"}
    assert fork.to_side is ForkSide.EXIT
    assert fork.to_occurrence == septum.traversal("circulation").occurrence
    line = machine.layout("injection").placements
    assert line[fork.from_index - 1].id == septum.traversal("injection").occurrence


def test_extraction_through_a_shared_septum():
    b = complex_builder()
    b.define("SEPT_EXT", "Bend", length=1.0, BendP={"angle_ref": -0.05})
    b.insert("extraction", "SEPT_EXT", before="D_1")
    fork = b.connect_extraction("ring", "extraction", via=("SEPT_RING_1", "SEPT_EXT_1"), component="septum")
    lattice = b.build()
    septum = lattice.shared_component("septum")
    assert {t.role for t in septum.traversals} == {"injection", "circulation", "extraction"}
    edge = lattice.topology.forks[fork]
    ring = lattice.layout("ring").placements
    assert ring[edge.from_index + 1].id == septum.traversal("circulation").occurrence
    assert edge.to_occurrence == septum.traversal("extraction").occurrence
    assert edge.to_side is ForkSide.ENTRANCE


def test_connect_requires_exactly_one_anchor():
    b = complex_builder()
    with pytest.raises(ValueError, match="exactly one"):
        b.connect_extraction("ring", "extraction")


def test_shared_roles_cannot_be_rebound():
    b = complex_builder()
    with pytest.raises(ValueError, match="already binds"):
        b.share("septum", {"circulation": ("ring", "Q_1")})
    with pytest.raises(ValueError, match="already a traversal"):
        b.share("other", {"x": ("ring", "SEPT_RING_1"), "y": ("ring", "Q_2")})


def test_fork_element_name_clash():
    b = LatticeBuilder("x", ids=SequentialIds())
    b.define("fork", "Marker")
    b.define("D", "Drift", length=1.0)
    b.add_branch("a", "linac", ["D"])
    b.add_branch("b", "linac", ["D"])
    with pytest.raises(ValueError, match="not a Fork"):
        b.add_fork("a", "b")


def test_include_keeps_families_separate():
    ring = fodo_builder().build()
    line = LatticeBuilder("line", ids=SequentialIds())
    line.define("D", "Drift", length=3.0)
    line.add_branch("tl", "linac", ["D"])
    other = line.build()

    b = LatticeBuilder.from_lattice(ring)
    with pytest.raises(ValueError, match="name clash"):
        b.include(other)
    with pytest.raises(ValueError, match="id clash"):
        b.include(other, prefix="tl_")
    mapping = b.include(other, prefix="tl_", new_ids=True)
    combined = b.build()
    assert combined.element("tl_D").length == 3.0 and combined.element("D").length == 2.0
    assert combined.root_branch == ring.root_branch
    assert set(mapping) == {"ele_1", "br_1", "occ_1"}


def test_from_lattice_round_trip_and_ids_are_not_reused(machine):
    b = LatticeBuilder.from_lattice(machine, ids=SequentialIds())
    assert b.build() == machine
    b.remove(("ring", "fork_ring_to_extraction"))
    new_id = b.add_fork("ring", "extraction", after="EXT_1")
    assert new_id not in {o.id for br in machine.branches for o in br.occurrences}


def test_ids_are_never_reused_within_a_builder():
    b = fodo_builder()
    removed = b.build().placement("D_2", branch="cell").id
    b.remove(removed)
    added = b.append("cell", "D")
    assert added != removed


def test_renames():
    b = fodo_builder()
    b.update_branch("cell", name="arc", kind="linac")
    b.update_element("QF", name="QF_family")
    b.update_occurrence(("arc", "QF_1"), name="QF_A", tags=["sector1"])
    lattice = b.build()
    assert lattice.branch("arc").kind == "linac"
    placed = lattice.placement("QF_A", branch="arc")
    assert placed.definition.name == "QF_family" and placed.tags == {"QUAD", "sector1"}


def test_large_lattice_builds():
    b = LatticeBuilder("big")
    b.define("Q", "Quadrupole", length=0.5, MagneticMultipoleP={"Kn1": 0.3})
    b.define("D", "Drift", length=1.0)
    b.add_branch("ring", "periodic", ["Q", "D"] * 5000)
    lattice = b.build()
    assert len(lattice.branch("ring").occurrences) == 10_000
    assert lattice.layout("ring").length == pytest.approx(7500.0)
