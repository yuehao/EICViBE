from eicvibe.domain.lattice import ChangeKind, LatticeBuilder, diff


def test_identical_revisions_have_no_changes(machine):
    assert diff(machine, machine) == ()


def test_insertion_and_rename_do_not_misattach(fodo):
    """Roadmap acceptance: renaming or inserting keeps identities distinct from positions."""
    b = LatticeBuilder.from_lattice(fodo)
    b.define("BPM", "Instrument")
    inserted = b.insert("cell", "BPM", before="QF_1")
    b.update_occurrence(("cell", "QD_1"), name="QD_A")
    changes = diff(fodo, b.build())
    assert [(c.kind, c.entity, c.field) for c in changes] == [
        (ChangeKind.ADDED, "element", ""),
        (ChangeKind.ADDED, "occurrence", ""),
        (ChangeKind.MODIFIED, "occurrence", "name"),
    ]
    assert changes[1].id == inserted and changes[1].new == "cell/BPM_1"
    assert (changes[2].old, changes[2].new) == ("QD_1", "QD_A")


def test_parameter_and_override_changes(fodo):
    b = LatticeBuilder.from_lattice(fodo)
    b.update_element("QF", MagneticMultipoleP={"Kn1": 1.25}, ApertureP={})
    b.set_overrides(("cell", "QD_1"), BodyShiftP={"x_offset": 1e-4})
    changes = {(c.entity, c.field): (c.old, c.new) for c in diff(fodo, b.build())}
    assert changes == {
        ("element", "MagneticMultipoleP.Kn1"): (1.2, 1.25),
        ("occurrence", "overrides.BodyShiftP.x_offset"): (None, 1e-4),
    }


def test_tracking_settings_are_compared_per_setting(fodo):
    b = LatticeBuilder.from_lattice(fodo)
    b.update_element("QF", TrackingP={"xsuite": {"num_steps": 4}})
    tracked = b.build()
    b.update_element("QF", TrackingP={"xsuite": {"num_steps": 8}})
    [change] = diff(tracked, b.build())
    assert (change.field, change.old, change.new) == ("TrackingP.xsuite.num_steps", 4, 8)
    [added] = diff(fodo, tracked)
    assert (added.field, added.old, added.new) == ("TrackingP.xsuite.num_steps", None, 4)


def test_group_removal_compares_against_defaults(fodo):
    b = LatticeBuilder.from_lattice(fodo)
    b.update_element("QF", MagneticMultipoleP=None)
    [change] = diff(fodo, b.build())
    assert (change.field, change.old, change.new) == ("MagneticMultipoleP.Kn1", 1.2, 0.0)


def test_reordering_is_reported_once(fodo):
    b = LatticeBuilder.from_lattice(fodo)
    qf = b.build().placement("QF_1", branch="cell")
    b.remove(qf.id)
    b.append("cell", "QF", name="QF_1")
    changes = diff(fodo, b.build())
    kinds = sorted((c.entity, c.kind.value) for c in changes)
    assert kinds == [("occurrence", "added"), ("occurrence", "removed")]

    moved = LatticeBuilder.from_lattice(fodo)
    moved.update_element("D", length=2.0)
    assert diff(fodo, moved.build()) == ()


def test_fork_and_component_changes(machine):
    b = LatticeBuilder.from_lattice(machine)
    b.remove(("ring", "fork_ring_to_extraction"))
    b.add_fork("ring", "extraction", after="Q_2", name="fork_ring_to_extraction")
    changes = diff(machine, b.build())
    assert {(c.entity, c.kind.value) for c in changes} == {("occurrence", "added"), ("occurrence", "removed")}
    assert str(changes[0]).startswith("occurrence")
