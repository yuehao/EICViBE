"""One supply for a magnet that several branches pass through, such as a septum."""

import math

import pytest

from eicvibe.domain.lattice import LatticeBuilder, SequentialIds
from eicvibe.domain.lattice.conventions import rigidity_from_P
from eicvibe.domain.machine import NamingArea, SharedMagnet, generate_machine_registry
from eicvibe.domain.machine.power import circuit_strengths, magnet_current

PC = 0.75e9
RIGIDITY = rigidity_from_P(PC, charge=-1)
ANGLE, LENGTH = -0.12, 0.401


def septum_lattice(drive_kicker: bool = False):
    """A ring whose kicker K is the stored-beam traversal of a septum the lines pass as bends."""
    b = LatticeBuilder("septum", ids=SequentialIds())
    reference = {"species_ref": "electron", "pc_ref": PC}
    field = {"MagneticMultipoleP": {"Kn0": ANGLE / LENGTH}}
    b.define("BEG", "BeginningEle", ReferenceP=reference)
    b.define("D", "Drift", length=2.0)
    b.define("B", "Bend", length=2.0, BendP={"angle_ref": math.pi / 2})
    b.define("K", "Kicker", length=0.4)
    b.define("SI", "Bend", length=LENGTH, BendP={"angle_ref": ANGLE, "Kn0_from_g_ref": False}, **field)
    b.define("SX", "Bend", length=LENGTH, BendP={"angle_ref": ANGLE, "Kn0_from_g_ref": False}, **field)
    b.define("Q", "Quadrupole", length=0.5, MagneticMultipoleP={"Kn1": 0.4})
    b.add_branch("ring", "periodic", ["BEG", "D", "K", "D", "B", "D", "B", "Q", "D", "B", "D", "B", "D"])
    b.add_branch("inj", "linac", ["BEG", "D", "SI"])
    b.add_branch("ext", "linac", ["BEG", "SX", "D"])
    b.connect_injection("inj", "ring", via=("SI_1", "K_1"), component="septum")
    b.connect_extraction("ring", "ext", via=("K_1", "SX_1"), component="septum")
    return b.build()


def with_supply(lattice, prefix="dr", home="circulation", driven=("injection", "extraction")):
    areas = (NamingArea(prefix=prefix, start=0.0, end=lattice.layout("ring").length + 1.0),)
    shared = {"septum": SharedMagnet(home=home, driven=driven)}
    return generate_machine_registry(lattice, {"ring": areas, "inj": None, "ext": None}, shared=shared)


def test_the_septum_is_one_device_at_every_traversal():
    lattice = septum_lattice()
    registry = with_supply(lattice)
    traversals = {t.occurrence for t in lattice.shared_components[0].traversals}
    [device] = [d for d in registry.devices if set(d.occurrences) & traversals]
    assert set(device.occurrences) == traversals
    assert device.name == "dr-sptm2-ps"  # the ring slot: its exit is 2.4 m from the start
    assert [c.name for c in registry.channels_of(device.id)] == ["dr-sptm2-ps:sp", "dr-sptm2-ps:rb"]
    # One magnet, one name, at every traversal; no traversal is named on its own.
    [slot] = [s for s in registry.slots if set(s.occurrences) & traversals]
    assert slot.name == "dr-sptm2" and set(slot.occurrences) == traversals
    assert registry.check_against(lattice) == []
    renamed = with_supply(lattice, prefix="xr")
    [again] = [d for d in renamed.devices if d.id == device.id]
    assert again.name == "xr-sptm2-ps"


def test_its_setting_drives_the_bends_and_never_the_stored_beam():
    lattice = septum_lattice()
    registry = with_supply(lattice)
    setting = registry.channel("dr-sptm2-ps:sp")
    [magnet] = registry.circuit(setting.binding.circuit).magnets  # one magnet, one transfer function ...
    driven = {lattice.placement(t).name for t in magnet.occurrences}
    assert driven == {"SI_1", "SX_1"} and magnet.parameter == "MagneticMultipoleP.Kn0"  # ... for two beams
    low, high = setting.control_limits
    design = ANGLE / LENGTH
    current = magnet_current(lattice, magnet, design, (low, high))
    assert low == 0.0 < current < high  # a positive current gives the design field of either sign
    strengths = circuit_strengths(lattice, registry.circuit(setting.binding.circuit), current)
    assert list(strengths.values()) == pytest.approx([design, design])  # one field, both beams
    assert all(
        v == 0.0 for v in circuit_strengths(lattice, registry.circuit(setting.binding.circuit), 0.0).values()
    )


def test_only_bends_can_be_driven_and_names_must_exist():
    lattice = septum_lattice()
    with pytest.raises(ValueError, match="must be a bend"):
        with_supply(lattice, driven=("circulation",))
    with pytest.raises(KeyError, match="no traversal 'stored'"):
        with_supply(lattice, home="stored")
    with pytest.raises(KeyError, match="no shared component"):
        generate_machine_registry(lattice, shared={"kicker": SharedMagnet(home="x", driven=())})
