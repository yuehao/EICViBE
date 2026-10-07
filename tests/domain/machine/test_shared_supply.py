"""One supply for a magnet that several branches pass through, such as a septum."""

import math

import pytest

from eicvibe.domain.lattice import LatticeBuilder, SequentialIds
from eicvibe.domain.lattice.conventions import rigidity_from_P
from eicvibe.domain.machine import NamingArea, add_shared_magnet_supply, generate_machine_registry

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


def with_supply(lattice, prefix="dr", **options):
    registry = generate_machine_registry(lattice)
    areas = (NamingArea(prefix=prefix, start=0.0, end=lattice.layout("ring").length + 1.0),)
    options = {"home": "circulation", "driven": ("injection", "extraction"), "areas": areas, **options}
    return add_shared_magnet_supply(registry, lattice, "septum", **options)


def test_the_septum_is_one_device_at_every_traversal():
    lattice = septum_lattice()
    registry = with_supply(lattice)
    traversals = {t.occurrence for t in lattice.shared_components[0].traversals}
    [device] = [d for d in registry.devices if set(d.occurrences) & traversals]
    assert set(device.occurrences) == traversals
    assert device.name == "dr-sptm2-ps"  # the ring slot: its exit is 2.4 m from the start
    assert [c.name for c in registry.channels_of(device.id)] == ["dr-sptm2-ps:sp", "dr-sptm2-ps:rb"]
    assert registry.check_against(lattice) == []
    renamed = with_supply(lattice, prefix="xr")
    assert renamed.devices[-1].id == device.id and renamed.devices[-1].name == "xr-sptm2-ps"


def test_its_setting_drives_the_bends_and_never_the_stored_beam():
    lattice = septum_lattice()
    setting = with_supply(lattice).channel("dr-sptm2-ps:sp")
    driven = {lattice.placement(t).name for t in setting.binding.targets}
    assert driven == {"SI_1", "SX_1"} and setting.binding.parameter == "MagneticMultipoleP.Kn0"
    calibration, (low, high) = setting.binding.calibration, setting.control_limits
    design = ANGLE / LENGTH
    current = calibration.to_engineering(design, low, high, rigidity=RIGIDITY)
    assert low == 0.0 < current < high  # a positive current gives the design field of either sign
    assert calibration.to_model(current, rigidity=RIGIDITY) == pytest.approx(design)
    assert calibration.to_model(0.0, rigidity=RIGIDITY) == 0.0  # and 0 A no field


def test_only_bends_can_be_driven_and_names_must_exist():
    lattice = septum_lattice()
    with pytest.raises(ValueError, match="must be a bend"):
        with_supply(lattice, driven=("circulation",))
    with pytest.raises(KeyError, match="no traversal 'stored'"):
        with_supply(lattice, home="stored")
    with pytest.raises(KeyError, match="no shared component"):
        add_shared_magnet_supply(generate_machine_registry(lattice), lattice, "kicker", home="x", driven=())
