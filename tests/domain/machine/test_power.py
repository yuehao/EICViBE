"""Power circuits: strings of magnets on one supply, and transfer functions from current to strength."""

import math

import pytest
from pydantic import ValidationError

from eicvibe.domain.lattice import ElementKind, LatticeBuilder, SequentialIds
from eicvibe.domain.lattice.conventions import rigidity_from_P
from eicvibe.domain.machine import MachineRegistry, generate_registry
from eicvibe.domain.machine.power import (
    CalibrationSource,
    CircuitMagnet,
    PolynomialTransfer,
    PowerCircuit,
    ServiceTransfer,
    TableTransfer,
    circuit_strengths,
    magnet_current,
    magnet_strength,
    total_strengths,
    transfer_current,
    transfer_field,
)

PC = 2e9
RIGIDITY = rigidity_from_P(PC, charge=-1)
CELLS = 4


def ring():
    """Four FODO cells: one dipole family, two quadrupole families, correctors."""
    b = LatticeBuilder("ring", ids=SequentialIds())
    b.define("BEG", "BeginningEle", ReferenceP={"species_ref": "electron", "pc_ref": PC})
    b.define("B", "Bend", length=2.0, BendP={"angle_ref": 2 * math.pi / (2 * CELLS)})
    b.define("QF", "Quadrupole", length=0.5, MagneticMultipoleP={"Kn1": 0.6})
    b.define("QD", "Quadrupole", length=0.5, MagneticMultipoleP={"Kn1": -0.6})
    b.define("QT", "Quadrupole", length=0.3, MagneticMultipoleP={"Kn1L": 0.03})
    b.define("C", "Kicker", length=0.1, tags=["HCOR"])
    b.define("D", "Drift", length=1.0)
    b.add_branch("ring", "periodic", ["BEG", "QT"] + ["QF", "D", "B", "D", "C", "QD", "D", "B", "D"] * CELLS)
    return b.build()


def linear(gain, unit="T/m", integrated=False):
    return PolynomialTransfer(coefficients=(0.0, gain), field_unit=unit, integrated=integrated)


def test_main_dipoles_and_quad_families_are_strings_others_are_powered_alone():
    lattice = ring()
    alone = generate_registry(lattice)
    strung = generate_registry(lattice, series=(ElementKind.BEND, ElementKind.QUADRUPOLE))
    sizes = sorted((len(c.magnets), c.name) for c in strung.circuits)
    assert sizes[-3:] == [(4, "QD"), (4, "QF"), (8, "B")]  # one supply per family
    assert {n for n, _ in sizes[:-3]} == {1}  # QT (alone in its family) and the correctors
    dipoles = next(c for c in strung.circuits if c.name == "B")
    supply = next(d for d in strung.devices if d.id == dipoles.device)
    assert supply.name == "ring-dh4-ps" and len(supply.occurrences) == 8  # named after its first magnet
    assert all(c.name != "B" for c in alone.circuits) and len([c for c in alone.circuits if c.series]) == 0
    assert (
        sum(1 for c in alone.circuits if c.magnets[0].parameter == "MagneticMultipoleP.Kn0") == 8
    )  # bends too
    corrector = next(d for d in strung.devices if d.name.startswith("ring-th"))
    assert corrector.id in {d.id for d in alone.devices}  # magnets powered alone keep their device ids
    assert strung.check_against(lattice) == [] and strung.power_problems(lattice) == []


def test_a_string_current_gives_every_magnet_its_design_strength():
    lattice = ring()
    registry = generate_registry(lattice, series=(ElementKind.BEND, ElementKind.QUADRUPOLE))
    dipoles = next(c for c in registry.circuits if c.name == "B")
    limits = registry.setting_of(dipoles.id).control_limits
    current = magnet_current(lattice, dipoles.magnets[0], math.pi / 4 / 2.0, limits)
    assert current == pytest.approx(math.pi / 8 * RIGIDITY / 1e-3)  # placeholder 1 mT per A
    strengths = circuit_strengths(lattice, dipoles, current)
    assert len(strengths) == 8 and all(v == pytest.approx(math.pi / 8) for v in strengths.values())
    integrated = next(c for c in registry.circuits if c.name == "ring/QT_1")  # QT: Kn1L, alone
    [magnet] = integrated.magnets
    assert magnet.parameter == "MagneticMultipoleP.Kn1L"  # the lattice gives it integrated ...
    assert not magnet.transfer.integrated and magnet.transfer.field_unit == "T/m"  # ... the curve per length
    current = magnet_current(lattice, magnet, 0.03, registry.setting_of(integrated.id).control_limits)
    assert magnet_strength(lattice, magnet, magnet.occurrences[0], current) == pytest.approx(0.03)


def test_a_magnet_unlike_the_rest_of_its_string_is_reported():
    lattice = ring()
    registry = generate_registry(lattice, series=(ElementKind.QUADRUPOLE,))
    data = registry.model_dump()
    focusing = next(c for c in data["circuits"] if c["name"] == "QF")
    measured = {"kind": "polynomial", "coefficients": [0.0, 0.102], "field_unit": "T/m", "source": "measured"}
    focusing["magnets"][2]["transfer"] = measured  # its measured gradient is 2 % higher
    problems = MachineRegistry.model_validate(data).power_problems(lattice)
    assert len(problems) == 1 and problems[0].startswith("QF: its magnets need") and "QF_3" in problems[0]


def test_a_trim_adds_to_its_string():
    lattice = ring()
    registry = generate_registry(lattice, series=(ElementKind.QUADRUPOLE,))
    string = next(c for c in registry.circuits if c.name == "QF")
    target = string.magnets[0].occurrences[0]
    trim = PowerCircuit(
        id="trim",
        name="QF_1-trim",
        device=string.device,
        magnets=(
            CircuitMagnet(occurrences=(target,), parameter="MagneticMultipoleP.Kn1", transfer=linear(0.01)),
        ),
    )
    main = magnet_current(lattice, string.magnets[0], 0.6, registry.setting_of(string.id).control_limits)
    totals = total_strengths(lattice, (string, trim), {string.id: main, trim.id: 2.0})
    key = (target, "MagneticMultipoleP.Kn1")
    assert totals[key] == pytest.approx(0.6 + 0.01 * 2.0 / RIGIDITY)  # main + trim
    others = [v for (o, _), v in totals.items() if o != target]
    assert others == pytest.approx([0.6] * 3)


def test_a_measured_table_interpolates_and_inverts():
    curve = TableTransfer(
        currents=(0.0, 100.0, 200.0, 300.0), fields=(0.0, 1.0, 1.9, 2.6), field_unit="T", source="measured"
    )
    assert curve.source is CalibrationSource.MEASURED
    assert transfer_field(curve, 150.0) == pytest.approx(1.45)  # between points
    assert transfer_field(curve, 350.0) == pytest.approx(2.95)  # extrapolated along the last segment
    assert transfer_current(curve, 2.25, (0.0, 300.0)) == pytest.approx(250.0)
    with pytest.raises(ValidationError, match="strictly monotonic"):
        TableTransfer(currents=(0.0, 1.0, 2.0), fields=(0.0, 1.0, 1.0), field_unit="T")
    with pytest.raises(ValidationError, match="currents must increase"):
        TableTransfer(currents=(0.0, 2.0, 1.0), fields=(0.0, 1.0, 2.0), field_unit="T")


def test_a_polynomial_inverts_within_the_supply_range():
    saturating = PolynomialTransfer(coefficients=(0.0, 0.01, -1e-6), field_unit="T/m")
    current = transfer_current(saturating, 0.5, (0.0, 1000.0))
    assert transfer_field(saturating, current) == pytest.approx(0.5) and current > 50.0
    with pytest.raises(ValueError, match="outside the range"):
        transfer_current(saturating, 100.0, (0.0, 1000.0))


class Hysteretic:
    """A toy transfer-function service: the field is lower on the way down than on the way up."""

    def __init__(self):
        self.last, self.down = {}, {}

    def field(self, transfer, current):
        previous = self.last.get(transfer.magnet, 0.0)
        if current != previous:
            self.down[transfer.magnet] = current < previous
        self.last[transfer.magnet] = current
        return 0.01 * current - (0.002 if self.down.get(transfer.magnet) else 0.0)

    def current(self, transfer, field):
        return (field + (0.002 if self.down.get(transfer.magnet) else 0.0)) / 0.01


def test_a_service_handles_hysteresis_and_a_fallback_serves_without_it():
    transfer = ServiceTransfer(
        service="magnet-tf", magnet="QF-017", hysteresis=True, field_unit="T/m", fallback=linear(0.0099)
    )
    services = {"magnet-tf": Hysteretic()}
    up = transfer_field(transfer, 100.0, services)
    transfer_field(transfer, 150.0, services)
    down = transfer_field(transfer, 100.0, services)  # the same current, reached from above
    assert up == pytest.approx(1.0) and down == pytest.approx(0.998)
    assert transfer_current(transfer, 0.998, (0.0, 200.0), services) == pytest.approx(100.0)
    assert transfer_field(transfer, 100.0) == pytest.approx(0.99)  # no service: the fallback
    with pytest.raises(LookupError, match="no service"):
        transfer_field(ServiceTransfer(service="magnet-tf", magnet="QF-017", field_unit="T/m"), 100.0)
    with pytest.raises(ValidationError, match="same field"):
        ServiceTransfer(service="s", magnet="m", field_unit="T", fallback=linear(1.0, "T/m"))


def test_magnets_of_one_length_and_strength_share_a_string_whatever_their_names():
    b = LatticeBuilder("ring", ids=SequentialIds())
    b.define("BEG", "BeginningEle", ReferenceP={"species_ref": "electron", "pc_ref": PC})
    b.define("QF", "Quadrupole", length=0.5, MagneticMultipoleP={"Kn1": 0.6})
    b.define("QF_TWIN", "Quadrupole", length=0.5, MagneticMultipoleP={"Kn1L": 0.3})  # the same magnet
    b.define("QF_LONG", "Quadrupole", length=0.6, MagneticMultipoleP={"Kn1": 0.6})  # a longer one
    b.define("QOFF", "Quadrupole", length=0.5)  # off
    b.define("D", "Drift", length=1.0)
    quads = ["QF", "QF_TWIN", "QF", "QF_LONG", "QF_LONG", "QOFF", "QOFF", "QF"]
    b.add_branch("ring", "periodic", ["BEG"] + [e for q in quads for e in (q, "D")])
    b.set_overrides(("ring", "QF_3"), MagneticMultipoleP={"Kn1": 0.61})  # one magnet retuned
    lattice = b.build()
    registry = generate_registry(lattice, series=(ElementKind.QUADRUPOLE,))
    strings = {
        c.name: [lattice.placement(m.occurrences[0]).name for m in c.magnets]
        for c in registry.circuits
        if c.series
    }
    assert strings == {"QF+QF_TWIN": ["QF_1", "QF_TWIN_1", "QF_2"], "QF_LONG": ["QF_LONG_1", "QF_LONG_2"]}
    assert registry.power_problems(lattice) == []  # one current gives each its design strength


def test_a_line_given_by_fields_has_supplies_that_set_fields():
    """An injector line (ADR 0012) gives its magnets by their fields: its supplies set those, with no
    rigidity in the conversion, also for magnets at zero; solenoids get supplies; screens are no BPMs."""
    b = LatticeBuilder("injector", ids=SequentialIds())
    b.define("BEG", "BeginningEle", ReferenceP={"species_ref": "electron", "pc_ref": 0.65e6})
    b.define("SOL", "Solenoid", length=0.1, SolenoidP={"Bsol": 0.0237})
    b.define("Q", "Quadrupole", length=0.1, MagneticMultipoleP={"Bn1": -1.72})
    b.define("Q0", "Quadrupole", length=0.1)
    b.define("T", "Kicker")
    b.define("BPM", "Instrument")
    b.define("YAG", "Instrument", tags=["SCREEN"])
    b.define("D", "Drift", length=0.5)
    b.add_branch("line", "linac", ["BEG", "SOL", "D", "Q", "D", "Q0", "T", "BPM", "D", "YAG"])
    lattice = b.build()
    registry = generate_registry(lattice)
    sets = {
        lattice.placement(o).name: m.parameter
        for c in registry.circuits
        for m in c.magnets
        for o in m.occurrences
    }
    assert sets["SOL_1"] == "SolenoidP.Bsol" and sets["Q_1"] == "MagneticMultipoleP.Bn1"
    assert sets["Q0_1"] == "MagneticMultipoleP.Bn1"  # at zero, in the line's form
    assert {p for n, p in sets.items() if n == "T_1"} <= {
        "MagneticMultipoleP.Bn0L",
        "MagneticMultipoleP.Bs0L",
    }
    bpms = [d.name for d in registry.devices if d.type.value == "bpm"]
    assert len(bpms) == 1  # the screen is not one
    quad = next(
        c
        for c in registry.circuits
        if any(lattice.placement(o).name == "Q_1" for m in c.magnets for o in m.occurrences)
    )
    [magnet] = quad.magnets
    current = magnet_current(lattice, magnet, -1.72, (-1e3, 1e3))
    assert magnet_strength(lattice, magnet, magnet.occurrences[0], current) == pytest.approx(-1.72)
    assert abs(current) == pytest.approx(1.72 / magnet.transfer.coefficients[1])  # the field: no rigidity


def test_a_design_beyond_its_supplys_limits_is_reported():
    lattice = ring()
    registry = generate_registry(lattice, series=(ElementKind.QUADRUPOLE,))
    data = registry.model_dump()
    alone = next(c for c in data["circuits"] if c["name"] == "ring/QT_1")
    # A weaker magnet than its supply was sized for: its design needs more current than the supply gives.
    alone["magnets"][0]["transfer"] = {
        "kind": "polynomial",
        "coefficients": [0.0, 0.001],
        "field_unit": "T/m",
        "source": "design",
    }
    problems = MachineRegistry.model_validate(data).power_problems(lattice)
    assert len(problems) == 1 and problems[0].startswith("ring/QT_1: QT_1 needs")
    assert "beyond its supply's limits [0, 100]" in problems[0]
    assert CalibrationSource("design") is CalibrationSource.DESIGN
