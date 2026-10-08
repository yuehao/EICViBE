import pytest
from pydantic import ValidationError

from eicvibe.domain.lattice import LatticeBuilder, SequentialIds
from eicvibe.domain.lattice.conventions import rigidity_from_P
from eicvibe.domain.machine import (
    Calibration,
    Channel,
    ChannelRole,
    MachineRegistry,
    NamingArea,
    NamingScheme,
    Observable,
    ObservableBinding,
    ParameterBinding,
    generate_registry,
)
from eicvibe.domain.machine.power import magnet_current, magnet_strength

RIGIDITY = rigidity_from_P(18e9, charge=-1)
SECTOR = (NamingArea(prefix="es6", start=0.0, end=10.0),)


def ring_builder() -> LatticeBuilder:
    b = LatticeBuilder("va-ring", ids=SequentialIds())
    b.define("BEG", "BeginningEle", ReferenceP={"species_ref": "electron", "pc_ref": 18e9})
    b.define("QF", "Quadrupole", length=0.5, MagneticMultipoleP={"Kn1": 0.5})
    b.define("QD", "Quadrupole", length=0.5, MagneticMultipoleP={"Kn1": -0.5})
    b.define("SF", "Sextupole", length=0.2)
    b.define("HC", "Kicker", tags=["HCOR"])
    b.define("KK", "Kicker")
    b.define("BPM", "Instrument")
    b.define("CAV", "RFCavity", length=1.0, RFP={"voltage": 5e6, "harmon": 100})
    b.define("D", "Drift", length=1.0)
    b.add_branch(
        "ring", "periodic", ["BEG", "QF", "HC", "BPM", "D", "QD", "KK", "BPM", "D", "SF", "CAV", "D"]
    )
    return b


@pytest.fixture
def lattice():
    return ring_builder().build()


@pytest.fixture
def registry(lattice):
    return generate_registry(lattice, areas=SECTOR)


def test_devices_are_named_from_their_slots(lattice, registry):
    names = {c.name for c in registry.channels}
    assert {"es6-q1-ps:sp", "es6-q1-ps:rb", "es6-q2-ps:sp", "es6-sx3-ps:sp", "es6-th1-ps:sp"} <= names
    assert {"es6-t2-ps:sp:h", "es6-t2-ps:rb:v", "es6-b1:rdx", "es6-b2:rdy", "es6-cav4:sp:volt"} <= names
    assert {"es-btun-rdx", "es-btun-rdy"} <= names
    assert registry.check_against(lattice) == []
    assert registry.channel("es6-q1-ps:rb").readback_of == "es6-q1-ps:sp"
    assert [c.name for c in registry.channels_of("es6-th1-ps")] == ["es6-th1-ps:sp", "es6-th1-ps:rb"]


def test_naming_problems_list_tokens_awaiting_approval(registry):
    problems = registry.naming_problems()
    assert "es6-t2-ps:sp:h: signal 'h' not in the repository" in problems
    assert "es6-cav4:sp:volt: signal 'volt' not in the repository" in problems
    assert not any(p.startswith(("es6-q1", "es6-b1", "es-btun")) for p in problems)


def test_each_magnet_has_a_circuit_reproducing_its_design_strength(lattice, registry):
    for name, occurrence, design in (("es6-q1-ps:sp", "QF_1", 0.5), ("es6-q2-ps:sp", "QD_1", -0.5)):
        channel = registry.channel(name)
        circuit = registry.circuit(channel.binding.circuit)
        [magnet] = circuit.magnets  # powered on its own
        assert magnet.occurrences == (lattice.placement(occurrence, branch="ring").id,)
        assert magnet.parameter == "MagneticMultipoleP.Kn1" and magnet.polarity == (1 if design > 0 else -1)
        current = magnet_current(lattice, magnet, design, channel.control_limits)
        assert 0.0 < current < channel.control_limits[1]  # positive currents, polarity in the wiring
        assert magnet_strength(lattice, magnet, magnet.occurrences[0], current) == pytest.approx(
            design, rel=1e-12
        )
        assert current == pytest.approx(abs(design) * RIGIDITY / 0.1)  # placeholder 0.1 T/m per A
    assert registry.channel("es6-sx3-ps:sp").control_limits == (-100.0, 100.0)  # off, so either polarity
    corrector = registry.circuit(registry.channel("es6-th1-ps:sp").binding.circuit)
    assert corrector.magnets[0].parameter == "MagneticMultipoleP.Kn0L"


def test_measurements_bind_to_observables(lattice, registry):
    x = registry.channel("es6-b2:rdx")
    assert x.role is ChannelRole.MEASUREMENT and x.unit == "mm"
    assert x.binding == ObservableBinding(
        observable=Observable.ORBIT_X, target=lattice.placement("BPM_2", branch="ring").id, scale=1e3
    )
    assert registry.channel("es-btun-rdy").binding.target == lattice.layout("ring").id


def test_rf_settings_are_in_engineering_units(registry):
    voltage = registry.channel("es6-cav4:sp:volt")
    assert voltage.unit == "MV" and voltage.binding.calibration.to_model(5.0) == 5e6
    phase = registry.channel("es6-cav4:sp:phase").binding.calibration
    assert phase.to_model(180.0) == pytest.approx(3.141592653589793)


def test_names_follow_the_scheme_and_devices_keep_their_ids(lattice, registry):
    scheme = NamingScheme(supply="psu", setpoint="set", readback="read", position_x="x")
    renamed = generate_registry(
        lattice, areas=(NamingArea(prefix="rcb1", start=0.0, end=10.0),), scheme=scheme
    )
    names = {c.name for c in renamed.channels}
    assert {"rcb1-q1-psu:set", "rcb1-q1-psu:read", "rcb1-b1:x"} <= names
    assert [d.id for d in renamed.devices] == [d.id for d in registry.devices]


def test_default_area_is_named_after_the_branch(lattice):
    registry = generate_registry(lattice)
    assert "ring-q1-ps:sp" in {c.name for c in registry.channels}
    assert "area 'ring'" in registry.naming_problems()[0]


def test_calibration_inversion_and_rigidity():
    quadratic = Calibration(coefficients=(0.0, 0.1, 1e-4), engineering_unit="A", field_unit="T/m")
    current = quadratic.to_engineering(0.5, 0.0, 500.0, rigidity=RIGIDITY)
    assert quadratic.to_model(current, rigidity=RIGIDITY) == pytest.approx(0.5, rel=1e-12)
    with pytest.raises(ValueError, match="rigidity"):
        quadratic.to_model(10.0)
    with pytest.raises(ValueError, match="outside the calibrated range"):
        quadratic.to_engineering(50.0, 0.0, 1.0, rigidity=RIGIDITY)


def test_channel_rules():
    with pytest.raises(ValidationError, match="not an EIC name"):
        Channel(name="QF:I", device="dev_1", role="measurement")
    with pytest.raises(ValidationError, match="needs control limits"):
        Channel(name="es6-q1-ps:sp", device="dev_1", role="setting")
    with pytest.raises(ValidationError, match="readback_of is required"):
        Channel(name="es6-q1-ps:rb", device="dev_1", role="readback")
    with pytest.raises(ValidationError, match="only a measurement"):
        Channel(
            name="es6-q1-ps:sp",
            device="dev_1",
            role="setting",
            control_limits=(0, 1),
            binding=ObservableBinding(observable="orbit_x", target="occ_1"),
        )
    with pytest.raises(ValidationError, match="no parameter"):
        ParameterBinding(targets=("ele_1",), parameter="MagneticMultipoleP.k1")


def test_registry_consistency(registry):
    data = registry.model_dump()
    data["channels"] = [*data["channels"], data["channels"][0]]
    with pytest.raises(ValidationError, match="duplicate channel name 'es6-q1-ps:sp'"):
        MachineRegistry.model_validate(data)
    data = registry.model_dump()
    readback = {"name": "es6-b1:rd", "device": data["channels"][0]["device"], "role": "readback"}
    data["channels"] = [*data["channels"], {**readback, "readback_of": "es6-b1:rdx"}]
    with pytest.raises(ValidationError, match="not a setting"):
        MachineRegistry.model_validate(data)
    assert MachineRegistry.model_validate_json(registry.model_dump_json()) == registry


def test_check_against_reports_bindings_that_do_not_resolve(registry):
    b = ring_builder()
    b.update_branch("ring", kind="linac")
    linac = b.build()
    assert registry.check_against(linac) == [
        "es-btun-rdx: a tune needs a periodic branch, not 'br_1'",
        "es-btun-rdy: a tune needs a periodic branch, not 'br_1'",
    ]
    data = registry.model_dump()
    data["circuits"][0]["magnets"][0]["occurrences"] = [linac.placement("D_1", branch="ring").id]
    problems = MachineRegistry.model_validate(data).check_against(linac)
    assert "circuit ring/QF_1: Drift 'D' has no MagneticMultipoleP" in problems


def test_generation_needs_a_reference_particle():
    b = LatticeBuilder("bare")
    b.define("Q", "Quadrupole", length=0.5)
    b.add_branch("line", "linac", ["Q"])
    with pytest.raises(ValueError, match="reference particle"):
        generate_registry(b.build())


def test_every_element_with_an_abbreviation_has_an_eic_name(lattice, registry):
    names = {p.name: registry.slot_of(p.id) for p in lattice.layout("ring").placements}
    assert names["BEG_1"] is None  # bookkeeping, no slot
    assert names["QF_1"].name == "es6-q1" and names["D_1"].name.startswith(
        "es6-dt"
    )  # drifts too, as in the ESR table
    assert all(s.name.split("-")[0] == "es6" for s in registry.slots)
    assert registry.check_against(lattice) == []
    supply = next(
        d for d in registry.devices if d.name.endswith("-ps") and d.occurrences == names["QF_1"].occurrences
    )
    assert supply.name == names["QF_1"].name + "-ps"


def test_eic_names_are_unique_and_name_one_element_each(registry):
    from eicvibe.domain.machine import Slot

    first, second = registry.slots[:2]
    with pytest.raises(ValidationError, match="duplicate element name"):
        MachineRegistry(
            name="r", lattice="va-ring", slots=(first, Slot(name=first.name, occurrences=second.occurrences))
        )
    with pytest.raises(ValidationError, match="duplicate element occurrence"):
        MachineRegistry(
            name="r", lattice="va-ring", slots=(first, Slot(name=second.name, occurrences=first.occurrences))
        )
