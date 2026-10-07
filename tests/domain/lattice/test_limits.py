import pytest
from pydantic import ValidationError

from eicvibe.domain.lattice import (
    LatticeBuilder,
    Limit,
    LimitEntry,
    LimitSource,
    ParameterLimits,
    SequentialIds,
    diagnose,
    placeholder_limits,
    tunable_parameters,
)


def quads() -> LatticeBuilder:
    b = LatticeBuilder("x", ids=SequentialIds())
    b.define("QF", "Quadrupole", length=0.5, MagneticMultipoleP={"Kn1": 1.3})  # ele_1
    b.define("QD", "Quadrupole", length=0.5, MagneticMultipoleP={"Kn1": -1.3})  # ele_2
    b.define("D", "Drift", length=2.0)
    b.add_branch("cell", "periodic", ["QF", "D", "QD", "D"])
    return b


def limit_codes(lattice, limits=None) -> list[tuple[str, str]]:
    return [(d.subject, d.code) for d in diagnose(lattice, limits) if d.code == "limit-exceeded"]


def test_every_tunable_parameter_of_every_element_gets_a_placeholder(fodo):
    limits = placeholder_limits(fodo)
    assert {(e.target, e.parameter) for e in limits.entries} == {
        (fodo.element(name).id, path) for name in ("QF", "QD") for path in tunable_parameters("Quadrupole")
    }
    assert all(e.limit.source is LimitSource.PLACEHOLDER for e in limits.entries)
    assert tunable_parameters("Drift") == ()


def test_placeholders_do_not_flag_ordinary_settings():
    assert limit_codes(quads().build()) == []


def test_per_element_limits_differ():
    lattice = quads().build()
    limits = placeholder_limits(lattice).with_limit("ele_1", "MagneticMultipoleP.Kn1", -1.25, 1.25)
    assert limit_codes(lattice, limits) == [("ele_1", "limit-exceeded")]  # QD keeps its placeholder
    [warning] = [d for d in diagnose(lattice, limits) if d.code == "limit-exceeded"]
    assert "1.3 1/m^2" in warning.message and "[-1.25, 1.25] (manual)" in warning.message


def test_occurrence_limit_takes_precedence_over_the_family():
    b = quads()
    trimmed = b.append("cell", "QF", MagneticMultipoleP={"Kn1": 1.35})
    lattice = b.build()
    limits = (
        ParameterLimits()
        .with_limit("ele_1", "MagneticMultipoleP.Kn1", -2.0, 2.0)
        .with_limit(trimmed, "MagneticMultipoleP.Kn1", -1.32, 1.32, source=LimitSource.EPICS, pv="Q:TRIM:I")
    )
    assert limit_codes(lattice, limits) == [(trimmed, "limit-exceeded")]
    assert limits.get(trimmed, "MagneticMultipoleP.Kn1").pv == "Q:TRIM:I"


def test_elements_missing_from_the_table_fall_back_to_the_kind_placeholder():
    b = quads()
    b.update_element("QF", MagneticMultipoleP={"Kn1": 2e3})
    assert limit_codes(b.build(), ParameterLimits()) == [("ele_1", "limit-exceeded")]


def test_cavity_voltage_and_kicker_limits():
    b = LatticeBuilder("x", ids=SequentialIds())
    b.define("RF", "RFCavity", length=1.0, RFP={"voltage": 3e7, "harmon": 2971})
    b.define("K", "Kicker", length=0.3, MagneticMultipoleP={"Kn0L": 2e-3})
    b.add_branch("line", "linac", ["RF", "K"])
    lattice = b.build()
    limits = (
        placeholder_limits(lattice)
        .with_limit("ele_1", "RFP.voltage", 0.0, 2e7)
        .with_limit("ele_2", "MagneticMultipoleP.Kn0L", -1e-3, 1e-3)
    )
    assert sorted(limit_codes(lattice, limits)) == [("ele_1", "limit-exceeded"), ("ele_2", "limit-exceeded")]


def test_limit_validation():
    with pytest.raises(ValidationError, match="exceeds maximum"):
        Limit(minimum=1.0, maximum=-1.0)
    with pytest.raises(ValidationError, match="no parameter"):
        LimitEntry(target="ele_1", parameter="MagneticMultipoleP.k1", limit=Limit())
    entry = LimitEntry(target="ele_1", parameter="MagneticMultipoleP.Kn1", limit=Limit())
    with pytest.raises(ValidationError, match="only one limit"):
        ParameterLimits(entries=(entry, entry))
    assert Limit(maximum=1.0).contains(-1e9) and not Limit(minimum=0.0).contains(-1e-9)


def test_tables_are_immutable_and_serializable(fodo):
    limits = placeholder_limits(fodo)
    changed = limits.with_limit(fodo.element("QF").id, "MagneticMultipoleP.Kn1", -1.5, 1.5)
    assert limits.get(fodo.element("QF").id, "MagneticMultipoleP.Kn1").source is LimitSource.PLACEHOLDER
    assert changed.get(fodo.element("QF").id, "MagneticMultipoleP.Kn1").maximum == 1.5
    assert ParameterLimits.model_validate_json(changed.model_dump_json()) == changed
