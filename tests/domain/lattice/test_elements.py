import pytest
from pydantic import ValidationError

from eicvibe.domain.lattice import ALLOWED_GROUPS, ElementDefinition, ElementKind, ParameterOverride


def element(**fields):
    return ElementDefinition(id="ele_1", name=fields.pop("name", "E"), **fields)


def test_legacy_kind_names_get_a_hint():
    element(kind="Bend", length=1.0, BendP={"angle_ref": 0.1})
    with pytest.raises(ValidationError, match="use 'Bend'"):
        element(kind="SBend", length=1.0)
    with pytest.raises(ValidationError, match="angle_ref/2"):
        element(kind="RBend", length=1.0)
    with pytest.raises(ValidationError, match="use 'Instrument'"):
        element(kind="Monitor")


def test_groups_must_be_allowed_for_the_kind():
    with pytest.raises(ValidationError, match="Drift does not accept BendP"):
        element(kind="Drift", length=1.0, BendP={"angle_ref": 0.1})
    element(kind="Drift", length=1.0, ApertureP={"x_max": 0.02})


@pytest.mark.parametrize("kind", ["Marker", "Fork", "BeginningEle", "Patch", "ReferenceChange", "BeamBeam"])
def test_control_kinds_have_zero_length(kind):
    with pytest.raises(ValidationError, match="zero length"):
        element(kind=kind, length=0.1)


def test_drift_needs_positive_length():
    with pytest.raises(ValidationError):
        element(kind="Drift")


def test_default_groups_are_dropped():
    quad = element(kind="Quadrupole", length=0.5, MagneticMultipoleP={"Kn1": 0.0}, ControlP={})
    assert quad.MagneticMultipoleP is None and quad.ControlP is None
    assert quad.groups() == {}


def test_parameter_returns_defaults_for_unset_groups():
    quad = element(kind="Quadrupole", length=0.5, MagneticMultipoleP={"Kn1": 1.5})
    assert quad.parameter("MagneticMultipoleP", "Kn1") == 1.5
    assert quad.parameter("BodyShiftP", "x_offset") == 0.0
    with pytest.raises(KeyError):
        quad.parameter("RFP", "voltage")
    with pytest.raises(KeyError):
        quad.parameter("MagneticMultipoleP", "k1")
    beginning = element(kind="BeginningEle")
    assert beginning.parameter("TwissP", "beta_a") is None


def test_with_groups_merges_replaces_and_removes():
    quad = element(kind="Quadrupole", length=0.5, MagneticMultipoleP={"Kn1": 1.0, "Ks1": 0.1})
    merged = quad.with_groups(MagneticMultipoleP={"Kn1": 2.0})
    assert (merged.MagneticMultipoleP.Kn1, merged.MagneticMultipoleP.Ks1) == (2.0, 0.1)
    assert quad.with_groups(MagneticMultipoleP=None).MagneticMultipoleP is None
    assert quad.MagneticMultipoleP.Kn1 == 1.0  # original untouched


def test_with_overrides_revalidates_the_whole_group():
    kicker = element(kind="Kicker", MagneticMultipoleP={"Kn0L": 1e-4})
    with pytest.raises(ValidationError, match="integrated Kn0L/Ks0L"):
        kicker.with_overrides([ParameterOverride(group="MagneticMultipoleP", parameter="Ks0", value=1e-4)])


def test_allowed_groups_follow_the_pals_kind_lists():
    assert set(ALLOWED_GROUPS) == set(ElementKind)
    assert all({"ControlP", "MetaP"} <= groups for groups in ALLOWED_GROUPS.values())
    assert "MagneticMultipoleP" in ALLOWED_GROUPS[ElementKind.INSTRUMENT]
    assert "TrackingP" not in ALLOWED_GROUPS[ElementKind.MARKER]
    assert "BodyShiftP" not in ALLOWED_GROUPS[ElementKind.PATCH]
    reference_change = {"ControlP", "MetaP", "ReferenceChangeP", "TrackingP"}
    assert ALLOWED_GROUPS[ElementKind.REFERENCE_CHANGE] == reference_change
