import math

import pytest
from pydantic import ValidationError

from eicvibe.domain.lattice import (
    GROUP_TYPES,
    RFP,
    ApertureP,
    BeamBeamP,
    BendP,
    MagneticMultipoleP,
    ReferenceChangeP,
    ReferenceP,
    SolenoidP,
    TrackingP,
    TwissP,
    parameter_info,
    parameters_of,
)
from eicvibe.domain.lattice.parameters import parameter_adapter


@pytest.mark.parametrize("bad", [True, "1.0", math.nan, math.inf, None])
def test_real_fields_reject_non_finite_and_non_numbers(bad):
    with pytest.raises(ValidationError):
        MagneticMultipoleP(Kn1=bad)


def test_real_fields_normalize_ints_and_negative_zero():
    group = MagneticMultipoleP(Kn1=2, Kn2=-0.0)
    assert group.Kn1 == 2.0 and isinstance(group.Kn1, float)
    assert math.copysign(1.0, group.Kn2) == 1.0


def test_groups_are_immutable_and_closed():
    group = MagneticMultipoleP(Kn1=1.0)
    with pytest.raises(ValidationError):
        group.Kn1 = 2.0
    with pytest.raises(ValidationError):
        MagneticMultipoleP(kn1=1.0)  # legacy alias belongs to the importer
    with pytest.raises(TypeError):
        group.model_copy(update={"Kn1": 3.0})


def test_is_default():
    assert MagneticMultipoleP().is_default()
    assert MagneticMultipoleP(Kn1=0.0).is_default()
    assert not MagneticMultipoleP(Kn1=0.1).is_default()
    assert not TwissP(beta_a=1.0, beta_b=1.0).is_default()  # required fields carry information


def test_bend_requires_angle_and_one_edge_frame():
    with pytest.raises(ValidationError):
        BendP()
    with pytest.raises(ValidationError, match="same edge"):
        BendP(angle_ref=0.1, e1=0.05, e1_rect=0.01)
    BendP(angle_ref=0.1, e1=0.05, e2_rect=0.01)


def test_redundant_pairs_are_rejected():
    with pytest.raises(ValidationError, match="frequency or harmon"):
        RFP(frequency=5e8, harmon=2971)
    with pytest.raises(ValidationError, match="voltage or gradient"):
        RFP(voltage=1e6, gradient=1e6)
    with pytest.raises(ValidationError, match="Ksol or Bsol"):
        SolenoidP(Ksol=0.1, Bsol=1.0)
    with pytest.raises(ValidationError, match="at most one of dE_ref"):
        ReferenceChangeP(dE_ref=1e6, pc_ref=1e9)
    with pytest.raises(ValidationError, match="at most one of dtime_ref"):
        ReferenceChangeP(dtime_ref=1e-9, time_ref=0.0)


def test_rf_harmonic_is_a_positive_integer():
    assert RFP(harmon=2971.0).harmon == 2971
    for bad in (0, 2971.5, True):
        with pytest.raises(ValidationError):
            RFP(harmon=bad)


def test_aperture_limits_are_ordered_and_optional():
    ApertureP(x_max=0.02)
    with pytest.raises(ValidationError, match="x_min"):
        ApertureP(x_min=0.02, x_max=-0.02)


def test_aperture_vertices_go_once_around_the_center():
    square = [(0.02, 0.0), (0.0, 0.02), (-0.02, 0.0), (0.0, -0.02)]
    aperture = ApertureP(shape="VERTICES", vertices={"list": [{"point": p} for p in square]})
    assert aperture.vertices.list[1].point == (0.0, 0.02)
    arc = {"point": (0.0, 0.02), "radius": [0.02]}
    rounded = ApertureP(vertices={"list": [{"point": square[0]}, arc, *({"point": p} for p in square[2:])]})
    assert rounded.vertices.list[1].radius == (0.02, 0.02)
    with pytest.raises(ValidationError, match="needs vertices"):
        ApertureP(shape="VERTICES")
    with pytest.raises(ValidationError, match="counterclockwise"):
        ApertureP(vertices={"list": [{"point": p} for p in reversed(square)]})
    with pytest.raises(ValidationError, match="counterclockwise"):  # the center lies on the outline
        ApertureP(vertices={"list": [{"point": p} for p in square[:3]]})
    shifted = [{"point": (x + 1.0, y)} for x, y in square]
    ApertureP(vertices={"center": (1.0, 0.0), "absolute_vertices": True, "list": shifted})


def test_multipole_orders_are_per_length_or_integrated():
    MagneticMultipoleP(Kn0L=1e-3, Ks0L=-1e-3, Kn1=0.5, tilt1=0.1)
    with pytest.raises(ValidationError, match="integrated Kn1L/Ks1L"):
        MagneticMultipoleP(Kn1=0.5, Ks1L=0.01)
    with pytest.raises(ValidationError):
        MagneticMultipoleP(tilt=0.1)  # PALS tilts are per order


def test_tracking_settings_are_tables_per_program():
    tracking = TrackingP(xsuite={"num_steps": 10, "exact": True, "ds": 0.3}, bmad={})
    assert tracking.programs() == {"xsuite": {"num_steps": 10, "exact": True, "ds": 0.3}}
    assert TrackingP(bmad={}).is_default() and not tracking.is_default()
    for bad in ({"xsuite": {"ds": math.nan}}, {"x-suite": {"ds": 0.3}}, {"xsuite": 3}):
        with pytest.raises(ValidationError):
            TrackingP(**bad)
    with pytest.raises(KeyError, match="cannot be overridden"):
        parameter_adapter("TrackingP", "xsuite")


def test_unset_beam_beam_parameters_are_none():
    assert BeamBeamP().is_default() and BeamBeamP().beta_x is None
    with pytest.raises(ValidationError):
        BeamBeamP(sigma_x=0.0)


def test_reference_energy_must_agree_for_known_species():
    ReferenceP(species_ref="electron", pc_ref=8999999985.0, E_tot_ref=9e9)
    with pytest.raises(ValidationError, match="disagree"):
        ReferenceP(species_ref="proton", pc_ref=9e9, E_tot_ref=9e9)
    with pytest.raises(ValidationError, match="rest energy"):
        ReferenceP(species_ref="proton", E_tot_ref=1e8)
    ReferenceP(species_ref="Au79+", pc_ref=1e11, E_tot_ref=2e11)  # unknown: diagnose() warns


def test_twiss_betas_have_no_default():
    with pytest.raises(ValidationError):
        TwissP(beta_a=1.0)
    with pytest.raises(ValidationError):
        TwissP(beta_a=0.0, beta_b=1.0)
    with pytest.raises(ValidationError):
        TwissP(beta_x=1.0, beta_y=1.0)  # PALS uses normal-mode names


def test_twiss_coupling_and_dispersion_slope():
    TwissP(beta_a=1.0, beta_b=1.0, cmat11=0.6, cmat22=0.6, cmat12=0.3, cmat21=-0.3)
    with pytest.raises(ValidationError, match="determinant"):
        TwissP(beta_a=1.0, beta_b=1.0, cmat11=1.2, cmat22=1.2)
    TwissP(beta_a=1.0, beta_b=1.0, etap_x=0.1, deta_y_ds=0.1)
    with pytest.raises(ValidationError, match="etap_x or deta_x_ds"):
        TwissP(beta_a=1.0, beta_b=1.0, etap_x=0.1, deta_x_ds=0.1)


def test_every_real_valued_field_documents_a_unit_or_is_dimensionless():
    info = parameter_info("MagneticMultipoleP", "Kn1")
    assert (info.unit, info.default, info.required) == ("1/m^2", 0.0, False)
    assert parameter_info("RFP", "phase").unit == "rad"
    assert parameter_info("BendP", "edge1_int").unit == "m"
    assert parameter_info("BendP", "angle_ref").required
    twiss_units = {p.name: p.unit for p in parameters_of("TwissP")}
    assert (twiss_units["cmat12"], twiss_units["cmat21"], twiss_units["dalpha_a_dpz"]) == ("m", "1/m", None)
    assert [p.name for p in parameters_of("SolenoidP")] == ["Ksol", "Bsol"]
    schema = MagneticMultipoleP.model_json_schema()
    assert schema["properties"]["Kn1"]["unit"] == "1/m^2"


def test_parameter_adapter_applies_field_bounds():
    assert parameter_adapter("ApertureP", "thickness").validate_python(1) == 1.0
    with pytest.raises(ValidationError):
        parameter_adapter("ApertureP", "thickness").validate_python(-0.01)
    with pytest.raises(KeyError):
        parameter_adapter("BendP", "angle")
    with pytest.raises(KeyError):
        parameter_adapter("MisalignP", "dx")


def test_group_registry_names_match_classes():
    assert all(name == cls.__name__ == cls.group_name for name, cls in GROUP_TYPES.items())
