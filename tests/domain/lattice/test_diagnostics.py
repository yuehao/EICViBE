from eicvibe.domain.lattice import LatticeBuilder, SequentialIds, diagnose


def codes(builder: LatticeBuilder) -> list[str]:
    return [d.code for d in diagnose(builder.build())]


def one_element(kind: str, length: float = 0.0, **groups) -> LatticeBuilder:
    b = LatticeBuilder("x", ids=SequentialIds())
    b.define("E", kind, length=length, **groups)
    b.add_branch("line", "linac", ["E"])
    return b


def test_clean_lattices_have_no_warnings(fodo, machine):
    assert diagnose(fodo) == () and machine.diagnose() == ()


def test_rf_phase_in_degrees_is_flagged():
    assert codes(one_element("RFCavity", 1.2, RFP={"voltage": 3e7, "harmon": 2971, "phase": 180.0})) == [
        "rf-phase-units"
    ]


def test_active_cavity_without_frequency():
    assert codes(one_element("RFCavity", 1.2, RFP={"voltage": 3e7})) == ["rf-frequency-missing"]


def test_a_gun_without_frequency_is_a_dc_gun():
    assert codes(one_element("EGun", 0.3, RFP={"voltage": 3.2e5})) == []


def test_zero_length_quadrupole_with_per_length_strength():
    assert codes(one_element("Quadrupole", MagneticMultipoleP={"Kn1": 0.5})) == ["thin-thick-strength"]


def test_thin_multipole_uses_integrated_strengths():
    assert codes(one_element("Multipole", MagneticMultipoleP={"Ks2": 0.5})) == ["thin-thick-strength"]
    assert codes(one_element("Multipole", MagneticMultipoleP={"Kn2L": 0.5, "Kn4L": 10.0})) == []


def test_unknown_species_cannot_be_checked():
    b = one_element("BeginningEle", ReferenceP={"species_ref": "Au79+", "pc_ref": 1e11})
    assert codes(b) == ["species-unknown"]


def test_large_kick_and_per_length_kick_on_a_thin_kicker():
    assert codes(one_element("Kicker", 0.3, MagneticMultipoleP={"Kn0L": 0.2})) == ["limit-exceeded"]
    assert codes(one_element("Kicker", MagneticMultipoleP={"Ks0": 1e-3})) == ["thin-thick-strength"]


def test_partial_elliptical_aperture_is_inactive():
    assert codes(one_element("Drift", 1.0, ApertureP={"x_max": 0.02})) == ["aperture-inactive"]
    assert codes(one_element("Drift", 1.0, ApertureP={"x_max": 0.02, "shape": "RECTANGULAR"})) == []
    assert codes(one_element("Drift", 1.0, ApertureP={"material": "Cu"})) == []


def test_floor_of_a_forked_branch_is_ignored():
    b = one_element("Drift", 1.0)
    b.define("BEG", "BeginningEle", FloorP={"x": 1.0})
    b.add_branch("spur", "linac", ["BEG", "E"])
    assert codes(b) == []
    b.connect_extraction("line", "spur", at="E_1")
    assert codes(b) == ["floor-ignored"]


def test_overridden_occurrences_are_checked_separately():
    b = one_element("Quadrupole", 0.5, MagneticMultipoleP={"Kn1": 0.5})
    b.append("line", "E", BodyShiftP={"x_offset": 20.0})
    [warning] = diagnose(b.build())
    assert warning.code == "misalignment-large" and warning.subject == b.build().placement("E_2", "line").id


def test_empty_branch():
    b = LatticeBuilder("x")
    b.add_branch("line")
    assert codes(b) == ["empty-branch"]
