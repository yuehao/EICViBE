import pytest

from eicvibe.domain.lattice.conventions import SPECIES, SPEED_OF_LIGHT, rigidity_from_E, rigidity_from_P

GOLD = SPECIES["gold197"]


def test_rigidity_of_one_gev_per_c():
    assert rigidity_from_P(1e9) == pytest.approx(1e9 / SPEED_OF_LIGHT) == pytest.approx(3.3356409520)


def test_rigidity_at_rhic_design_energies():
    # RHIC runs 100 GeV/u gold and 250 GeV protons at nearly the same rigidity.
    assert rigidity_from_E(100e9, GOLD.mass / 197, nucleon_mass_num=197, charge=79) == pytest.approx(
        831.76, abs=0.01
    )
    assert rigidity_from_E(250e9, SPECIES["proton"].mass) == pytest.approx(833.90, abs=0.01)
    assert rigidity_from_E(18e9, SPECIES["electron"].mass) == pytest.approx(60.04, abs=0.01)


def test_per_nucleon_and_total_quantities_agree():
    per_nucleon = rigidity_from_E(110e9, GOLD.mass / 197, nucleon_mass_num=197, charge=79)
    total = rigidity_from_E(197 * 110e9, GOLD.mass, charge=79)
    assert per_nucleon == pytest.approx(total, rel=1e-12)
    assert rigidity_from_P(1e11, nucleon_mass_num=197, charge=79) == rigidity_from_P(197e11, charge=79)


def test_rigidity_from_energy_near_and_below_rest():
    mass = SPECIES["proton"].mass
    assert rigidity_from_E(mass, mass) == 0.0
    kinetic = 1.0  # eV; pc = sqrt(2 m T) to leading order
    assert rigidity_from_E(mass + kinetic, mass) == pytest.approx(
        (2 * mass * kinetic) ** 0.5 / SPEED_OF_LIGHT, rel=1e-6
    )
    with pytest.raises(ValueError, match="below the rest energy"):
        rigidity_from_E(0.9 * mass, mass)
