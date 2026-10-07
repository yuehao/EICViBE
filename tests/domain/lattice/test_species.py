import math

import pytest
from pydantic import ValidationError

from eicvibe.domain.lattice import SPECIES, ReferenceP

U = 931.49410372e6  # CODATA 2022 atomic mass unit, eV
M_E = 0.51099895069e6  # CODATA 2022 electron, eV


def test_supported_species():
    assert set(SPECIES) == {
        "electron", "positron", "antiproton", "proton", "deuteron", "helium3", "gold197", "uranium238",
    }  # fmt: skip
    ions = {name: (s.mass_number, s.atomic_number, s.charge) for name, s in SPECIES.items() if s.mass_number}
    assert ions == {
        "proton": (1, 1, 1),
        "deuteron": (2, 1, 1),
        "helium3": (3, 2, 2),
        "gold197": (197, 79, 79),
        "uranium238": (238, 92, 92),
    }


@pytest.mark.parametrize(
    ("name", "atomic_mass_micro_u", "z"),
    [("gold197", 196_966_570.103, 79), ("uranium238", 238_050_786.936, 92)],
)
def test_heavy_ion_masses_are_bare_nuclei_from_ame2020(name, atomic_mass_micro_u, z):
    """Atomic mass (AME2020) minus Z electrons plus their total binding energy (AME2020 formula)."""
    binding = 14.4381 * z**2.39 + 1.55468e-6 * z**5.35
    nucleus = atomic_mass_micro_u * 1e-6 * U - z * M_E + binding
    assert SPECIES[name].mass == pytest.approx(nucleus, abs=1.0)


def test_gold_is_not_a_times_u():
    assert 197 * U - SPECIES["gold197"].mass == pytest.approx(70.99e6, rel=1e-3)


def test_reference_energy_check_for_gold():
    gold = SPECIES["gold197"]
    e_tot = 197 * 110e9  # 110 GeV per nucleon, total energy
    ReferenceP(species_ref="gold197", pc_ref=math.sqrt(e_tot**2 - gold.mass**2), E_tot_ref=e_tot)
    with pytest.raises(ValidationError, match="disagree"):
        ReferenceP(species_ref="gold197", pc_ref=math.sqrt(e_tot**2 - (197 * U) ** 2), E_tot_ref=e_tot)


def test_other_species_are_accepted_unchecked():
    ReferenceP(species_ref="#197Au+31", pc_ref=1e12, E_tot_ref=1e11)
