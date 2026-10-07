"""Units, signs and reference conventions of the lattice description.

These conventions are part of the serialized contract (``format_version`` 1).
Engine adapters convert from them; nothing in this package converts to an
engine's units.

Units
    SI throughout: lengths in m, angles in rad, times in s, frequencies in Hz,
    voltages in V and fields in T. Energies and momenta are in eV (momentum as
    ``pc``). Charges are in units of the elementary charge. Phases, RF phases
    included, are in rad, as in PALS (which used rad / 2π until October 2026).

Longitudinal position
    ``length`` is the path length along the reference trajectory, i.e. the arc
    length for bends. Each branch starts at s = 0 at its entrance and s grows
    in the direction of beam travel. Positions are derived per revision and
    are not identities: inserting an element moves every later s.

Magnetic strengths
    ``MagneticMultipoleP`` holds normalized strengths
    Kn_n = (1/Bρ) ∂ⁿB_y/∂xⁿ and Ks_n likewise for the skew part, either per
    unit length (``Kn1``, in 1/m^(n+1)) or integrated over the element
    (``Kn1L``, in 1/m^n). Kn1 is the quadrupole k1 in 1/m²; Kn1 > 0 focuses
    horizontally. A kicker's kicks are the integrated dipole strengths
    ``Kn0L`` and ``Ks0L``, angles in rad with the multipole sign, so
    Kn0L > 0 deflects toward −x, the same way a positive bend angle does.
    MAD-X ``HKICK`` is therefore −Kn0L, while ``VKICK`` is +Ks0L.

    Normalized strengths refer to the branch reference momentum. This
    description does not say whether fields or normalized strengths stay
    fixed when the reference energy changes (PALS ``field_master``); a
    ramping scenario must state that explicitly.

Orientation
    Positive ``BendP.angle_ref`` bends the reference trajectory toward −x:
    clockwise in the floor plane seen from above (+y), as in MAD-X and Bmad.
    Rotations such as ``PatchP.y_rot`` are right-handed, as in PALS, so a
    patch that turns the frame like a bend of angle a has y_rot = −a.

Reference particle
    A branch's ``BeginningEle`` may carry ``ReferenceP``. ``pc_ref`` and
    ``E_tot_ref`` are both total quantities (momentum times c, total energy)
    for the whole particle, not per nucleon. When both are given for a known
    species they must agree through the species mass.

Species
    ``SPECIES`` holds the supported particles: ``electron``, ``positron``,
    ``antiproton``, and the ions ``proton``, ``deuteron``, ``helium3`` (the ³He
    nucleus), ``gold197`` (Au⁷⁹⁺) and ``uranium238`` (U⁹²⁺), all fully
    stripped. Other species names are accepted but their energies are not
    checked.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Species:
    """Rest mass and charge of a particle species."""

    name: str
    mass: float
    """Rest energy m c², in eV."""
    charge: int
    """Charge in units of the elementary charge."""
    mass_number: int | None = None
    """Nucleon number A, for ions."""
    atomic_number: int | None = None
    """Proton number Z, for ions."""


#: Speed of light in vacuum, in m/s; exact by the SI definition of the metre.
SPEED_OF_LIGHT = 299_792_458.0

# CODATA 2022 rest energies, in eV.
_ELECTRON_MASS = 0.51099895069e6
_PROTON_MASS = 938.27208943e6
_DEUTERON_MASS = 1875.61294500e6
_HELIUM3_MASS = 2808.39161112e6  # CODATA "helion"

# Bare-nucleus rest energies, in eV: the AME2020 atomic mass minus Z electrons
# plus their total binding energy. Uncertain at the keV level. Not A*u, which
# is 71 MeV too heavy for gold.
_GOLD197_MASS = 183_433.347107e6
_URANIUM238_MASS = 221_696.655183e6

SPECIES: dict[str, Species] = {
    s.name: s
    for s in (
        Species("electron", _ELECTRON_MASS, -1),
        Species("positron", _ELECTRON_MASS, +1),
        Species("antiproton", _PROTON_MASS, -1),
        Species("proton", _PROTON_MASS, +1, mass_number=1, atomic_number=1),
        Species("deuteron", _DEUTERON_MASS, +1, mass_number=2, atomic_number=1),
        Species("helium3", _HELIUM3_MASS, +2, mass_number=3, atomic_number=2),
        Species("gold197", _GOLD197_MASS, +79, mass_number=197, atomic_number=79),
        Species("uranium238", _URANIUM238_MASS, +92, mass_number=238, atomic_number=92),
    )
}

#: Relative and absolute (eV) tolerance for pc/E_tot consistency.
ENERGY_REL_TOL = 1e-9
ENERGY_ABS_TOL = 1.0


def total_energy(pc: float, mass: float) -> float:
    """Total energy for momentum ``pc`` and rest energy ``mass``, both in eV."""
    return math.hypot(pc, mass)


def rigidity_from_P(pc: float, nucleon_mass_num: int = 1, charge: int = 1) -> float:
    """Magnetic rigidity Bρ, in T·m, for momentum per nucleon ``pc`` in eV.

    For a heavy ion give its nucleon number ``nucleon_mass_num`` and charge
    ``charge``. For total momentum, keep ``nucleon_mass_num=1``.
    """
    return pc * nucleon_mass_num / abs(charge) / SPEED_OF_LIGHT


def rigidity_from_E(e_tot: float, mass: float, nucleon_mass_num: int = 1, charge: int = 1) -> float:
    """Magnetic rigidity Bρ, in T·m, for total energy per nucleon ``e_tot`` in eV.

    ``mass`` is the rest energy per nucleon in eV: the species mass divided by
    its nucleon number. For total energy and mass, keep ``nucleon_mass_num=1``.
    For gold at 110 GeV/u::

        gold = SPECIES["gold197"]
        rigidity_from_E(110e9, gold.mass / 197, nucleon_mass_num=197, charge=79)  # 914.9 T·m
    """
    if e_tot < mass:
        raise ValueError(f"total energy {e_tot} eV is below the rest energy {mass} eV")
    # (E - m)(E + m) instead of E² - m² keeps precision close to rest.
    return rigidity_from_P(math.sqrt((e_tot - mass) * (e_tot + mass)), nucleon_mass_num, charge)


def energies_agree(pc: float, e_tot: float, mass: float) -> bool:
    """Whether ``pc`` and ``e_tot`` describe the same particle of rest energy ``mass``."""
    return math.isclose(e_tot, total_energy(pc, mass), rel_tol=ENERGY_REL_TOL, abs_tol=ENERGY_ABS_TOL)
