# ADR 0014: Magnets' field per ampere and supplies' limits from the designers' databases

Status: accepted (2026-10-16). Scope: transfer functions of source `design` (amends ADR 0003), the machine check of design currents against supply limits, `eicvibe.importers.magnet_tables`, `eicvibe.machines.ein_magnets`, `eicvibe.store.device_files.set_supplies`, and the EIN machine folder's `devices/power.yaml`.

## Context

Every supply of the EIN machine folder had a placeholder transfer function (1 mT per ampere for a dipole, 0.1 T/m for a quadrupole, 1 T/m² for a sextupole, 1 mT m for a corrector) and limits of twice its design current through it (ADR 0003). So the currents a person saw and set in the virtual accelerator were not the machine's: an LTR quadrupole read 109 A where its magnet takes 71.6 A, an RCS main dipole 54 A where it takes 287 A.

The EIC parameter databases in the EIN folder (`einlattices/01_PreInjector`, `03_RCS`, `04_LTR-RTE Transfer Line Parameter Database.xlsx`) give the magnets' coils and apertures, the currents their designers computed, and for some the most current they or their supplies take:

- the transfer lines' sheet (`magnets_feb2026`): every LTR and RTE dipole and quadrupole, its length, gap, turns, and its current at the line's top energy; the MIT Bates quadrupole's maximum (K1 6.75 at 0.75 GeV: 110.9 A);
- the RCS's (`9GeV Lat`, `Magnets9GeV`, `Magnets`): the 40-cell ring's dipole (6 turns, 4 cm gap), quadrupole (4 turns a pole, 2 cm bore), the strongest K1 and K2 at 9 GeV (0.662, 10), the 0.2 m sextupole (4 turns, 2 cm), the corrector redesigned for 13 mT m at 25.2 A (2025-10-10);
- the preinjector's (`magnets_Jul2025_latest`, `power supply`): its quadrupoles (10 turns a pole, 5 or 2 cm bore), solenoid coils (40 or 80 turns), gun dipole (40 turns, 10 cm gap), and its supplies' models (16 V 150 A for the linac quadrupoles, 30 V 50 A for the gun solenoids, 20 V 76 A and 100 V 150 A for the spin solenoids, 30 V 25 A for the gun dipole).

The BAR's database describes the older ring derived from the NSLS VUV ring, not the BAR of the machine folder (V9.6); the correctors of the lines and the preinjector have no coil in theirs.

## Decisions

1. **A transfer function from a magnet's design is of source `design`** (amends ADR 0003): its coil and aperture, as its designers give them; a property of the magnet type, not measured on the magnet. It sits between `placeholder` and `measured`; the viewer's inspector shows it.
2. **A magnet's field per ampere is its coil's, by the formula its designers used**, with the exact mu0 (`eicvibe.importers.magnet_tables`): a quadrupole's gradient 2 mu0 N I / r², a sextupole's 6 mu0 N I / r³, a solenoid's field mu0 N I / L, a dipole's 2 mu0 N I / g with N turns a coil (the lines' and the preinjector's sheets) or mu0 N I / g with N turns in all (the RCS's). The sheets round mu0 (1.257 G cm/A, 1/796178), so their currents are 3 to 5e-4 off ours.
3. **A supply's limits are its rating or its magnet's design maximum where the databases give one**: the MIT Bates quadrupole's 110.9 A (LTR, RTE); the RTE's QFH and QDH at 18 GeV, the line's top energy; the RCS's magnets at what its designers ask of them at 9 GeV (dipoles 3450 A, quadrupoles 791 A, sextupoles 79.7 A, correctors ±25.3 A); the preinjector's supply models. Elsewhere, twice the design current, as before, now through the real transfer function (or twice the designers' current where the lattice's is zero). A supply keeps its sign: unipolar, its wiring giving the field's sign, or bipolar.
4. **The machine check compares design currents with limits, and people see both.** `eicvibe check` (`MachineRegistry.power_problems`) reports a magnet whose design value needs a current beyond its supply's limits: a design its supply cannot run. A string's magnets are still checked for one current first. The viewer and the admin page show such a current in red, with the limits it is beyond (the inspector's power block, the setpoints, the settings table: the VA's settings say `beyond` and the element view its design current, beyond the limits too); and they mark a placeholder transfer function as one (an amber badge), since its current is not the real magnet's, beside `design` (blue) and `measured` (green). A setting's `source` is the least sure of its magnets'.
5. **The EIN machine folder takes them in its power file, in place** (`python -m eicvibe.machines.ein FOLDER --magnets`; `apply_ein_magnets`, `set_supplies`): the named transfer functions are added under `transfer:` with a comment saying where each comes from, and each circuit takes its transfer function and limits; nothing else in the file, or in the folder, changes. A magnet the databases give no coil for keeps its placeholder.

## Findings

- **Eight preinjector quadrupoles need more than their supplies give.** The magnet table (July 2025, "updated for 1.5 nC") asks 151 to 228 A of the three 5 cm quadrupoles before the linac and of five 2 cm linac quadrupoles (the strongest, QD2_L13 at 14.3 T/m, 227.6 A), while the supplies' sheet (September 2024) has 16 V, 150 A supplies for them ("max operating current" 122 A). One of the two is out of date; `eicvibe check` lists them.
- The RCS dipoles are three strings (the 156 main bends, and the arc ends' 2 and 4) of one magnet type: at 9 GeV the main string needs 3444 A, the others 1722 and 2583 A.
- The RCS's sextupoles have a supply each in the machine folder, where its designers' plan has two families (80 focusing, 80 defocusing) on two supplies.

## Not done yet

- The supplies' dynamics: the databases give some magnets' resistance and inductance (the RCS's 13 mT m corrector 5.2 Ohm; the BAR's and the 18 GeV RCS's magnets) and some supplies' voltages, which limit how fast a current can change; the virtual accelerator ramps settings over a fixed number of turns instead.
- Correctors of the transfer lines and the preinjector, septa, kickers, the RTE's dipoles, the spin line's magnets (whose energy is not settled, ADR 0012) and the BAR: no coil in the databases.
- Measured excitation curves, and saturation: the design transfer functions are linear (the RCS dipole at 0.65 T, the preinjector's quadrupoles at up to 0.29 T on the pole).

## Validation (2026-10-16)

- Design currents through the new transfer functions against the databases' own, where the magnet and its strength are the same: the LTR's quadrupoles 01 to 06 and dipoles 01, 02, 04, 05 within 3e-4; the RTE's quadrupoles at 10 GeV within 1e-3 (the sheet's Bρ = E/0.3 adds 7e-4), but QDH3, at zero in the lattice and K1 0.0085 in the sheet; the preinjector's 27 quadrupoles within 5.1e-4; the RCS's main dipole at injection 286.98 A against the sheet's 287.03 A. The LTR's quadrupoles 07 to 19 differ as they should: they were rematched for one LTR setting without a BAR mode (ADR 0007's EIN decisions).
- The EIN machine folder: 669 circuits took 13 transfer functions; 66 supplies kept placeholders; only `devices/power.yaml` changed; `eicvibe check` reports the eight preinjector quadrupoles above.
