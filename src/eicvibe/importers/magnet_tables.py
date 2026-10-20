"""The magnet tables of the EIN's parameter databases, as EICViBE reads them (ADR 0014).

The EIC parameter databases (workbooks in the EIN folder's ``einlattices``) give each magnet type's
coils and aperture, the current its designers computed for its field, and, for some, the most current
it or its supply takes. What makes a magnet's field per ampere is its turns and aperture, by the
formula its designers used:

- a quadrupole's gradient, 2 mu0 N I / r^2 (N turns a pole, r the bore radius);
- a sextupole's, 6 mu0 N I / r^3;
- a solenoid's field, mu0 N I / L (N turns over its length L);
- a dipole's field, 2 mu0 N I / g with N turns a coil around a gap g (the transfer lines' and the
  preinjector's sheets), or mu0 N I / g with N turns in all (the RCS's sheet).

Each reader returns :class:`MagnetDesign` records, one per magnet or type, with its field per ampere
from that formula (the exact mu0: the sheets round it to 1.257 or 1/796178, 3e-4 off), and its
designers' own numbers to check against. Only what a sheet states is read; a corrector whose sheet
gives no coil is left out.
"""

import math
from dataclasses import dataclass
from pathlib import Path

from .xlsx import read_xlsx

MU0 = 4e-7 * math.pi  # T m / A
GAUSS = 1e-4  # T
CM = 1e-2  # m
GEV = 1e9  # eV
C_LIGHT = 299_792_458.0


@dataclass(frozen=True)
class MagnetDesign:
    """A magnet (or a magnet type) as its designers give it, and its field per ampere."""

    name: str
    """The designers' name, e.g. LTR_QUA07, QF1_L1, Gun1."""
    kind: str
    """dipole, quadrupole, sextupole, solenoid or corrector."""
    per_ampere: float
    """Field per ampere: T/A (dipoles, solenoids), T/m/A, T/m^2/A; T m/A for an integrated one."""
    integrated: bool = False
    field: float | None = None
    """The field it is designed for (T, T/m, ...; T m if integrated), at ``energy``, as tabulated."""
    current: float | None = None
    """The current its designers give for ``field``, A."""
    maximum: float | None = None
    """The most current it takes, A: its supply's rating, or its field at the machine's top energy."""
    energy: float | None = None
    """The beam energy of ``field`` and ``current``, eV."""
    length: float | None = None
    """m"""
    note: str = ""
    """Its coil and aperture, and where the numbers come from."""


# -- the LTR-RTE transfer line database ---------------------------------------------------------


def read_transfer_line_magnets(path: str | Path, sheet: str = "magnets_feb2026") -> list[MagnetDesign]:
    """The LTR's and RTE's dipoles and quadrupoles, and the MIT Bates quadrupole's maximum.

    A row gives a magnet's length, full gap, energies, field at the highest energy, turns and the
    current there (``Imax``). Rows without turns (a septum, a kicker, correctors to be decided) are
    left out.
    """
    rows = read_xlsx(path)[sheet]
    out: list[MagnetDesign] = []
    header: list[str] = []
    for row in rows:
        cells = [c.strip() for c in row]
        if cells and cells[0] == "Number":
            header = [c.lower() for c in cells]
            continue
        if not header or len(cells) < 3 or not cells[1]:
            continue
        get = _getter(header, cells)
        turns, gap = get("turns"), get("full gap [cm]")
        if turns is None or gap is None or turns <= 0:
            continue
        name = cells[1]
        emax = get("emax [gev]")
        length = get("traj length[cm]") or get("length [cm]")
        if "b'max [g/cm]" in header:  # a quadrupole: 2 mu0 N / r^2
            r = gap / 2 * CM
            per_ampere = 2 * MU0 * turns / r**2
            field, kind = (get("b'max [g/cm]") or 0.0) * GAUSS / CM, "quadrupole"
            note = f"{turns:g} turns a pole, {gap:g} cm full gap"
        else:  # a dipole: B = mu0 N I / (g / 2)
            per_ampere = 2 * MU0 * turns / (gap * CM)
            field, kind = (get("bmax[g]") or 0.0) * GAUSS, "dipole"
            note = f"{turns:g} turns a coil, {gap:g} cm gap"
        current = get("imax [a]")
        out.append(
            MagnetDesign(
                name=name,
                kind=kind,
                per_ampere=per_ampere,
                field=field,
                current=current,
                maximum=current if name.lower().startswith("mit bates") else None,
                energy=emax * GEV if emax is not None else None,
                length=length * CM if length is not None else None,
                note=f"{note} ({Path(path).name}, {sheet})",
            )
        )
    return out


# -- the preinjector database -------------------------------------------------------------------

#: The preinjector's supplies (sheet "power supply"): its magnet types' supply models' current ratings
#: (A) and whether they are bipolar, by the words their rows describe the magnets with.
_SUPPLY_TYPES = (
    ("s-bend for gun", "gun dipole"),
    ("standard quad, linac", "linac quadrupole"),
    ("long quad, linac", "linac quadrupole"),
    ("regular gun solenoid", "gun solenoid"),
    ("spin soleonoid_weak", "spin solenoid"),
    ("spin soleonoid_strong", "strong solenoid"),
)


def read_preinjector_magnets(
    path: str | Path, sheet: str = "magnets_Jul2025_latest", supplies: str = "power supply"
) -> list[MagnetDesign]:
    """The preinjector's gun dipole, linac quadrupoles and solenoids, with their supplies' ratings.

    Quadrupoles: length, half gap (bore radius), gradient (G/cm), the current for it, turns a pole.
    Solenoids: length, field (G), turns, by coil (a bucking pair is two rows, the second unnamed:
    named here after the first, ``W1:2``). The gun dipole: its gap and the turns of its "ED" coil.
    Ratings come from the supplies' sheet, by the magnet type its rows name; the Wien filters' short
    quadrupoles and the correctors are left out (no magnet of the lattice, no coil given).
    """
    workbook = read_xlsx(path)
    ratings = _preinjector_ratings(workbook[supplies])
    rows = workbook[sheet]
    out: list[MagnetDesign] = []
    where = f"({Path(path).name}, {sheet})"
    for row in rows:
        cells = [c.strip() for c in row] + [""] * 12
        if cells[1] == "Gun bending":  # traj length, H aperture, gap, KinE, angle, ... B, ..., turns (ED)
            gap, turns, field = _number(cells[5]), _number(cells[16]), _number(cells[11])
            per_ampere = 2 * MU0 * turns / (gap * CM)
            out.append(
                MagnetDesign(
                    name="Gun bending",
                    kind="dipole",
                    per_ampere=per_ampere,
                    field=field * GAUSS,
                    current=_number(cells[15]),
                    maximum=ratings.get("gun dipole"),
                    length=_number(cells[3]) * CM,
                    note=f"{turns:g} turns a coil (ED), {gap:g} cm gap {where}",
                )
            )
        elif cells[2] == "quad" and not cells[1].startswith("Wien"):
            length, radius, gradient = _number(cells[3]), _number(cells[4]), _number(cells[5])
            turns = _number(cells[8])
            out.append(
                MagnetDesign(
                    name=cells[1],
                    kind="quadrupole",
                    per_ampere=2 * MU0 * turns / (radius * CM) ** 2,
                    field=gradient * GAUSS / CM,
                    current=_number(cells[7]),
                    maximum=ratings.get("linac quadrupole"),
                    length=length * CM,
                    note=f"{turns:g} turns a pole, {radius:g} cm bore radius {where}",
                )
            )
    out += _preinjector_solenoids(rows, ratings, where)
    return out


def _preinjector_solenoids(rows, ratings: dict[str, float], where: str) -> list[MagnetDesign]:
    out = []
    first = ""
    for row in rows:
        cells = [c.strip() for c in row] + [""] * 10
        if cells[2] != "Solenoid":
            continue
        name = f"{first}:2" if not cells[1] else cells[1]
        first = cells[1] or first
        length, field, turns = _number(cells[3]), _number(cells[5]), _number(cells[8])
        base = name.split(":")[0]
        rating = (
            ratings.get("gun solenoid")
            if base.startswith("Gun")
            else ratings.get("strong solenoid")
            if base == "T1"
            else ratings.get("spin solenoid")
            if base in ("W1", "W2", "W3", "B1", "B2")
            else None
        )
        out.append(
            MagnetDesign(
                name=name,
                kind="solenoid",
                per_ampere=MU0 * turns / (length * CM),
                field=field * GAUSS,
                maximum=rating,
                length=length * CM,
                note=f"{turns:g} turns over {length:g} cm {where}",
            )
        )
    return out


def _preinjector_ratings(rows) -> dict[str, float]:
    """Supply ratings (A) by magnet type, from the models' names, e.g. '16V-150A (... Unipolar)'."""
    ratings: dict[str, float] = {}
    for row in rows:
        cells = [c.strip() for c in row]
        if len(cells) < 11:
            continue
        described = cells[1].lower()
        for words, kind in _SUPPLY_TYPES:
            if described.startswith(words):
                model = cells[10]
                try:
                    amperes = float(model.split("-")[1].split("A")[0])
                except IndexError, ValueError:
                    continue
                ratings[kind] = max(ratings.get(kind, 0.0), amperes)
    return ratings


# -- the RCS database ---------------------------------------------------------------------------


def read_rcs_magnets(
    path: str | Path, ramp: str = "9GeV Lat", magnets: str = "Magnets", families: str = "Magnets9GeV"
) -> list[MagnetDesign]:
    """The 40-cell RCS's dipole, quadrupole, sextupole and corrector, with their maxima at 9 GeV.

    The ramp's sheet (``ramp``) gives the dipole's turns and gap, the quadrupole's bore radius and turns
    (in its current formula, 4 a pole) and the bending radius; ``families`` the largest normalized
    strengths at extraction (K1, K2) and the extraction energy; ``magnets`` the sextupole's coil (the
    0.2 m sextupole) and the corrector redesigned for 13 mT m (2025-10-10), the latest.
    """
    workbook = {title.strip(): rows for title, rows in read_xlsx(path).items()}
    where = f"({Path(path).name}"
    ramp_rows, magnet_rows, family_rows = workbook[ramp], workbook[magnets], workbook[families]
    extraction = _labelled(ramp_rows, "Ef=", ramp) * GEV
    rigidity = extraction / C_LIGHT  # T m, as the sheets take E/c
    rho = _labelled(ramp_rows, "Rho=", ramp)
    dipole_turns = _labelled(ramp_rows, "Dipole Turns=", ramp)
    gap = _labelled(ramp_rows, "Dipole Gap [m] = 2G", ramp)
    radius = _labelled(ramp_rows, "Quad Radius[m]", ramp)
    quad_turns = 4.0  # the sheet's quadrupole current: dBx/dx (r[mm])^2 / 2.52 / 4 turns
    k1 = _row_value(family_rows, "Quadrupole Long", 3)
    k2 = _row_value(family_rows, "Sextupole", 4)
    sext_radius = _column(magnet_rows, "Magnet type", "Sextupole", "Inscribed Radius")
    sext_turns = _column(magnet_rows, "Magnet type", "Sextupole", "Number of Turns")
    corrector_field = _after(magnet_rows, "13 mTm redesign parameters", "Integrated Field Strength (Tm)")
    corrector_current = _after(magnet_rows, "13 mTm redesign parameters", "Current (A)")
    dipole = MU0 * dipole_turns / gap
    quadrupole = 2 * MU0 * quad_turns / radius**2
    sext = 6 * MU0 * sext_turns / sext_radius**3
    top = {
        "dipole": rigidity / rho / dipole,
        "quadrupole": k1 * rigidity / quadrupole,
        "sextupole": k2 * rigidity / sext,
    }
    return [
        MagnetDesign(
            name="RCS dipole",
            kind="dipole",
            per_ampere=dipole,
            field=rigidity / rho,
            current=top["dipole"],
            maximum=top["dipole"],
            energy=extraction,
            note=f"{dipole_turns:g} turns in all, {gap * 100:g} cm gap; its field at "
            f"{extraction / GEV:g} GeV {where}, {ramp})",
        ),
        MagnetDesign(
            name="RCS quadrupole",
            kind="quadrupole",
            per_ampere=quadrupole,
            field=k1 * rigidity,
            maximum=top["quadrupole"],
            energy=extraction,
            note=f"{quad_turns:g} turns a pole, {radius * 100:g} cm bore radius {where}, {ramp}); "
            f"K1 {k1:g} at {extraction / GEV:g} GeV at most ({families})",
        ),
        MagnetDesign(
            name="RCS sextupole",
            kind="sextupole",
            per_ampere=sext,
            field=k2 * rigidity,
            maximum=top["sextupole"],
            energy=extraction,
            note=f"{sext_turns:g} turns a pole, {sext_radius * 100:g} cm bore radius {where}, {magnets}); "
            f"K2 {k2:g} at {extraction / GEV:g} GeV at most ({families})",
        ),
        MagnetDesign(
            name="RCS corrector",
            kind="corrector",
            per_ampere=corrector_field / corrector_current,
            integrated=True,
            field=corrector_field,
            current=corrector_current,
            maximum=corrector_current,
            note=f"{corrector_field * 1e3:g} mT m at {corrector_current:g} A, the 2025-10-10 redesign "
            f"{where}, {magnets})",
        ),
    ]


# -- reading cells ------------------------------------------------------------------------------


def _number(text: str) -> float | None:
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _getter(header: list[str], cells: list[str]):
    def get(name: str) -> float | None:
        if name not in header:
            return None
        k = header.index(name)
        return _number(cells[k]) if k < len(cells) else None

    return get


def _labelled(rows, label: str, sheet: str) -> float:
    """The number in the first cell right of the first cell reading ``label``."""
    for row in rows:
        for k, cell in enumerate(row):
            if cell.strip() == label:
                for value in row[k + 1 :]:
                    if (number := _number(value)) is not None:
                        return number
    raise ValueError(f"sheet {sheet!r}: no number right of {label!r}")


def _row_cells(rows, first: str) -> list[str]:
    for row in rows:
        if row and row[0].strip().startswith(first):
            return [c.strip() for c in row] + [""] * 12
    raise ValueError(f"no row {first!r}")


def _column(rows, header: str, first: str, title: str) -> float:
    """The number in the column whose title starts with ``title``, in the row starting with ``first``,
    below the header row starting with ``header``."""
    titles: list[str] = []
    for row in rows:
        cells = [c.strip() for c in row]
        if cells and cells[0].startswith(header):
            titles = cells
        elif titles and cells and cells[0].startswith(first):
            k = next((k for k, t in enumerate(titles) if t.startswith(title)), None)
            if k is not None and k < len(cells) and (value := _number(cells[k])) is not None:
                return value
            break
    raise ValueError(f"no {title!r} for {first!r}")


def _row_value(rows, first: str, column: int) -> float:
    value = _number(_row_cells(rows, first)[column])
    if value is None:
        raise ValueError(f"row {first!r}: no number in column {column + 1}")
    return value


def _after(rows, heading: str, label: str) -> float:
    """The number labelled ``label`` in the block below the cell starting with ``heading``."""
    seen = False
    for row in rows:
        cells = [c.strip() for c in row]
        if cells and cells[0].startswith(heading):
            seen = True
        elif seen and cells and cells[0] == label:
            for value in cells[1:]:
                if (number := _number(value)) is not None:
                    return number
    raise ValueError(f"no {label!r} under {heading!r}")
