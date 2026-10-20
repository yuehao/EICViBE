"""The EIN preinjector: from the cathode to the LTR, with its gun and spin diagnostic lines (ADR 0012).

Its designer's files are in the EIN folder's ``einlattices/Preinjector``: the engineers' survey
(``toengineer/Main_lattice.xlsx``, and ``Gun_diag.xlsx`` and ``Spin_diag.xlsx`` for the
diagnostic lines), and the PARMELA deck that tracked it (``parmela/rr6_sep.inp``), with the beam
it delivered to the LTR (``parmela/afterlinac_wk_jun13_1p7nC.dat``: 1.7 nC, 750 MeV). The survey
gives the elements, their EIC names and where they are; the deck, element for element at the
same positions, their strengths. Three branches result:

- ``prj``, the main line: the cathode and the 320 kV DC gun (an ``EGun`` from the cathode through
  the survey's first drift, where the deck tracks in the gun's field); solenoids (most in pairs of
  opposite coils on two supplies); the 16 degree gun dipole; two Wien filters (spin rotators);
  the 197 MHz buncher, the 1.3 GHz tapered capture cell and the 1.3 GHz booster; a quadrupole
  triplet; fourteen 3 m S-band (2856 MHz) travelling-wave structures with triplets between them;
  steerers and BPMs. It ends where the survey connects it to the LTR, and forks into the LTR there.
- ``prj_gun``, the gun diagnostic line: the gun dipole reversed (-16 degrees) sends the beam to a
  YAG screen and a Faraday cup. The dipole is one magnet, a shared component of both lines.
- ``prj_spin``, the spin diagnostic line: a 15 degree kicker after the capture cell (the deck's
  "reserved diagnostic bend") and a 15 degree dipole lead to the Mott polarimeter. The kicker is
  a shared component: off for the main line, a bend for this one.

Magnets are given by their fields, as designed (``Bsol``, ``Bn1``; ADR 0012): the beam's energy
along the line, from the gun's 320 keV to 750 MeV, comes from tracking it, not from the lattice.
The reference particle of all three branches is the gun's (the bends at the gun's energy take the
reference field). The gun carries its voltage, its field map's scaled by the deck, and no
frequency: DC. Cavities carry their frequency, the deck's field amplitude times their length as
voltage, and the deck's phase. Steerers are at zero: the deck's values correct the alignment
errors of its run. The Wien filters' crossed electric and magnetic fields are not represented
(drifts tagged ``WIEN_FILTER``). Names the survey does not give are chosen here.
"""

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from eicvibe.domain.lattice import (
    SPECIES,
    BranchKind,
    CavityType,
    ElementKind,
    Lattice,
    LatticeBuilder,
    derived_id,
)
from eicvibe.importers.madx import ImportNote, Severity
from eicvibe.importers.parmela import DeckElement, read_deck, read_distribution, read_field_maps
from eicvibe.importers.xlsx import read_xlsx

_FOLDER = "einlattices/Preinjector"

#: Branch names: the main line and its two diagnostic lines.
MAIN, GUN, SPIN = "prj", "prj_gun", "prj_spin"
BRANCHES = (MAIN, GUN, SPIN)
#: The shared components: the gun dipole (main and gun diagnostic traversals) and the spin kicker.
GUN_DIPOLE, SPIN_KICKER = "prj_gun_dipole", "prj_spin_kicker"
#: Device abbreviations of the preinjector's names that the naming repository does not list yet: the
#: designer's (steerers thv, Wien filters wf, cavities tbd) and ours (qc, spk, mott, yag, vlv).
DEVICES = ("thv", "wf", "tbd", "qc", "spk", "mott", "yag", "vlv")
#: Instruments that are not BPMs, by tag.
SCREEN, FARADAY_CUP, POLARIMETER = "SCREEN", "FARADAY_CUP", "POLARIMETER"

#: Survey and deck positions agree to this, m.
_SAME = 1e-4

GAUSS = 1e-4  # T
GAUSS_PER_CM = 1e-2  # T/m


@dataclass(frozen=True)
class PreinjectorSources:
    """The preinjector's design files, relative to the EIN folder."""

    deck: str = f"{_FOLDER}/parmela/rr6_sep.inp"
    survey: str = f"{_FOLDER}/toengineer/Main_lattice.xlsx"
    gun_survey: str = f"{_FOLDER}/toengineer/Gun_diag.xlsx"
    spin_survey: str = f"{_FOLDER}/toengineer/Spin_diag.xlsx"
    beam: str = f"{_FOLDER}/parmela/afterlinac_wk_jun13_1p7nC.dat"
    charge: float = 1.7e-9
    """The bunch charge the deck tracked, C."""
    gun_map_voltage: float = 350e3
    """V, of the deck's gun field map (``GUN350_22.T7``) at a scale of 1, as the deck's comment gives it
    ("-1 means 350 keV"); the deck's scale of it is the gun's voltage."""


@dataclass(frozen=True)
class Preinjector:
    """The preinjector's branches and how they were read."""

    lattice: Lattice
    notes: tuple[ImportNote, ...]
    names: dict[tuple[str, str], str]
    """EIC names, by (branch, occurrence name): the survey's, and those chosen here."""
    files: dict[str, Path]
    """Every file read, by role."""
    gun_energy: float
    """The gun's kinetic energy, eV: the reference of every preinjector branch."""


@dataclass
class _Row:
    name: str
    element: str
    s1: float
    s2: float
    phase: float = 0.0
    amp: float = 0.0
    angle: float = 0.0
    theta: float = 0.0
    comment: str = ""
    local: float = 0.0
    """Where it is in its own survey, cm (the spin kicker: from the solenoid that survey starts at)."""


@dataclass
class _Line:
    """A branch being made: its occurrences in order, as (name, definition name, EIC name)."""

    branch: str
    builder: LatticeBuilder
    names: dict[tuple[str, str], str]
    placed: list[str] = field(default_factory=list)

    def add(
        self, name: str, kind: ElementKind, *, eic: str | None = "", length: float = 0.0, **groups
    ) -> str:
        """Place a new element; ``eic`` its EIC name (by default its name; None: named elsewhere, as a
        shared magnet's other traversal is)."""
        definition = f"{self.branch}.{name}"
        tags = groups.pop("tags", ())
        self.builder.define(definition, kind, length=length, tags=tags, **groups)
        occurrence = self.builder.append(self.branch, definition, name=name)
        if eic is not None and kind is not ElementKind.BEGINNING_ELE:
            self.names[(self.branch, name)] = eic or name
        self.placed.append(name)
        return occurrence


def build_preinjector(folder: str | Path, sources: PreinjectorSources | None = None) -> Preinjector:
    """The preinjector's three branches from its survey and deck under the EIN ``folder``."""
    folder, sources = Path(folder), sources or PreinjectorSources()
    deck = read_deck(folder / sources.deck)
    notes: list[ImportNote] = []
    [bend] = [e for e in deck if e.kind == "bend"]
    gun_energy = bend.values[2] * 1e6  # the energy the gun dipole bends: PARMELA's Wr, MeV
    [gun_map] = [m for m in read_field_maps(folder / sources.deck) if m.command == "poisson"]
    gun_voltage = abs(gun_map.values[1]) * sources.gun_map_voltage  # its scale, negative for electrons
    mass = SPECIES["electron"].mass
    pc = math.sqrt(gun_energy * (gun_energy + 2 * mass))
    reference = {"species_ref": "electron", "pc_ref": pc}

    counter = iter(range(10**9))
    builder = LatticeBuilder(
        "preinjector", ids=lambda prefix: derived_id(prefix, "ein", "preinjector", str(next(counter)))
    )
    names: dict[tuple[str, str], str] = {}
    lines = {b: _Line(b, builder, names) for b in BRANCHES}
    for line in lines.values():
        builder.add_branch(line.branch, BranchKind.LINAC)
        line.add(f"{line.branch}_start", ElementKind.BEGINNING_ELE, ReferenceP=reference)

    main = _survey(folder / sources.survey)
    gun = _survey(folder / sources.gun_survey)
    spin = _spin_survey(folder / sources.spin_survey)
    kicker = _spin_kicker(main, spin)
    _main_line(lines[MAIN], main, deck, bend, kicker, gun_voltage, notes)
    _gun_line(lines[GUN], gun, bend, notes)
    _spin_line(lines[SPIN], spin, kicker, notes)

    builder.connect_extraction(MAIN, GUN, via=("prj-d144", "prj-d144"), component=GUN_DIPOLE)
    builder.connect_extraction(MAIN, SPIN, via=(kicker.name, kicker.name), component=SPIN_KICKER)
    builder.update_element("fork", name=f"{MAIN}.fork")  # its own, beside the machine's other forks
    if repeated := [n for n, count in Counter(names.values()).items() if count > 1]:
        raise ValueError(f"preinjector EIC names used twice: {', '.join(repeated)}")
    lattice = builder.build()
    notes.append(
        ImportNote(
            "reference",
            MAIN,
            f"every preinjector branch has the gun's reference particle ({gun_energy / 1e3:.2f} keV); "
            "its energy along the line comes from tracking, so its magnets are given by their fields",
        )
    )
    files = {
        "preinjector_deck": folder / sources.deck,
        "preinjector_survey": folder / sources.survey,
        "preinjector_gun_survey": folder / sources.gun_survey,
        "preinjector_spin_survey": folder / sources.spin_survey,
        "preinjector_beam": folder / sources.beam,
    }
    return Preinjector(lattice, tuple(notes), names, {k: v.resolve() for k, v in files.items()}, gun_energy)


def preinjector_beam(
    folder: str | Path, reference_pc: float, sources: PreinjectorSources | None = None
) -> np.ndarray:
    """The beam the preinjector delivers to the LTR, as particles about the LTR's ``reference_pc`` (eV)."""
    sources = sources or PreinjectorSources()
    return read_distribution(Path(folder) / sources.beam, reference_pc=reference_pc)


# -- the main line ------------------------------------------------------------------------------


def _main_line(
    line: _Line,
    rows: list[_Row],
    deck: list[DeckElement],
    bend: DeckElement,
    kicker: _Row,
    gun_voltage: float,
    notes: list,
) -> None:
    by_start: dict[str, list[DeckElement]] = {}
    for element in deck:
        by_start.setdefault(element.kind, []).append(element)
    done: set[str] = set()
    pending_fringe: _Row | None = None
    gun_next = False
    for k, row in enumerate(rows):
        kind = row.element
        length = _metres(row.s2 - row.s1)
        if kind == "drift":
            if length <= 0:
                continue
            if gun_next:  # the survey's first drift, from the cathode: the deck tracks in the gun's field
                name = row.name.replace("prj-dt", "prj-gun")
                line.add(name, ElementKind.EGUN, length=length, RFP={"voltage": gun_voltage})
                notes.append(
                    ImportNote("renamed", row.name, f"the survey's drift from the cathode is the gun {name}")
                )
                gun_next = False
            elif row.s1 < kicker.s1 < row.s2:  # the spin kicker sits in this drift
                _split_for_kicker(line, row, kicker)
            else:
                line.add(row.name, ElementKind.DRIFT, length=length)
            if "connect to LTR" in row.comment:
                return
        elif kind == "cathode":
            line.add(row.name, ElementKind.MARKER, tags=("CATHODE",))
            gun_next = True
        elif kind == "steerer":
            if (row.name, row.s1) in done:
                continue  # a two-plane steerer is listed once per plane
            done.add((row.name, row.s1))
            line.add(row.name, ElementKind.KICKER)
        elif kind == "bpm":
            line.add(row.name, ElementKind.INSTRUMENT)
        elif kind == "wien":  # the survey's name has no device letters, which an EIC name needs
            name = row.name.replace("prj-", "prj-wf")
            line.add(name, ElementKind.DRIFT, length=length, tags=("WIEN_FILTER",))
            notes.append(ImportNote("renamed", row.name, f"the survey's Wien filter is {name} here"))
        elif kind == "solenoid":
            coils = [
                e for e in by_start["solenoid"] if row.s1 / 100 - _SAME <= e.start < row.s2 / 100 - _SAME
            ]
            if len(coils) == 1:
                line.add(
                    row.name,
                    ElementKind.SOLENOID,
                    length=length,
                    SolenoidP={"Bsol": coils[0].values[2] * GAUSS},
                )
            else:  # two coils of opposite field, each on its own supply: append numbers 1 and 2
                for k, (coil, suffix) in enumerate(zip(coils, "ab", strict=True), start=1):
                    line.add(
                        f"{row.name}{suffix}",
                        ElementKind.SOLENOID,
                        eic=f"{row.name}:{k}",
                        length=round(coil.length, 7),
                        SolenoidP={"Bsol": coil.values[2] * GAUSS},
                    )
        elif kind == "bend":
            line.add(row.name, ElementKind.BEND, length=length, BendP=_gun_dipole(row, bend, main=True))
        elif kind == "quad" and length == 0.0:
            name = row.name.replace("prj-", "prj-qc")
            line.add(name, ElementKind.MULTIPOLE, tags=("QUAD_CORRECTOR",))
            notes.append(ImportNote("renamed", row.name, f"the survey's quadrupole corrector is {name} here"))
        elif kind == "quad":
            [quad] = [e for e in by_start["quad"] if abs(e.start - row.s1 / 100) < _SAME]
            # PARMELA's gradient focuses the tracked electrons horizontally when positive: their
            # field gradient is the opposite (the deck's comments give QF gradients as negative).
            line.add(
                row.name,
                ElementKind.QUADRUPOLE,
                length=length,
                MagneticMultipoleP={"Bn1": -quad.values[2] * GAUSS_PER_CM},
            )
        elif kind == "cell":
            if k + 1 < len(rows) and rows[k + 1].element == "trwave":
                pending_fringe = row  # the input coupler's field, part of the structure that follows
                continue
            [cell] = [e for e in by_start["cell"] if abs(e.start - row.s1 / 100) < _SAME]
            line.add(
                row.name, ElementKind.RFCAVITY, length=length,
                RFP={"frequency": cell.values[-1] * 1e6, "voltage": cell.values[3] * 1e6 * length,
                     "phase": _phase(cell.values[2]), "cavity_type": CavityType.STANDING_WAVE},
            )  # fmt: skip
        elif kind == "trwave":
            start = pending_fringe.s1 if pending_fringe else row.s1
            cells = [e for e in by_start["trwave"] if row.s1 / 100 - _SAME <= e.start < row.s2 / 100 - _SAME]
            first = cells[0]
            line.add(
                row.name, ElementKind.RFCAVITY, length=_metres(row.s2 - start),
                RFP={"frequency": first.values[6] * 1e6, "voltage": first.values[3] * 1e6 * length,
                     "phase": _phase(first.values[2]), "cavity_type": CavityType.TRAVELING_WAVE,
                     "L_active": length},
            )  # fmt: skip
            pending_fringe = None
        else:
            notes.append(
                ImportNote(
                    "skipped", row.name, f"a survey {row.element!r} is not represented", Severity.WARNING
                )
            )
    raise ValueError("the survey never connects the preinjector to the LTR")


def _gun_dipole(row: _Row, bend: DeckElement, *, main: bool) -> dict:
    """The gun dipole for the main line (bending to +x: a negative PALS angle) or, reversed, for
    the diagnostic line. Its iron is the same either way, so are its pole faces (e1, e2 of the main
    line), which the floor geometry matches where the two lines meet."""
    angle = math.radians(bend.values[3])
    faces = -math.radians(bend.values[4]), -math.radians(bend.values[5])
    return {"angle_ref": -angle if main else angle, "e1": faces[0], "e2": faces[1]}


def _split_for_kicker(line: _Line, row: _Row, kicker: _Row) -> None:
    """The drift the spin kicker sits in, as drift, kicker (off: the main beam passes), drift."""
    line.add(f"prj-dt{round(kicker.s1)}", ElementKind.DRIFT, length=_metres(kicker.s1 - row.s1))
    line.add(kicker.name, ElementKind.KICKER, length=_metres(kicker.s2 - kicker.s1), tags=("PULSED",))
    line.add(row.name, ElementKind.DRIFT, length=_metres(row.s2 - kicker.s2))


# -- the diagnostic lines -----------------------------------------------------------------------


def _gun_line(line: _Line, rows: list[_Row], bend: DeckElement, notes: list) -> None:
    """From the gun dipole, reversed, to the Faraday cup; the rows before it are the main line's."""
    start = next(k for k, r in enumerate(rows) if r.element == "bend")
    seen: set[tuple[str, float]] = set()
    for row in rows[start:]:
        kind, length = row.element, _metres(row.s2 - row.s1)
        eic = row.name if row.name not in _MAIN_NAMES_AFTER_DIPOLE else f"{row.name}:2"
        if kind == "bend":  # the gun dipole, named at its main-line traversal
            line.add(
                row.name, ElementKind.BEND, eic=None, length=length, BendP=_gun_dipole(row, bend, main=False)
            )
        elif kind == "drift" and length > 0:
            line.add(row.name, ElementKind.DRIFT, length=length)
        elif kind == "steerer":
            if (row.name, row.s1) in seen:
                continue
            seen.add((row.name, row.s1))
            line.add(row.name, ElementKind.KICKER, eic=eic)
        elif kind == "quad":
            line.add(row.name, ElementKind.MULTIPOLE, tags=("QUAD_CORRECTOR",))
        elif kind == "bpm":
            line.add(row.name, ElementKind.INSTRUMENT)
        elif kind == "solenoid":
            # The gun line's own solenoid: the survey's requirements give it 173 G.
            line.add(row.name, ElementKind.SOLENOID, length=length, SolenoidP={"Bsol": 173 * GAUSS})
        elif kind == "yag":
            line.add(row.name.replace("prj-", "prj-yag"), ElementKind.INSTRUMENT, tags=(SCREEN,))
        elif kind == "valve":
            line.add(
                f"{row.name}-valve",
                ElementKind.MARKER,
                eic=f"prj-vlv{row.name.removeprefix('prj-')}",
                tags=("VALVE",),
            )
        elif kind == "faraday cup":
            line.add(
                f"{row.name}-fc",
                ElementKind.INSTRUMENT,
                eic=f"prj-fc{row.name.removeprefix('prj-')}",
                tags=(FARADAY_CUP,),
            )
    notes.append(
        ImportNote("names", GUN, "the gun line's steerer after the dipole is prj-thv144:2, its YAG screen "
                   "prj-yag274, its valve prj-vlv284 and its Faraday cup prj-fc284: the survey names them "
                   "like main-line elements, or without device letters")
    )  # fmt: skip


#: Main-line names the gun diagnostic survey uses again for its own elements after the dipole.
_MAIN_NAMES_AFTER_DIPOLE = {"prj-thv144"}


def _spin_line(line: _Line, rows: list[_Row], kicker: _Row, notes: list) -> None:
    """From the spin kicker, on (15 degrees), to the Mott polarimeter. Names are chosen here: the
    survey gives none; each is the main line's position (cm) where it is."""
    origin = kicker.s1 - kicker.local
    start = next(k for k, r in enumerate(rows) if r.element == "kicker")
    for row in rows[start:]:
        kind, length = row.element, _metres(row.s2 - row.s1)
        at = round(origin + row.s2)
        if kind == "kicker":  # named at its main-line traversal
            line.add(
                kicker.name,
                ElementKind.BEND,
                eic=None,
                length=length,
                BendP={"angle_ref": -math.radians(row.angle)},
            )
        elif kind == "dipole":
            line.add(
                f"prj-d{at}", ElementKind.BEND, length=length, BendP={"angle_ref": -math.radians(row.angle)}
            )
        elif kind == "drift" and length > 0:
            line.add(f"prj-dt{at}", ElementKind.DRIFT, length=length)
        elif kind == "steer":
            line.add(f"prj-thv{at}", ElementKind.KICKER)
        elif kind == "quad":
            line.add(f"prj-q{at}", ElementKind.QUADRUPOLE, length=length)
        elif kind == "mott":
            line.add(f"prj-mott{at}", ElementKind.INSTRUMENT, tags=(POLARIMETER,))
    notes.append(
        ImportNote("names", SPIN, "the spin line's elements are unnamed in its survey: their names here "
                   "are temporary, from the main line's position (cm)", Severity.WARNING)
    )  # fmt: skip
    notes.append(
        ImportNote("energy", SPIN, "the spin line's energy is not in the design files: its bends take the "
                   "gun's reference field (the capture cell before it runs near -90 degrees, bunching)",
                   Severity.WARNING)
    )  # fmt: skip


def _spin_kicker(main: list[_Row], spin: list[_Row]) -> _Row:
    """Where the spin kicker is on the main line: the spin survey starts at a main-line solenoid
    (the one after the capture cell, where the deck reserves a diagnostic bend)."""
    [kick] = [r for r in spin if r.element == "kicker"]
    solenoids = [r for r in main if r.element == "solenoid"]
    capture = next(r for r in main if r.element == "cell" and 99 <= r.s2 - r.s1 <= 101)  # the 1.3 GHz taper
    home = next(r for r in solenoids if r.s1 > capture.s2)
    s1, s2 = home.s1 + kick.s1, home.s1 + kick.s2
    return _Row(f"prj-spk{round(s2)}", "kicker", s1, s2, angle=kick.angle, local=kick.s1)


# -- the survey sheets ----------------------------------------------------------------------------


def _survey(path: Path) -> list[_Row]:
    """Rows of an engineers' survey sheet (n, naming, s1 cm, element, s2 cm, ds, phase, amp, angle, X,
    Y, Z, theta, comment)."""
    [sheet, *_] = read_xlsx(path).values()
    rows = []
    for cells in sheet[1:]:
        if len(cells) < 5 or not cells[1]:
            continue
        rows.append(
            _Row(
                name=cells[1].strip(),
                element=cells[3].strip().lower(),
                s1=float(cells[2]),
                s2=float(cells[4]),
                phase=_float(cells, 6),
                amp=_float(cells, 7),
                angle=_float(cells, 8),
                theta=_float(cells, 12),
                comment=cells[13] if len(cells) > 13 else "",
            )
        )
    return rows


def _spin_survey(path: Path) -> list[_Row]:
    """Rows of the spin diagnostic survey (s1, element, s2, length, X, Y, Z, theta; cm and degrees),
    its bends' angles from the change of theta."""
    [sheet, *_] = read_xlsx(path).values()
    rows, theta = [], 0.0
    for cells in sheet[1:]:
        if len(cells) < 3 or not cells[1]:
            continue
        new_theta = _float(cells, 7)
        rows.append(
            _Row(
                "",
                cells[1].strip().lower(),
                float(cells[0]),
                float(cells[2]),
                angle=new_theta - theta,
                theta=new_theta,
            )
        )
        theta = new_theta
    return rows


def _phase(degrees: float) -> float:
    """A deck's RF phase in rad, within (-pi, pi]: the deck adds whole turns where its optimizer went."""
    return math.remainder(math.radians(degrees), 2 * math.pi)


def _metres(cm: float) -> float:
    """A survey length (cm) in metres, to the survey's precision (0.1 µm)."""
    return round(cm / 100.0, 7)


def _float(cells: Sequence[str], k: int) -> float:
    try:
        return float(cells[k]) if k < len(cells) and cells[k] != "" else 0.0
    except ValueError:
        return 0.0
