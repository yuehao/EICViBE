"""The EIN's magnets: field per ampere and supplies' limits, from its parameter databases (ADR 0014).

The machine folder's supplies started with placeholder transfer functions (0.1 T/m per ampere for
every quadrupole) and limits of twice their design currents. The EIC parameter databases in the EIN
folder give the magnets' coils and apertures (``eicvibe.importers.magnet_tables``); this module turns
them into transfer functions (source ``design``) and limits, supply by supply:

- LTR: its quadrupoles are MIT Bates quadrupoles (64 turns a pole, 6.5 cm gap), up to the most the
  MIT Bates quadrupole takes (K1 6.75 at 0.75 GeV: 110.9 A); its dipoles 01 to 05 (32 turns a coil,
  4 cm gap) by the rule below.
- RTE: its quadrupoles QFH and QDH (64 turns, 4 cm gap) up to their current at 18 GeV, the line's top
  energy; its MIT Bates quadrupoles as the LTR's.
- RCS: its dipoles (6 turns, 4 cm gap), quadrupoles (4 turns a pole, 2 cm bore), sextupoles (4 turns,
  2 cm) and correctors (the 13 mT m redesign), up to what its designers ask of them at 9 GeV.
- Preinjector: its 27 quadrupoles (10 turns; 5 or 2 cm bore), 16 solenoid coils and gun dipole, up to
  their supplies' ratings (its supplies' sheet).

Where a database states no maximum, the limits are twice the design current, as before (or the
designers' current where the lattice's is zero). A supply keeps its sign: unipolar (its polarity says
how it is wired) or bipolar. Magnets the databases give no coil for (correctors of the lines and the
preinjector, septa, kickers, the BAR) keep their placeholders.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from eicvibe.domain.lattice import ElementKind, Lattice, PlacedElement
from eicvibe.domain.machine.power import (
    CalibrationSource,
    CircuitMagnet,
    PolynomialTransfer,
    design_value,
    magnet_current,
)
from eicvibe.engines.convert import reference
from eicvibe.importers.magnet_tables import (
    MagnetDesign,
    read_preinjector_magnets,
    read_rcs_magnets,
    read_transfer_line_magnets,
)
from eicvibe.store import Machine, load_machine
from eicvibe.store.device_files import set_supplies

#: The parameter databases, relative to the EIN folder.
DATABASES = {
    "preinjector": "einlattices/01_PreInjector Parameter Database.xlsx",
    "rcs": "einlattices/03_RCS Parameter Database.xlsx",
    "lines": "einlattices/04_LTR-RTE Transfer Line Parameter Database.xlsx",
}

_UNITS = {"dipole": "T", "solenoid": "T", "quadrupole": "T/m", "sextupole": "T/m^2", "corrector": "T m"}
_WIDE = (-1e12, 1e12)
_RATINGS = {50.0: "30 V, 50 A", 76.0: "20 V, 76 A", 150.0: "100 V, 150 A"}
#: The preinjector's magnets are its magnet table's rows in order; a row's field agrees with the
#: deck's to this (the table rounds some, 1000 G for the deck's 1001.0).
_SAME = 1e-2

#: The designers' current for a magnet at the branch's energy, A (to check ours against).
Theirs = Callable[[Lattice, PlacedElement], float]


@dataclass(frozen=True)
class Assignment:
    """A supply circuit's transfer function and limits, and where the limits come from."""

    supply: str
    plane: str | None
    transfer: str
    limits: tuple[float, float]
    why: str


@dataclass(frozen=True)
class Check:
    """A magnet's design current, ours (the lattice's value through its transfer function) against its
    designers' (their table's, scaled to our energy)."""

    magnet: str
    ours: float
    theirs: float
    note: str = ""

    @property
    def difference(self) -> float:
        return abs(self.ours - self.theirs) / max(abs(self.theirs), 1e-12)


@dataclass
class EinMagnets:
    transfers: dict[str, tuple[PolynomialTransfer, str]] = field(default_factory=dict)
    """Named transfer functions, with a note each (the coil, the aperture and the source)."""
    assignments: list[Assignment] = field(default_factory=list)
    checks: list[Check] = field(default_factory=list)
    left: list[str] = field(default_factory=list)
    """Supplies of the databases' areas that keep their placeholders, by key."""


@dataclass(frozen=True)
class _Choice:
    transfer: str
    design: MagnetDesign
    maximum: float | None
    why: str
    theirs: Theirs | None = None


def ein_magnets(machine: Machine, folder: str | Path) -> EinMagnets:
    """The transfer functions and limits the EIN folder's databases give the machine's supplies."""
    chooser = _Chooser(machine.lattice, Path(folder))
    plan = EinMagnets()
    lattice = machine.lattice
    for supply in machine.description.supplies:
        for circuit in supply.circuits:
            placed = lattice.placement(circuit.magnets[0].occurrences[0])
            branch = lattice.branch(placed.branch_id).name
            choice = chooser.choose(branch, placed)
            if choice is None:
                if chooser.covers(branch) and supply.key not in plan.left:
                    plan.left.append(supply.key)
                continue
            if choice.transfer not in plan.transfers:
                plan.transfers[choice.transfer] = (_transfer(choice.design), choice.design.note)
            transfer = plan.transfers[choice.transfer][0]
            ours = max(_current(lattice, m, transfer) for m in circuit.magnets)
            theirs = choice.theirs(lattice, placed) if choice.theirs else None
            if theirs is not None:
                plan.checks.append(Check(f"{branch}/{placed.name}", ours, theirs, choice.design.name))
            maximum, why = choice.maximum, choice.why
            if maximum is None:
                maximum = 2.0 * max(ours, theirs or 0.0)
                why = "twice its design current" if ours else "twice its designers' current"
            top = _round_up(maximum)
            limits = (-top if circuit.limits[0] < 0 else 0.0, top)
            plan.assignments.append(Assignment(supply.key, circuit.plane, choice.transfer, limits, why))
    return plan


def apply_ein_magnets(machine: str | Path, folder: str | Path) -> tuple[Path, EinMagnets]:
    """Give an EIN machine folder's supplies the databases' transfer functions and limits, in its power
    file, in place (nothing else changes). Returns the file and the plan it followed."""
    plan = ein_magnets(load_machine(Path(machine)), folder)
    circuits = {(a.supply, a.plane): (a.transfer, a.limits) for a in plan.assignments}
    return set_supplies(Path(machine), plan.transfers, circuits), plan


# -- choosing a magnet's design -----------------------------------------------------------------


class _Chooser:
    """The databases found under an EIN folder, and which of their designs a magnet is."""

    def __init__(self, lattice: Lattice, folder: Path) -> None:
        found = {key: folder / path for key, path in DATABASES.items() if (folder / path).is_file()}
        self.lines = read_transfer_line_magnets(found["lines"]) if "lines" in found else []
        self.pre = read_preinjector_magnets(found["preinjector"]) if "preinjector" in found else []
        self.rcs = {m.name: m for m in read_rcs_magnets(found["rcs"])} if "rcs" in found else {}
        self.by_name = {m.name.upper(): m for m in self.lines}
        self.bates = [m for m in self.lines if m.name.lower().startswith("mit bates")]
        self.bates_max = max((m.maximum or 0.0 for m in self.bates), default=None)
        # The preinjector's quadrupoles and main-line solenoid coils, in beam order, are the magnet
        # table's rows in order (checked by their fields, which the deck took from it).
        main = lattice.layout("prj").placements if self.pre else []
        self.pre_quads = dict(
            zip(
                [p.id for p in main if p.kind is ElementKind.QUADRUPOLE],
                [m for m in self.pre if m.kind == "quadrupole"],
                strict=False,
            )
        )
        self.pre_solenoids = dict(
            zip(
                [p.id for p in main if p.kind is ElementKind.SOLENOID],
                [m for m in self.pre if m.kind == "solenoid" and m.maximum is not None],
                strict=False,
            )
        )

    def covers(self, branch: str) -> bool:
        """Whether a branch's database was read, so that its magnets ought to be found."""
        return (
            (branch in ("ltr", "rte") and bool(self.lines))
            or (branch == "rcs" and bool(self.rcs))
            or (branch.startswith("prj") and bool(self.pre))
        )

    def choose(self, branch: str, placed: PlacedElement) -> _Choice | None:
        if branch in ("ltr", "rte") and self.lines:
            return self._line(branch, placed)
        if branch == "rcs" and self.rcs:
            return self._rcs(placed)
        if branch == "prj" and self.pre:
            return self._preinjector(placed)
        return None

    def _line(self, branch: str, placed: PlacedElement) -> _Choice | None:
        kind, name = placed.kind, placed.name
        if kind is ElementKind.QUADRUPOLE and (branch == "ltr" or "bates" in name) and self.bates:
            rows = ("QFRAMPH" if name.startswith("qf") else "QDRAMPH",)
            design = self.by_name.get(name.upper()) or next(
                (self.by_name[r] for r in rows if r in self.by_name), self.bates[0]
            )
            why = "the most the MIT Bates quadrupole takes (K1 6.75 at 0.75 GeV)"
            return _Choice("mit-bates-quadrupole", design, self.bates_max, why, _at_our_energy(design))
        design = self.by_name.get(name.upper())
        if design is None:
            return None
        if kind is ElementKind.QUADRUPOLE:
            why = "its current at 18 GeV, the line's top energy"
            return _Choice("rte-quadrupole", design, design.current, why, _at_our_energy(design))
        if kind is ElementKind.BEND:
            return _Choice("ltr-dipole", design, None, "", _at_our_energy(design))
        return None

    def _rcs(self, placed: PlacedElement) -> _Choice | None:
        rcs = self.rcs
        if placed.kind is ElementKind.BEND:
            design = rcs["RCS dipole"]
            # The table's dipole is the main one (rho 46.24 m); the arc ends' bend less on the same current.
            theirs = _at_our_energy(design) if placed.definition.name == "rcs.b0" else None
            return _Choice("rcs-dipole", design, design.maximum, "its field at 9 GeV", theirs)
        if placed.kind is ElementKind.QUADRUPOLE:
            why = "K1 0.662 at 9 GeV, the most asked of it"
            return _Choice("rcs-quadrupole", rcs["RCS quadrupole"], rcs["RCS quadrupole"].maximum, why)
        if placed.kind is ElementKind.SEXTUPOLE:
            why = "K2 10 at 9 GeV, the most asked of it"
            return _Choice("rcs-sextupole", rcs["RCS sextupole"], rcs["RCS sextupole"].maximum, why)
        if placed.kind is ElementKind.KICKER and placed.definition.name == "rcs.cor":
            why = "its 13 mT m at 25.2 A"
            return _Choice("rcs-corrector", rcs["RCS corrector"], rcs["RCS corrector"].maximum, why)
        return None

    def _preinjector(self, placed: PlacedElement) -> _Choice | None:
        element = placed.element
        if placed.id in self.pre_quads:
            design = self.pre_quads[placed.id]
            gradient = element.MagneticMultipoleP.Bn1 if element.MagneticMultipoleP else 0.0
            if math.isclose(abs(gradient), abs(design.field), rel_tol=_SAME):
                bore = "5cm" if design.per_ampere < 0.03 else "2cm"
                return _Choice(
                    f"preinjector-quadrupole-{bore}",
                    design,
                    design.maximum,
                    "its supply's rating (16 V, 150 A)",
                    lambda _lattice, _placed, design=design: abs(design.current),
                )
        if placed.id in self.pre_solenoids:
            design = self.pre_solenoids[placed.id]
            if math.isclose(abs(element.SolenoidP.Bsol), abs(design.field), rel_tol=_SAME):
                why = f"its supply's rating ({_RATINGS.get(design.maximum, '')})"
                return _Choice(
                    f"preinjector-solenoid-{round(design.length * 100)}cm", design, design.maximum, why
                )
        if placed.kind is ElementKind.BEND and placed.name == "prj-d144":
            [design] = [m for m in self.pre if m.kind == "dipole"]
            return _Choice(
                "preinjector-gun-dipole",
                design,
                design.maximum,
                "its supply's rating (30 V, 25 A)",
                lambda _lattice, _placed, design=design: abs(design.current),
            )
        return None


def _at_our_energy(design: MagnetDesign) -> Theirs | None:
    """The designers' current, scaled from their energy to the branch's reference momentum."""
    if design.current is None or not design.energy:
        return None

    def at(lattice: Lattice, placed: PlacedElement) -> float:
        return abs(design.current) * reference("ein", lattice, placed.branch_id)["pc"] / design.energy

    return at


def _transfer(design: MagnetDesign) -> PolynomialTransfer:
    return PolynomialTransfer(
        coefficients=(0.0, design.per_ampere),
        field_unit=_UNITS[design.kind],
        integrated=design.integrated,
        source=CalibrationSource.DESIGN,
    )


def _current(lattice: Lattice, magnet: CircuitMagnet, transfer: PolynomialTransfer) -> float:
    """A magnet's design current through ``transfer`` (A, its size)."""
    with_design = CircuitMagnet(
        occurrences=magnet.occurrences,
        parameter=magnet.parameter,
        transfer=transfer,
        polarity=magnet.polarity,
    )
    value = design_value(lattice.placement(magnet.occurrences[0]), magnet.parameter)
    return abs(magnet_current(lattice, with_design, value, _WIDE))


def _round_up(value: float, digits: int = 3) -> float:
    """``value`` rounded up to ``digits`` significant digits, for limits people read."""
    if value <= 0:
        return 0.0
    scale = 10.0 ** (math.floor(math.log10(value)) - digits + 1)
    return round(math.ceil(value / scale - 1e-9) * scale, 12)
