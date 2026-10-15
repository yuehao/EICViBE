"""A starting description of a machine's devices, generated from its lattice (ADR 0003).

The virtual accelerator needs devices before the real device list exists.
:func:`describe_machine` applies rules to a lattice and returns a
:class:`~.description.MachineDescription`, which is what a machine's naming
and device files hold; people then edit those files (:mod:`eicvibe.store`).
:func:`generate_machine_registry` assembles a registry straight from the rules.

The rules, with EIC names (EIC-SEG-RSI-009 Rev 02, :mod:`.naming`) for every
element that has a device abbreviation, drifts and markers included, as in
the ESR naming table (``ltr-q6``, ``ltr-dt5``):

- every bend, quadrupole, sextupole and octupole is on a power circuit
  (:mod:`.power`) whose supply has ``:sp`` and ``:rb``, in A. A magnet
  powered on its own has the supply ``<name>-ps`` (as the standard's example
  "es6-qf50_2-ps, the supply powering the second focusing quadrupole"). The
  kinds in ``series`` are wired in strings instead: magnets of one kind in a
  branch with the same length and the same design strength (within
  :data:`SERIES_LENGTH_TOLERANCE` and :data:`SERIES_STRENGTH_TOLERANCE`),
  whatever their lattice definitions, most likely share a supply. It is
  named after the string's first magnet. A magnet at zero strength has no
  family current to share and is powered on its own;
- every corrector has its own supply, with one circuit per plane; a
  two-plane corrector has ``:sp:h``, ``:sp:v`` and their readbacks;
- every solenoid has its own supply;
- a branch whose magnets are given by their fields (``Bn1``, ``Bsol``: an
  injector line, ADR 0012) has supplies that set fields, also for its
  magnets at zero; others set normalized strengths;
- every BPM (an Instrument not tagged as another instrument: a screen, a
  Faraday cup, a polarimeter) reads ``<name>:rdx`` and ``<name>:rdy``, in mm;
- every RF cavity has ``<name>:sp:volt`` (MV), ``<name>:sp:phase`` (deg)
  and their readbacks;
- a periodic branch has a tune monitor, ``<machine>-btun-rdx`` and ``-rdy``;
- a magnet that several branches pass through (a shared component, given
  in ``shared``) is named once and has one supply, which sets the bends at
  its ``driven`` traversals and nothing else: a septum's field is on for the
  injected and extracted beams and pulsed off for the stored beam.

Each magnet's transfer function is a linear placeholder of typical
magnitude, per length (integrated only for a thin magnet), and its polarity
that of its design value, so supplies run at positive currents; replace
them, magnet by magnet, with measured curves or a transfer-function service.
Ids derive from the description's keys and the lattice's ids, never from
names, so renaming areas or tokens keeps every device's identity.
"""

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from ..lattice import ElementKind, Lattice, has_fields, strength_parameter
from .description import (
    Bpm,
    MachineDescription,
    Naming,
    RfStation,
    SharedName,
    Supply,
    SupplyCircuit,
    TuneMonitor,
    assemble_registry,
    default_area,
)
from .naming import NamingArea, NamingScheme
from .power import CircuitMagnet, PolynomialTransfer, design_value, is_field, rigidity
from .registry import MachineRegistry

#: Placeholder field per ampere: T/m for a quadrupole, T/m^2 for a sextupole, T/m^3 for an octupole.
PLACEHOLDER_GRADIENT_PER_AMPERE = {1: 0.1, 2: 1.0, 3: 10.0}
#: Placeholder solenoid field per ampere, T.
PLACEHOLDER_SOLENOID_PER_AMPERE = 1e-3
#: Instruments that are not BPMs, by tag: they get no position channels.
NOT_BPMS = frozenset({"SCREEN", "FARADAY_CUP", "POLARIMETER"})
#: Placeholder integrated corrector field per ampere, T m.
PLACEHOLDER_CORRECTOR_PER_AMPERE = 1e-3
#: Placeholder dipole field per ampere, T.
PLACEHOLDER_DIPOLE_PER_AMPERE = 1e-3
#: Placeholder current ranges, A: magnets run to twice their design current or at least this.
PLACEHOLDER_MAGNET_CURRENT = 100.0
PLACEHOLDER_CORRECTOR_CURRENT = 50.0

#: Magnets of one kind whose lengths (m) and design strengths (relative) agree within these are
#: taken to be one family, wired in series where their kind is.
SERIES_LENGTH_TOLERANCE = 1e-6
SERIES_STRENGTH_TOLERANCE = 1e-6

#: Magnets on power circuits, by the multipole order of their main field.
_ORDER = {ElementKind.BEND: 0, ElementKind.QUADRUPOLE: 1, ElementKind.SEXTUPOLE: 2, ElementKind.OCTUPOLE: 3}


@dataclass(frozen=True)
class SharedMagnet:
    """A magnet that several branches pass through, such as a septum.

    It is named at the ``home`` traversal (with ``abbreviation``), and its
    supply sets the field of the bends at the ``driven`` traversals only.
    """

    home: str
    driven: tuple[str, ...]
    abbreviation: str = "sptm"


def placeholder_transfer(order: int, *, thin: bool = False) -> PolynomialTransfer:
    """The placeholder transfer function of a magnet whose main field has ``order``."""
    per_ampere = PLACEHOLDER_DIPOLE_PER_AMPERE if order == 0 else PLACEHOLDER_GRADIENT_PER_AMPERE[order]
    power = order - 1 if thin else order
    unit = {-1: "T m", 0: "T", 1: "T/m"}.get(power, f"T/m^{power}")
    return PolynomialTransfer(coefficients=(0.0, per_ampere), field_unit=unit, integrated=thin)


def corrector_transfer() -> PolynomialTransfer:
    return PolynomialTransfer(
        coefficients=(0.0, PLACEHOLDER_CORRECTOR_PER_AMPERE), field_unit="T m", integrated=True
    )


def describe_machine(
    lattice: Lattice,
    branches: Mapping[str, Iterable[NamingArea] | None] | Iterable[str] | None = None,
    *,
    scheme: NamingScheme | None = None,
    series: Mapping[str, Iterable[ElementKind]] | None = None,
    shared: Mapping[str, SharedMagnet] | None = None,
    names: Mapping[str, str] | None = None,
) -> MachineDescription:
    """Naming rules and devices of the given branches (all by default), by the rules above.

    ``branches`` maps branch names to their naming areas (None: one area
    named after the branch), or lists branch names. ``series`` maps branch
    names to the magnet kinds wired there in strings. ``shared`` maps shared
    component names to how their one magnet is named and powered. ``names``
    gives elements EIC names of their own, by occurrence id, in place of the
    rules' (names a designer gave).
    """
    if branches is None:
        branches = [b.name for b in lattice.branches]
    if not isinstance(branches, Mapping):
        branches = dict.fromkeys(branches)
    shared = dict(shared or {})
    components = {c.name: c for c in lattice.shared_components}
    naming = Naming.from_scheme(
        scheme or NamingScheme(),
        # Written out even where they are the default, so the naming file shows every rule.
        areas={
            lattice.layout(n).id: tuple(a) if a is not None else (default_area(lattice, n),)
            for n, a in branches.items()
        },
        shared=tuple(
            SharedName(component=_component(components, name).id, home=m.home, abbreviation=m.abbreviation)
            for name, m in shared.items()
        ),
        names=dict(names or {}),
    )
    kinds = {lattice.layout(n).id: frozenset(k) for n, k in (series or {}).items()}
    supplies: list[Supply] = []
    bpms: list[Bpm] = []
    rf: list[RfStation] = []
    tunes: list[TuneMonitor] = []
    for name in branches:
        layout = lattice.layout(name)
        _describe_branch(lattice, layout, kinds.get(layout.id, frozenset()), supplies, bpms, rf)
        if layout.periodic:
            tunes.append(TuneMonitor(branch=layout.id))
    for name, magnet in shared.items():
        supplies.append(_shared_supply(lattice, _component(components, name), magnet))
    return MachineDescription(
        lattice=lattice.name,
        naming=naming,
        supplies=tuple(supplies),
        bpms=tuple(bpms),
        rf=tuple(rf),
        tune_monitors=tuple(tunes),
    )


def generate_machine_registry(
    lattice: Lattice,
    branches: Mapping[str, Iterable[NamingArea] | None] | Iterable[str] | None = None,
    *,
    scheme: NamingScheme | None = None,
    series: Mapping[str, Iterable[ElementKind]] | None = None,
    shared: Mapping[str, SharedMagnet] | None = None,
) -> MachineRegistry:
    """The registry of :func:`describe_machine`'s description."""
    description = describe_machine(lattice, branches, scheme=scheme, series=series, shared=shared)
    return assemble_registry(lattice, description)


def generate_registry(
    lattice: Lattice,
    branch: str | None = None,
    *,
    areas: Iterable[NamingArea] | None = None,
    scheme: NamingScheme | None = None,
    series: Iterable[ElementKind] = (),
) -> MachineRegistry:
    """The registry of one branch's devices (the root branch by default).

    ``areas`` assign EIC area prefixes and clock positions along the branch,
    e.g. :func:`~eicvibe.domain.machine.naming.ring_sectors` for the ESR. By
    default the whole branch is one area named after the branch, with
    positions in metres from its start. The branch's BeginningEle must carry
    the reference particle, which scales the placeholder ranges.
    """
    name = branch or lattice.root_branch
    return generate_machine_registry(lattice, {name: areas}, scheme=scheme, series={name: series})


def _round_up(value: float, digits: int = 3) -> float:
    """``value`` rounded up to ``digits`` significant digits, for ranges people read."""
    scale = 10.0 ** (math.floor(math.log10(abs(value))) - digits + 1)
    return round(math.ceil(value / scale - 1e-9) * scale, 12)


def _component(components, name: str):
    if name not in components:
        raise KeyError(f"no shared component {name!r}")
    return components[name]


def _main_parameter(placement, order: int, fields: bool = False) -> str:
    """The parameter a magnet's circuit sets: in the form the lattice gives it (normalized or field,
    per length or integrated); a magnet at zero, in the branch's form (``fields``), integrated if thin.
    A bend's implied reference field is normalized."""
    group = placement.element.MagneticMultipoleP
    given = strength_parameter(group, order)
    if group is not None and getattr(group, given):
        return f"MagneticMultipoleP.{given}"
    letter = "B" if fields and placement.kind is not ElementKind.BEND else "K"
    return f"MagneticMultipoleP.{letter}n{order}{'L' if placement.length == 0.0 else ''}"


def _by_fields(placements) -> bool:
    """Whether a branch gives its magnets by their fields (ADR 0012)."""
    return any(
        has_fields(p.element.MagneticMultipoleP) or (p.element.SolenoidP and p.element.SolenoidP.Bsol)
        for p in placements
    )


def _strings(placements: list) -> list[list]:
    """Magnets that most likely share a supply: one kind, the same length and the same design strength.

    Compared as integrated strengths, so a lattice may give either form.
    Magnets at zero strength are left out. Strings keep beam order.
    """
    groups: list[tuple[tuple, list]] = []
    for placement in placements:
        parameter = _main_parameter(placement, _ORDER[placement.kind])
        value = design_value(placement, parameter)
        integrated = value if parameter.endswith("L") else value * placement.length
        if integrated == 0.0:
            continue
        for (kind, length, strength), members in groups:
            if (
                kind is placement.kind
                and abs(length - placement.length) <= SERIES_LENGTH_TOLERANCE
                and math.isclose(strength, integrated, rel_tol=SERIES_STRENGTH_TOLERANCE)
            ):
                members.append(placement)
                break
        else:
            groups.append(((placement.kind, placement.length, integrated), [placement]))
    return [members for _, members in groups if len(members) > 1]


def _describe_branch(lattice, layout, series: frozenset, supplies: list, bpms: list, rf: list) -> None:
    branch_rigidity = rigidity(lattice, layout.id)
    placements = [p for p in layout.placements if lattice.component_of(p.id) is None]
    fields = _by_fields(placements)
    string_of = {}
    for string in _strings([p for p in placements if p.kind in series and p.kind in _ORDER]):
        for placement in string:
            string_of[placement.id] = string
    for placement in placements:
        kind = placement.kind
        if kind in _ORDER:
            string = string_of.get(placement.id)
            if string is None:
                supplies.append(
                    _magnet_supply(
                        [placement], f"{layout.branch.name}/{placement.name}", branch_rigidity, fields
                    )
                )
            elif placement.id == string[0].id:  # the first magnet of a string brings in the whole string
                families = sorted({p.definition.name for p in string})
                supplies.append(_magnet_supply(string, "+".join(families), branch_rigidity, fields))
        elif kind is ElementKind.KICKER:
            supplies.append(_corrector_supply(placement, f"{layout.branch.name}/{placement.name}", fields))
        elif kind is ElementKind.SOLENOID:
            supplies.append(
                _solenoid_supply(placement, f"{layout.branch.name}/{placement.name}", branch_rigidity, fields)
            )
        elif kind is ElementKind.INSTRUMENT and not NOT_BPMS & set(placement.element.tags):
            bpms.append(Bpm(occurrence=placement.id))
        elif kind is ElementKind.RFCAVITY:
            voltage = placement.element.RFP.voltage if placement.element.RFP else 0.0
            top = _round_up(max(2.0 * voltage / 1e6, 1.0))
            rf.append(RfStation(occurrence=placement.id, voltage=(0.0, top)))


def _magnet_supply(members: list, key: str, branch_rigidity: float, fields: bool = False) -> Supply:
    """One supply driving ``members`` in series, or one magnet alone.

    The supply's range covers twice the largest design current, or at least
    :data:`PLACEHOLDER_MAGNET_CURRENT`.
    """
    order = _ORDER[members[0].kind]
    magnets, tops, designs = [], [], []
    for placement in members:
        parameter = _main_parameter(placement, order, fields)
        design = design_value(placement, parameter)
        thin = placement.length == 0.0
        transfer = placeholder_transfer(order, thin=thin)
        magnets.append(
            CircuitMagnet(
                occurrences=(placement.id,),
                parameter=parameter,
                transfer=transfer,
                polarity=-1 if design < 0 else 1,
            )
        )
        per_length = design if thin or not parameter.endswith("L") else design / placement.length
        scale = 1.0 if is_field(parameter) else branch_rigidity
        tops.append(2.0 * abs(per_length) * scale / transfer.coefficients[1])
        designs.append(design)
    top = _round_up(max(*tops, PLACEHOLDER_MAGNET_CURRENT))
    limits = (0.0, top) if any(designs) else (-top, top)
    return Supply(key=key, circuits=(SupplyCircuit(limits=limits, magnets=tuple(magnets)),))


def _corrector_supply(placement, key: str, fields: bool = False) -> Supply:
    tags = placement.element.tags
    letter = "B" if fields else "K"
    normal, skew = f"{letter}n0L", f"{letter}s0L"
    planes = [(None, normal)] if "HCOR" in tags else [(None, skew)] if "VCOR" in tags else []
    planes = planes or [("h", normal), ("v", skew)]
    limits = (-PLACEHOLDER_CORRECTOR_CURRENT, PLACEHOLDER_CORRECTOR_CURRENT)
    circuits = tuple(
        SupplyCircuit(
            plane=plane,
            limits=limits,
            magnets=(
                CircuitMagnet(
                    occurrences=(placement.id,),
                    parameter=f"MagneticMultipoleP.{parameter}",
                    transfer=corrector_transfer(),
                ),
            ),
        )
        for plane, parameter in planes
    )
    return Supply(key=key, circuits=circuits)


def _solenoid_supply(placement, key: str, branch_rigidity: float, fields: bool = False) -> Supply:
    """A solenoid's own supply, setting its field (``Bsol``) or normalized strength (``Ksol``) as the
    lattice gives it; its range twice the design current, or at least
    :data:`PLACEHOLDER_MAGNET_CURRENT`."""
    group = placement.element.SolenoidP
    by_field = (group is not None and group.Bsol != 0.0) or (fields and not (group and group.Ksol))
    parameter = "SolenoidP.Bsol" if by_field else "SolenoidP.Ksol"
    design = design_value(placement, parameter)
    transfer = PolynomialTransfer(coefficients=(0.0, PLACEHOLDER_SOLENOID_PER_AMPERE), field_unit="T")
    field = abs(design) * (1.0 if by_field else branch_rigidity)
    top = _round_up(max(2.0 * field / PLACEHOLDER_SOLENOID_PER_AMPERE, PLACEHOLDER_MAGNET_CURRENT))
    magnet = CircuitMagnet(
        occurrences=(placement.id,), parameter=parameter, transfer=transfer, polarity=-1 if design < 0 else 1
    )
    limits = (0.0, top) if design else (-top, top)
    return Supply(key=key, circuits=(SupplyCircuit(limits=limits, magnets=(magnet,)),))


def _shared_supply(lattice: Lattice, component, magnet: SharedMagnet) -> Supply:
    roles = {t.role: lattice.placement(t.occurrence) for t in component.traversals}
    missing = [r for r in (magnet.home, *magnet.driven) if r not in roles]
    if missing:
        raise KeyError(f"{component.name!r} has no traversal {', '.join(map(repr, missing))}")
    targets = [roles[r] for r in magnet.driven]
    if not targets:
        raise ValueError("a shared magnet supply needs at least one driven traversal")
    fields, rigidities = [], []
    for placed in targets:
        bend = placed.element.BendP if placed.kind is ElementKind.BEND else None
        if bend is None or placed.length == 0.0:
            raise ValueError(f"{placed.name!r} must be a bend with length to be driven by a supply")
        fields.append(bend.angle_ref / placed.length)
        rigidities.append(rigidity(lattice, placed.branch_id))
    if any(
        not math.isclose(f * r, fields[0] * rigidities[0], rel_tol=1e-9)
        for f, r in zip(fields, rigidities, strict=True)
    ):
        raise ValueError(f"the driven traversals of {component.name!r} ask for different fields")
    if any(not math.isclose(r, rigidities[0], rel_tol=1e-9) for r in rigidities):
        raise ValueError(f"the driven traversals of {component.name!r} have different reference rigidities")
    design = fields[0]
    one = CircuitMagnet(
        occurrences=tuple(p.id for p in targets),
        parameter="MagneticMultipoleP.Kn0",
        transfer=placeholder_transfer(0),
        polarity=-1 if design < 0 else 1,
    )
    top = _round_up(
        max(2.0 * abs(design) * rigidities[0] / PLACEHOLDER_DIPOLE_PER_AMPERE, PLACEHOLDER_MAGNET_CURRENT)
    )
    return Supply(key=component.name, circuits=(SupplyCircuit(limits=(0.0, top), magnets=(one,)),))
