"""EIC names, after EIC-SEG-RSI-009 Rev 02, "Naming Requirements" (effective 2026-08-18).

Syntax (Rev 02 made it EPICS compatible: ``:`` replaces the ``.`` of ADO
names and the ``_`` of Rev 01):

- lattice (beamline) devices: ``aa-ddpp:nn-cc:ss``, e.g. ``es6-q42``,
  ``es6-q42:2-ps:sp``;
- non-lattice devices: ``aa:bb-ddpp:zz:nn-ss``, e.g. ``es-btun-rdx``.

``aa`` is the area, for the lattice the machine prefix and sector (``es6``);
``dd`` a device abbreviation; ``pp`` a position; ``nn`` an append number;
``cc`` a controller; ``ss`` a signal, optionally followed by more ``:``
components. Names are lower case. Abbreviations come from the naming
repository, which SEG maintains on SharePoint; :data:`DEFAULT_REPOSITORY` is
a snapshot (see its ``source``).

The lattice position is a clock coordinate: metres from the area's nominal
reference point (an IP), measured at the element's exit, in the direction
the area counts, modulo the sector length, rounded half up. Positions of the
ESR naming table (``es swn lattice.csv``, 7411 names) follow this rule except
for the ring's start marker. The append number orders, by s, the devices of
one kind that share an area and a position, starting at 1; a lone device has
none.
"""

import json
import math
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from functools import cache
from importlib import resources
from typing import Annotated, Literal

from pydantic import AfterValidator, Field, model_validator

from ..lattice import ElementDefinition, ElementKind
from ..lattice._base import DomainModel, Real

_TOKEN = r"[a-z][A-Za-z]*"
_SIGNALS = r"(?P<signals>(?::[a-z][A-Za-z0-9]*)*)"
LATTICE_NAME = re.compile(
    rf"(?P<area>[a-z][a-z0-9]*)-(?P<device>{_TOKEN})(?P<position>\d+)(?::(?P<append>\d+))?"
    rf"(?:-(?P<controller>{_TOKEN}))?{_SIGNALS}"
)
NON_LATTICE_NAME = re.compile(
    rf"(?P<area>[a-z0-9]+)(?::(?P<specific_area>[A-Za-z0-9.]+))?-(?P<device>{_TOKEN})(?P<position>\d+)?"
    rf"(?::(?P<secondary_position>\d+))?(?::(?P<append>\d+))?"
    rf"(?:-(?P<signal>[a-z][A-Za-z0-9]*)(?P<more>(?::[a-z][A-Za-z0-9]*)*))?"
)


@dataclass(frozen=True, slots=True)
class ParsedName:
    """The elements of an EIC name."""

    form: Literal["lattice", "non-lattice"]
    area: str
    device: str
    position: int | None = None
    append: int | None = None
    controller: str | None = None
    signals: tuple[str, ...] = ()
    specific_area: str | None = None
    secondary_position: int | None = None


def parse_name(name: str) -> ParsedName:
    """The elements of ``name``; a name that fits both forms is read as a lattice name."""
    if match := LATTICE_NAME.fullmatch(name):
        return ParsedName(
            form="lattice",
            area=match["area"],
            device=match["device"],
            position=int(match["position"]),
            append=_int(match["append"]),
            controller=match["controller"],
            signals=tuple(match["signals"].split(":")[1:]),
        )
    if match := NON_LATTICE_NAME.fullmatch(name):
        signal = match["signal"]
        return ParsedName(
            form="non-lattice",
            area=match["area"],
            device=match["device"],
            position=_int(match["position"]),
            append=_int(match["append"]),
            signals=(signal, *match["more"].split(":")[1:]) if signal else (),
            specific_area=match["specific_area"],
            secondary_position=_int(match["secondary_position"]),
        )
    raise ValueError(f"{name!r} is not an EIC name (aa-ddpp:nn-cc:ss or aa:bb-ddpp:zz:nn-ss)")


def _int(text: str | None) -> int | None:
    return None if text is None else int(text)


def _check_syntax(name: str) -> str:
    parse_name(name)
    return name


#: A name in EIC syntax; its abbreviations are checked against a repository separately.
EicName = Annotated[str, AfterValidator(_check_syntax)]


def lattice_name(
    area: str,
    device: str,
    position: int,
    append: int | None = None,
    controller: str | None = None,
    signals: tuple[str, ...] = (),
) -> str:
    """``aa-ddpp:nn-cc:ss``, omitting the elements that are None."""
    name = f"{area}-{device}{position}" + (f":{append}" if append is not None else "")
    name += (f"-{controller}" if controller else "") + "".join(f":{s}" for s in signals)
    return _check_syntax(name)


def non_lattice_name(
    area: str, device: str, position: int | None = None, signals: tuple[str, ...] = ()
) -> str:
    """``aa-ddpp-ss``, the non-lattice form without specific area, secondary position or append number."""
    name = f"{area}-{device}{'' if position is None else position}"
    if signals:
        name += f"-{signals[0]}" + "".join(f":{s}" for s in signals[1:])
    return _check_syntax(name)


class NamingRepository(DomainModel):
    """Approved abbreviations. Areas are machine or building prefixes; a sector number may follow."""

    source: str
    areas: frozenset[str]
    devices: frozenset[str]
    controllers: frozenset[str]
    signals: frozenset[str]

    def extended(
        self,
        *,
        areas: Iterable[str] = (),
        devices: Iterable[str] = (),
        controllers: Iterable[str] = (),
        signals: Iterable[str] = (),
        note: str = "",
    ) -> NamingRepository:
        """A copy that also lists the given abbreviations, e.g. temporary ones awaiting approval."""
        return NamingRepository(
            source=f"{self.source} Extended: {note}" if note else self.source,
            areas=self.areas | set(areas),
            devices=self.devices | set(devices),
            controllers=self.controllers | set(controllers),
            signals=self.signals | set(signals),
        )

    def problems(self, name: str) -> list[str]:
        """Elements of ``name`` that the repository does not list."""
        parsed = parse_name(name)
        found = []
        prefix = re.match(r"[a-z]+", parsed.area)
        if parsed.area not in self.areas and (prefix is None or prefix[0] not in self.areas):
            found.append(f"area {parsed.area!r}")
        if parsed.device not in self.devices:
            found.append(f"device {parsed.device!r}")
        if parsed.controller is not None and parsed.controller not in self.controllers:
            found.append(f"controller {parsed.controller!r}")
        found += [f"signal {s!r}" for s in parsed.signals if s not in self.signals]
        return found


@cache
def default_repository() -> NamingRepository:
    """The snapshot shipped with EICViBE (``eic_naming_repository.json``)."""
    data = json.loads(resources.files(__package__).joinpath("eic_naming_repository.json").read_text())
    return NamingRepository.model_validate(data)


class NamingArea(DomainModel):
    """A stretch of a branch named with one area prefix, e.g. ESR sector 6 (``es6``)."""

    prefix: Annotated[str, Field(pattern=r"^[a-z][a-z0-9]*$")]
    start: Real = Field(description="s where the area begins, m")
    end: Real = Field(description="s where the area ends, m")
    origin: Real = Field(0.0, description="s of the reference point positions count from, m")
    direction: Literal[1, -1] = Field(1, description="+1 if positions grow with s, -1 if they shrink")
    period: Real | None = Field(None, description="Positions wrap modulo this length (the sector length)")

    @model_validator(mode="after")
    def _ordered(self):
        if not self.start < self.end:
            raise ValueError(f"area {self.prefix}: start {self.start} is not before end {self.end}")
        return self

    def position(self, s: float) -> int:
        """Clock-coordinate position, in m, of an element whose exit is at ``s``."""
        distance = self.direction * (s - self.origin)
        if self.period is not None:
            distance %= self.period
        if distance < -0.5:
            raise ValueError(f"s = {s} m lies before the reference point of area {self.prefix}")
        return math.floor(distance + 0.5)


def ring_sectors(
    machine: str,
    *,
    sectors: int = 12,
    sector_length: float = 320.0,
    reference_sector: int = 6,
    boundaries: Mapping[int, float] | None = None,
) -> tuple[NamingArea, ...]:
    """Clock sectors of a ring whose s = 0 is the IP of ``reference_sector`` (IP6 for the ESR).

    Even sectors start at their IP and count forward; odd sectors end at the
    next IP and count back from it. IPs sit every two sector lengths, at
    nominal positions (3840 m around the ESR, whose lattice is 3834 m).
    ``boundaries`` overrides where sectors begin in s, by sector number.
    """
    numbers = [(reference_sector - 1 + k) % sectors + 1 for k in range(sectors)]
    starts = [(boundaries or {}).get(n, k * sector_length) for k, n in enumerate(numbers)]
    ends = [*starts[1:], max(sectors * sector_length, starts[-1] + sector_length)]
    areas = []
    for k, (number, start, end) in enumerate(zip(numbers, starts, ends, strict=True)):
        forward = number % 2 == reference_sector % 2
        areas.append(
            NamingArea(
                prefix=f"{machine}{number}",
                start=start,
                end=end,
                origin=(k if forward else k + 1) * sector_length,
                direction=1 if forward else -1,
                period=sector_length,
            )
        )
    return tuple(areas)


def eic_device(element: ElementDefinition) -> str | None:
    """Device abbreviation of a lattice element (repository column "EIC Lattice Keyword").

    None for bookkeeping elements that have no slot.
    """
    kind = element.kind
    m = element.MagneticMultipoleP

    def skew_only(order: int) -> bool:
        if m is None:
            return False
        normal = getattr(m, f"Kn{order}") or getattr(m, f"Kn{order}L")
        return not normal and bool(getattr(m, f"Ks{order}") or getattr(m, f"Ks{order}L"))

    if kind is ElementKind.BEND:
        tilt = element.BendP.tilt_ref if element.BendP else 0.0
        return "dv" if math.isclose(abs(tilt), math.pi / 2, abs_tol=1e-6) else "dh"
    if kind is ElementKind.KICKER:
        return "th" if "HCOR" in element.tags else "tv" if "VCOR" in element.tags else "t"
    return {
        ElementKind.QUADRUPOLE: "qs" if skew_only(1) else "q",
        ElementKind.SEXTUPOLE: "sxs" if skew_only(2) else "sx",
        ElementKind.OCTUPOLE: "cm",
        ElementKind.MULTIPOLE: "cm",
        ElementKind.SOLENOID: "sol",
        ElementKind.RFCAVITY: "cav",
        ElementKind.CRABCAVITY: "cav",
        ElementKind.INSTRUMENT: "b",
        ElementKind.DRIFT: "dt",
        ElementKind.MARKER: "mark",
    }.get(kind)


@dataclass(frozen=True)
class NamingScheme:
    """The abbreviations and signal tokens generated names use.

    The naming committee has not fixed these for the virtual accelerator's
    channels, so they are parameters; the defaults are temporary choices
    (ADR 0003). Changing them renames channels but not devices, whose ids
    come from the lattice.
    """

    supply: str = "ps"
    """Controller of a magnet or corrector supply."""
    setpoint: str = "sp"
    readback: str = "rb"
    position_x: str = "rdx"
    position_y: str = "rdy"
    planes: tuple[str, str] = ("h", "v")
    """Extra signal components of a two-plane corrector."""
    rf_voltage: str = "volt"
    rf_phase: str = "phase"
    tune_monitor: str = "btun"
    tune_x: str = "rdx"
    tune_y: str = "rdy"
    device: Callable[[ElementDefinition], str | None] = eic_device
    """Device abbreviation of a lattice element; None for elements without a slot."""


def slot_names(
    placements: Iterable,
    areas: Iterable[NamingArea],
    device: Callable[[ElementDefinition], str | None] = eic_device,
) -> dict[str, str]:
    """Slot name of every placed element that has a device abbreviation, by occurrence id.

    ``placements`` are a branch's :class:`~eicvibe.domain.lattice.PlacedElement`
    objects in s order. An element belongs to the area containing its centre.
    """
    areas = tuple(areas)
    named = []
    for placed in placements:
        abbreviation = device(placed.element)
        if abbreviation is None:
            continue
        centre = 0.5 * (placed.s_enter + placed.s_exit)
        area = next((a for a in areas if a.start <= centre < a.end), None)
        if area is None and areas and centre == areas[-1].end:
            area = areas[-1]
        if area is None:
            raise ValueError(f"{placed.name} at s = {centre} m is in no naming area")
        named.append((placed.id, area.prefix, abbreviation, area.position(placed.s_exit)))
    shared = Counter((prefix, abbreviation, position) for _, prefix, abbreviation, position in named)
    seen: Counter = Counter()
    names = {}
    for occurrence, prefix, abbreviation, position in named:
        key = (prefix, abbreviation, position)
        seen[key] += 1
        append = seen[key] if shared[key] > 1 else None
        names[occurrence] = lattice_name(prefix, abbreviation, position, append)
    return names
