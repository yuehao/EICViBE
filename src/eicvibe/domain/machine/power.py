"""Power circuits and magnet transfer functions (ADR 0003).

A power circuit is one supply and the magnets it drives. A magnet powered on
its own is a circuit of one; main dipoles and quadrupole families are often
strings, many magnets in series on one supply, all carrying its current. A
trim winding on a magnet of a string is a second circuit: the fields of all
circuits that set one parameter of one magnet add.

Each magnet on a circuit has its own transfer function from the supply's
engineering value (a current, in A) to the magnet's field, so measured
excitation curves can replace placeholders magnet by magnet:

- :class:`PolynomialTransfer`: B = c0 + c1 I + c2 I^2 + ...; a linear
  approximation has two coefficients;
- :class:`TableTransfer`: a measured excitation curve, interpolated linearly
  between its points and extrapolated along its end segments;
- :class:`ServiceTransfer`: a dedicated transfer-function service, for
  nonlinear magnets and hysteresis, where the field depends on the current's
  history. The service keeps each magnet's history; a local fallback
  (polynomial or table) serves where no service is available, e.g. offline.

The field is the magnet's field, gradient, ... (T, T/m, T/m^2) or, for an
``integrated`` transfer function, its integral over the magnet (T m, T,
T/m). A magnet's lattice parameter is its normalized strength, the field
divided by the reference rigidity B rho of the branch, times the magnet's
wiring polarity, per length or integrated as the parameter is (``Kn1``,
``Kn1L``); or, where the lattice gives the magnet by its field (``Bn1``,
``Bsol``: an injector line, ADR 0012), the field itself times the polarity.
A setting of a bend's ``Kn0`` gives its field explicitly: whoever applies it
also sets ``BendP.Kn0_from_g_ref`` false, so that 0 A means no field.
"""

from collections.abc import Iterable, Mapping
from enum import StrEnum
from typing import Annotated, Literal, Protocol, Self

from pydantic import Field, model_validator

from ..lattice import SPECIES, ElementKind, Lattice, PlacedElement, parameter_info
from ..lattice._base import DomainModel, Id, Label, Real
from ..lattice.conventions import rigidity_from_P


class CalibrationSource(StrEnum):
    PLACEHOLDER = "placeholder"
    """Generic value for a device type; not a property of the real device."""
    MEASURED = "measured"
    """From magnet or device measurements."""


class _Transfer(DomainModel):
    engineering_unit: str = "A"
    field_unit: str = Field(description="Unit of the field, e.g. T, T/m; T m, T for integrated fields")
    integrated: bool = Field(False, description="The field integrated over the magnet length")
    source: CalibrationSource = CalibrationSource.PLACEHOLDER


class PolynomialTransfer(_Transfer):
    """B = c0 + c1 I + c2 I^2 + ...; two coefficients make a linear approximation."""

    kind: Literal["polynomial"] = "polynomial"
    coefficients: tuple[Real, ...] = Field(min_length=2, description="c0, c1, ...")

    def field(self, current: float) -> float:
        return sum(c * current**k for k, c in enumerate(self.coefficients))


class TableTransfer(_Transfer):
    """A measured excitation curve: field at each current, interpolated linearly."""

    kind: Literal["table"] = "table"
    currents: tuple[Real, ...] = Field(min_length=2, description="Increasing")
    fields: tuple[Real, ...] = Field(min_length=2, description="Strictly monotonic, so the curve inverts")

    @model_validator(mode="after")
    def _monotonic(self) -> Self:
        if len(self.currents) != len(self.fields):
            raise ValueError("a table needs as many fields as currents")
        if any(b <= a for a, b in zip(self.currents, self.currents[1:], strict=False)):
            raise ValueError("table currents must increase")
        steps = [b - a for a, b in zip(self.fields, self.fields[1:], strict=False)]
        if not (all(s > 0 for s in steps) or all(s < 0 for s in steps)):
            raise ValueError("table fields must be strictly monotonic")
        return self

    def field(self, current: float) -> float:
        return _interpolate(current, self.currents, self.fields)

    def current(self, field: float) -> float:
        if self.fields[-1] < self.fields[0]:
            return _interpolate(field, self.fields[::-1], self.currents[::-1])
        return _interpolate(field, self.fields, self.currents)


Approximation = Annotated[PolynomialTransfer | TableTransfer, Field(discriminator="kind")]


class ServiceTransfer(_Transfer):
    """A transfer function computed by a dedicated service, e.g. with hysteresis."""

    kind: Literal["service"] = "service"
    service: Label = Field(description="Name of the transfer-function service; the deployment resolves it")
    magnet: str = Field(
        min_length=1, description="The service's identifier of this magnet, e.g. a serial number"
    )
    hysteresis: bool = Field(False, description="The field depends on the current's history")
    fallback: Approximation | None = Field(None, description="Used where the service is not available")

    @model_validator(mode="after")
    def _same_units(self) -> Self:
        fallback = self.fallback
        if fallback is not None and (fallback.field_unit, fallback.integrated) != (
            self.field_unit,
            self.integrated,
        ):
            raise ValueError("the fallback must give the same field as the service")
        return self


TransferFunction = Annotated[
    PolynomialTransfer | TableTransfer | ServiceTransfer, Field(discriminator="kind")
]


class TransferService(Protocol):
    """A transfer-function service, as :class:`ServiceTransfer` refers to it.

    A service with hysteresis keeps each magnet's history: ask it for the
    field at every current the supply goes through, in order.
    """

    def field(self, transfer: ServiceTransfer, current: float) -> float: ...

    def current(self, transfer: ServiceTransfer, field: float) -> float: ...


class CircuitMagnet(DomainModel):
    """One magnet on a circuit: its transfer function, wiring polarity and the parameter it sets."""

    occurrences: tuple[Id, ...] = Field(
        min_length=1, description="Traversals it sets: one, or several for a magnet shared by branches"
    )
    parameter: str = Field(description="Parameter path, e.g. MagneticMultipoleP.Kn1")
    transfer: TransferFunction
    polarity: Literal[1, -1] = Field(1, description="-1 when the magnet is wired to reverse its field")

    @model_validator(mode="after")
    def _known_parameter(self) -> Self:
        group, _, name = self.parameter.partition(".")
        try:
            parameter_info(group, name)
        except KeyError as error:
            raise ValueError(error.args[0]) from None
        return self


class PowerCircuit(DomainModel):
    """A power supply and the magnets it drives: one, or a string in series."""

    id: Id
    name: str = Field(
        min_length=1, max_length=128, description="E.g. the family a string powers, or the magnet's EIC name"
    )
    device: Id = Field(description="The power supply")
    magnets: tuple[CircuitMagnet, ...] = Field(min_length=1)

    @property
    def series(self) -> bool:
        return len(self.magnets) > 1


# -- conversions -------------------------------------------------------------


def transfer_field(
    transfer: TransferFunction, current: float, services: dict[str, TransferService] | None = None
) -> float:
    """The field a magnet gives at ``current``, as its transfer function defines it."""
    if isinstance(transfer, ServiceTransfer):
        service = (services or {}).get(transfer.service)
        if service is not None:
            return service.field(transfer, current)
        if transfer.fallback is None:
            raise LookupError(
                f"no service {transfer.service!r} and no fallback for magnet {transfer.magnet!r}"
            )
        transfer = transfer.fallback
    return transfer.field(current)


def transfer_current(
    transfer: TransferFunction,
    field: float,
    limits: tuple[float, float],
    services: dict[str, TransferService] | None = None,
) -> float:
    """The current at which a magnet gives ``field``, within the supply's ``limits``."""
    if isinstance(transfer, ServiceTransfer):
        service = (services or {}).get(transfer.service)
        if service is not None:
            return service.current(transfer, field)
        if transfer.fallback is None:
            raise LookupError(
                f"no service {transfer.service!r} and no fallback for magnet {transfer.magnet!r}"
            )
        transfer = transfer.fallback
    if isinstance(transfer, TableTransfer):
        return transfer.current(field)
    return _bisect(transfer.field, field, *limits)


def rigidity(lattice: Lattice, branch: str) -> float:
    """Reference rigidity B rho of a branch, T m, from its BeginningEle."""
    conditions = lattice.initial_conditions(branch)
    reference = conditions.reference if conditions else None
    if reference is None or reference.pc_ref is None or reference.species_ref not in SPECIES:
        name = lattice.branch(branch).name
        raise ValueError(f"branch {name!r} needs a reference particle with pc_ref and a known species")
    return rigidity_from_P(reference.pc_ref, charge=SPECIES[reference.species_ref].charge)


def design_value(placed: PlacedElement, parameter: str) -> float:
    """The lattice's value of ``parameter`` at an occurrence; a bend's reference field where implied."""
    group_name, _, name = parameter.partition(".")
    group = getattr(placed.element, group_name, None)
    value = getattr(group, name, 0.0) if group is not None else 0.0
    bend = placed.element.BendP if placed.kind is ElementKind.BEND else None
    if group_name == "MagneticMultipoleP" and name in ("Kn0", "Kn0L") and bend is not None:
        given = group is not None and (group.Kn0 or group.Kn0L)
        if not given and bend.Kn0_from_g_ref:
            return bend.angle_ref if name == "Kn0L" else bend.angle_ref / placed.length
    return value


def magnet_strength(
    lattice: Lattice,
    magnet: CircuitMagnet,
    occurrence: str,
    current: float,
    services: dict[str, TransferService] | None = None,
) -> float:
    """The value of the magnet's parameter at ``occurrence`` when its circuit carries ``current``."""
    placed = lattice.placement(occurrence)
    field = magnet.polarity * transfer_field(magnet.transfer, current, services)
    field = _convert(field, magnet.transfer.integrated, _integrated(magnet.parameter), placed)
    return field / _scale(lattice, placed, magnet.parameter)


def circuit_strengths(
    lattice: Lattice,
    circuit: PowerCircuit,
    current: float,
    services: dict[str, TransferService] | None = None,
) -> dict[tuple[str, str], float]:
    """Parameter values, by (occurrence id, parameter path), of every magnet a circuit drives."""
    return {
        (occurrence, magnet.parameter): magnet_strength(lattice, magnet, occurrence, current, services)
        for magnet in circuit.magnets
        for occurrence in magnet.occurrences
    }


def total_strengths(
    lattice: Lattice,
    circuits: Iterable[PowerCircuit],
    currents: Mapping[str, float],
    services: dict[str, TransferService] | None = None,
) -> dict[tuple[str, str], float]:
    """Parameter values set by ``circuits`` at ``currents`` (by circuit id).

    Where several circuits drive one parameter of one magnet, e.g. a string
    and a trim winding, their contributions add.
    """
    totals: dict[tuple[str, str], float] = {}
    for circuit in circuits:
        if circuit.id in currents:
            for key, value in circuit_strengths(lattice, circuit, currents[circuit.id], services).items():
                totals[key] = totals.get(key, 0.0) + value
    return totals


def magnet_current(
    lattice: Lattice,
    magnet: CircuitMagnet,
    value: float,
    limits: tuple[float, float],
    services: dict[str, TransferService] | None = None,
    occurrence: str | None = None,
) -> float:
    """The circuit current at which a magnet's parameter takes ``value`` (at its first occurrence)."""
    placed = lattice.placement(occurrence or magnet.occurrences[0])
    field = value * _scale(lattice, placed, magnet.parameter) * magnet.polarity
    field = _convert(field, _integrated(magnet.parameter), magnet.transfer.integrated, placed)
    return transfer_current(magnet.transfer, field, limits, services)


def string_problems(
    lattice: Lattice,
    circuit: PowerCircuit,
    limits: tuple[float, float],
    services: dict[str, TransferService] | None = None,
    tolerance: float = 1e-6,
) -> list[str]:
    """Magnets of a string that need different currents for their design values.

    One current drives a string, so its magnets reach their design values
    together only if their transfer functions and design values agree.
    """
    currents = []
    for magnet in circuit.magnets:
        placed = lattice.placement(magnet.occurrences[0])
        try:
            currents.append(
                (
                    placed.name,
                    magnet_current(lattice, magnet, design_value(placed, magnet.parameter), limits, services),
                )
            )
        except (ValueError, LookupError) as error:
            return [f"{circuit.name}: {placed.name}: {error}"]
    low, high = min(c for _, c in currents), max(c for _, c in currents)
    if high - low > tolerance * max(abs(high), abs(low), 1.0):
        first, last = min(currents, key=lambda c: c[1]), max(currents, key=lambda c: c[1])
        return [
            f"{circuit.name}: its magnets need {low:.6g} to {high:.6g} for their design values "
            f"({first[0]} to {last[0]}), but a string carries one current"
        ]
    return []


def _integrated(parameter: str) -> bool:
    return parameter.endswith("L")


def is_field(parameter: str) -> bool:
    """Whether a parameter is a field (``BnN``, ``BsNL``, ``Bsol``; T, T/m, ...) rather than a strength
    normalized by the reference rigidity (``KnN``, ``Ksol``)."""
    return parameter.partition(".")[2].startswith("B")


def _scale(lattice: Lattice, placed: PlacedElement, parameter: str) -> float:
    """A magnet's field per unit of its parameter: the rigidity for a normalized strength, 1 for a field."""
    return 1.0 if is_field(parameter) else rigidity(lattice, placed.branch_id)


def _convert(field: float, integrated: bool, wanted: bool, placed: PlacedElement) -> float:
    """A field, per length or integrated over the magnet, as the other kind if ``wanted`` differs."""
    if integrated == wanted:
        return field
    if wanted:
        return field * placed.length
    if placed.length == 0.0:
        raise ValueError(f"{placed.name} is thin: it has no per-length strength")
    return field / placed.length


def _interpolate(x: float, xs: tuple[float, ...], ys: tuple[float, ...]) -> float:
    k = 1
    while k < len(xs) - 1 and x > xs[k]:
        k += 1
    x0, x1, y0, y1 = xs[k - 1], xs[k], ys[k - 1], ys[k]
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def _bisect(function, target: float, low: float, high: float) -> float:
    """The x in [low, high] where a monotonic ``function`` reaches ``target``."""
    f_low, f_high = function(low) - target, function(high) - target
    if f_low == 0.0:
        return low
    if f_high == 0.0:
        return high
    if f_low * f_high > 0.0:
        raise ValueError(f"field {target:.6g} is outside the range of [{low:.6g}, {high:.6g}]")
    for _ in range(200):
        mid = 0.5 * (low + high)
        f_mid = function(mid) - target
        if f_mid == 0.0 or high - low <= 1e-15 * max(1.0, abs(mid)):
            return mid
        low, high, f_low = (mid, high, f_mid) if f_low * f_mid > 0.0 else (low, mid, f_low)
    return 0.5 * (low + high)
