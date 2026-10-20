"""Devices and channels the control system exposes, and how they relate to the lattice (ADR 0003).

A :class:`MachineRegistry` lists the EIC names of the lattice elements, the
devices (power supplies, BPMs, RF stations) and their channels, each a
pvAccess PV with a role, a Normative Type, a unit and limits, and the power
circuits (:mod:`.power`): which magnets each supply drives, alone or as a
string in series, and each magnet's transfer function. A magnet supply's
setting binds to its circuit; another setting binds to a lattice parameter
through a :class:`Calibration` (an RF voltage, say); a measurement channel
binds to an observable such as the closed orbit at a BPM. The registry is kept outside the
:class:`~eicvibe.domain.lattice.Lattice`, keyed by its stable ids, because
hardware changes independently of lattice revisions.
"""

from enum import StrEnum
from functools import cached_property
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from ..lattice import ALLOWED_GROUPS, Lattice, parameter_info
from ..lattice._base import DomainModel, Id, Label, Real
from .naming import EicName, NamingRepository, default_repository
from .power import (
    CalibrationSource,
    CircuitMagnet,
    PowerCircuit,
    TransferService,
    limit_problems,
    string_problems,
)


class DeviceType(StrEnum):
    POWER_SUPPLY = "power_supply"
    BPM = "bpm"
    RF_STATION = "rf_station"
    TUNE_MONITOR = "tune_monitor"


class ChannelRole(StrEnum):
    SETTING = "setting"
    """A writable demand value."""
    READBACK = "readback"
    """The device's reading of one of its settings."""
    MEASUREMENT = "measurement"
    """An observation of the beam or the machine, such as a BPM position."""
    STATUS = "status"
    """A state, such as on/off or an interlock."""


class ValueType(StrEnum):
    """Normative Type of the channel's value."""

    DOUBLE = "double"
    INT = "int"
    STRING = "string"
    ENUM = "enum"
    DOUBLE_ARRAY = "double_array"


class Observable(StrEnum):
    """Quantities a measurement channel reports, in SI units, before its ``scale``."""

    ORBIT_X = "orbit_x"
    """Closed-orbit x at an occurrence, in m."""
    ORBIT_Y = "orbit_y"
    """Closed-orbit y at an occurrence, in m."""
    TUNE_X = "tune_x"
    """Fractional horizontal tune of a periodic branch."""
    TUNE_Y = "tune_y"
    """Fractional vertical tune of a periodic branch."""


class Calibration(DomainModel):
    """Engineering value x (e.g. a current in A) to a field f(x) = sum of c_k x^k.

    For a normalized lattice parameter (``normalized``), the model value is
    f(x) / (B rho); otherwise it is f(x). The polarity is part of the
    coefficients. f must be monotonic over the channel's control limits so
    that it can be inverted.
    """

    coefficients: tuple[Real, ...] = Field(min_length=2, description="c_0, c_1, ... of f(x)")
    engineering_unit: str
    field_unit: str
    normalized: bool = Field(True, description="Divide by the reference rigidity B rho")
    source: CalibrationSource = CalibrationSource.PLACEHOLDER

    def field(self, x: float) -> float:
        return sum(c * x**k for k, c in enumerate(self.coefficients))

    def to_model(self, x: float, rigidity: float | None = None) -> float:
        """Model value for engineering value ``x``; ``rigidity`` (T m) is needed when normalized."""
        return self.field(x) / self._rigidity(rigidity) if self.normalized else self.field(x)

    def to_engineering(self, value: float, low: float, high: float, rigidity: float | None = None) -> float:
        """Engineering value in [low, high] giving model ``value``, by bisection."""
        target = value * self._rigidity(rigidity) if self.normalized else value
        f_low, f_high = self.field(low) - target, self.field(high) - target
        if f_low == 0.0:
            return low
        if f_low * f_high > 0.0:
            raise ValueError(f"model value {value} is outside the calibrated range [{low}, {high}]")
        for _ in range(200):
            mid = 0.5 * (low + high)
            f_mid = self.field(mid) - target
            if f_mid == 0.0 or high - low <= 1e-15 * max(1.0, abs(mid)):
                return mid
            low, high, f_low = (mid, high, f_mid) if f_low * f_mid > 0.0 else (low, mid, f_low)
        return 0.5 * (low + high)

    def _rigidity(self, rigidity: float | None) -> float:
        if rigidity is None or not rigidity > 0.0:
            raise ValueError("a normalized calibration needs a positive reference rigidity")
        return rigidity


class ParameterBinding(DomainModel):
    """A setting's lattice parameter, e.g. ``MagneticMultipoleP.Kn1`` of a family definition."""

    kind: Literal["parameter"] = "parameter"
    targets: tuple[Id, ...] = Field(min_length=1, description="Definition or occurrence ids")
    parameter: str = Field(description="Parameter path, group.name")
    calibration: Calibration | None = Field(None, description="None if the channel is in model units")

    @model_validator(mode="after")
    def _known_parameter(self) -> Self:
        group, _, name = self.parameter.partition(".")
        try:
            parameter_info(group, name)
        except KeyError as error:
            raise ValueError(error.args[0]) from None
        return self


class ObservableBinding(DomainModel):
    """A measurement's observable: the channel value is ``scale`` times the observable."""

    kind: Literal["observable"] = "observable"
    observable: Observable
    target: Id = Field(description="Occurrence id for an orbit, branch id for a tune")
    scale: Real = Field(1.0, description="Channel units per SI unit, e.g. 1000 for mm")


class CircuitBinding(DomainModel):
    """A supply's setting: the current of a power circuit, which sets the magnets on it."""

    kind: Literal["circuit"] = "circuit"
    circuit: Id


Binding = Annotated[ParameterBinding | ObservableBinding | CircuitBinding, Field(discriminator="kind")]


class Slot(DomainModel):
    """The EIC name of one physical lattice element, ``aa-ddpp:nn`` (e.g. ``es6-q7``, ``es6-dt5``).

    Every element with a device abbreviation has one, drifts and markers
    included, as in the ESR naming table. A magnet that several branches
    pass through (a shared component) is one slot with several occurrences.
    """

    name: EicName
    occurrences: tuple[Id, ...] = Field(min_length=1, description="Occurrences of the element")


class Device(DomainModel):
    id: Id
    name: EicName
    type: DeviceType
    occurrences: tuple[Id, ...] = Field((), description="Occurrences of the device, if placed in the lattice")


class Channel(DomainModel):
    """One pvAccess PV."""

    name: EicName
    device: Id
    role: ChannelRole
    value_type: ValueType = ValueType.DOUBLE
    unit: str = ""
    precision: int = Field(4, ge=0, le=17)
    display_limits: tuple[Real, Real] | None = None
    control_limits: tuple[Real, Real] | None = Field(None, description="Writable range of a setting")
    readback_of: EicName | None = Field(None, description="The setting a readback reads")
    binding: Binding | None = None

    @model_validator(mode="after")
    def _role_rules(self) -> Self:
        for limits in (self.display_limits, self.control_limits):
            if limits is not None and not limits[0] < limits[1]:
                raise ValueError(f"{self.name}: limits {limits} are not increasing")
        if (self.role is ChannelRole.READBACK) != (self.readback_of is not None):
            raise ValueError(f"{self.name}: readback_of is required for, and only for, a readback")
        if isinstance(self.binding, ObservableBinding) and self.role is not ChannelRole.MEASUREMENT:
            raise ValueError(f"{self.name}: only a measurement binds to an observable")
        if (
            isinstance(self.binding, ParameterBinding | CircuitBinding)
            and self.role is not ChannelRole.SETTING
        ):
            raise ValueError(f"{self.name}: only a setting binds to a lattice parameter or a circuit")
        if self.role is ChannelRole.SETTING and self.control_limits is None:
            raise ValueError(f"{self.name}: a setting needs control limits")
        return self


class MachineRegistry(DomainModel):
    """Element names, devices, channels and power circuits of one machine, for one lattice."""

    format_version: Literal[1] = 1
    name: Label
    lattice: Label = Field(description="Name of the lattice the bindings refer to")
    slots: tuple[Slot, ...] = Field((), description="EIC names of the lattice elements")
    devices: tuple[Device, ...] = ()
    channels: tuple[Channel, ...] = ()
    circuits: tuple[PowerCircuit, ...] = Field((), description="Which magnets each supply drives, and how")

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        problems = []
        for what, names in (
            ("element name", [s.name for s in self.slots]),
            ("element occurrence", [o for s in self.slots for o in s.occurrences]),
            ("device id", [d.id for d in self.devices]),
            ("device name", [d.name for d in self.devices]),
            ("channel name", [c.name for c in self.channels]),
            ("circuit id", [c.id for c in self.circuits]),
        ):
            problems += [f"duplicate {what} {n!r}" for n in sorted({n for n in names if names.count(n) > 1})]
        devices = {d.id for d in self.devices}
        supplies = {d.id for d in self.devices if d.type is DeviceType.POWER_SUPPLY}
        circuits = {c.id for c in self.circuits}
        problems += [
            f"circuit {c.name}: {c.device!r} is not a power supply"
            for c in self.circuits
            if c.device not in supplies
        ]
        settings = {c.name: c for c in self.channels if c.role is ChannelRole.SETTING}
        for channel in self.channels:
            if channel.device not in devices:
                problems.append(f"{channel.name}: unknown device {channel.device!r}")
            if isinstance(channel.binding, CircuitBinding) and channel.binding.circuit not in circuits:
                problems.append(f"{channel.name}: unknown circuit {channel.binding.circuit!r}")
            if channel.readback_of is not None:
                setting = settings.get(channel.readback_of)
                if setting is None:
                    problems.append(
                        f"{channel.name}: reads back {channel.readback_of!r}, which is not a setting"
                    )
                elif setting.device != channel.device:
                    problems.append(f"{channel.name}: reads back a setting of another device")
        if problems:
            raise ValueError("; ".join(problems))
        return self

    @cached_property
    def _slot_by_occurrence(self) -> dict[str, Slot]:
        return {o: s for s in self.slots for o in s.occurrences}

    def slot_of(self, occurrence: str) -> Slot | None:
        """The EIC name of the element an occurrence places, if it has one."""
        return self._slot_by_occurrence.get(occurrence)

    @cached_property
    def _circuits(self) -> dict[str, PowerCircuit]:
        return {c.id: c for c in self.circuits}

    def circuit(self, circuit_id: str) -> PowerCircuit:
        try:
            return self._circuits[circuit_id]
        except KeyError:
            raise KeyError(f"no circuit {circuit_id!r}") from None

    @cached_property
    def _magnets_at(self) -> dict[str, list[tuple[PowerCircuit, CircuitMagnet]]]:
        found: dict[str, list] = {}
        for circuit in self.circuits:
            for magnet in circuit.magnets:
                for occurrence in magnet.occurrences:
                    found.setdefault(occurrence, []).append((circuit, magnet))
        return found

    def circuits_at(self, occurrence: str) -> tuple[tuple[PowerCircuit, CircuitMagnet], ...]:
        """The circuits that set a parameter of the magnet at an occurrence, each with its entry for it."""
        return tuple(self._magnets_at.get(occurrence, ()))

    def setting_of(self, circuit_id: str) -> Channel | None:
        """The setting channel that drives a circuit."""
        return next(
            (
                c
                for c in self.channels
                if isinstance(c.binding, CircuitBinding) and c.binding.circuit == circuit_id
            ),
            None,
        )

    def power_problems(
        self, lattice: Lattice, services: dict[str, TransferService] | None = None
    ) -> list[str]:
        """Strings whose magnets need different currents for their design values, and magnets whose
        design values need a current beyond their supply's limits."""
        problems = []
        for circuit in self.circuits:
            setting = self.setting_of(circuit.id)
            if setting is None or setting.control_limits is None:
                continue
            if circuit.series:
                found = string_problems(lattice, circuit, (-1e12, 1e12), services)
                if found:
                    problems += found
                    continue
            problems += limit_problems(lattice, circuit, setting.control_limits, services)
        return problems

    @cached_property
    def _by_name(self) -> dict[str, Channel]:
        return {c.name: c for c in self.channels}

    def channel(self, name: str) -> Channel:
        try:
            return self._by_name[name]
        except KeyError:
            raise KeyError(f"no channel {name!r}") from None

    def channels_of(self, device: str) -> tuple[Channel, ...]:
        """Channels of a device, by device id or name."""
        ids = {d.id for d in self.devices if device in (d.id, d.name)}
        return tuple(c for c in self.channels if c.device in ids)

    def naming_problems(self, repository: NamingRepository | None = None) -> list[str]:
        """Names with abbreviations the naming repository does not list (yet), one line per name."""
        repository = repository or default_repository()
        found = []
        for name in (
            *(s.name for s in self.slots),
            *(d.name for d in self.devices),
            *(c.name for c in self.channels),
        ):
            if problems := repository.problems(name):
                found.append(f"{name}: {', '.join(problems)} not in the repository")
        return found

    def check_against(self, lattice: Lattice) -> list[str]:
        """Problems with the bindings for ``lattice``; empty when every binding resolves."""
        problems = []
        if lattice.name != self.lattice:
            problems.append(f"registry is for lattice {self.lattice!r}, not {lattice.name!r}")
        definitions = {e.id: e for e in lattice.elements}
        occurrences = {o.id: o for b in lattice.branches for o in b.occurrences}
        branches = {b.id: b for b in lattice.branches}
        for slot in self.slots:
            problems += [
                f"{slot.name}: no occurrence {o!r}" for o in slot.occurrences if o not in occurrences
            ]
        for circuit in self.circuits:
            for magnet in circuit.magnets:
                group = magnet.parameter.partition(".")[0]
                for occurrence in magnet.occurrences:
                    if occurrence not in occurrences:
                        problems.append(f"circuit {circuit.name}: no occurrence {occurrence!r}")
                        continue
                    element = definitions[occurrences[occurrence].element]
                    if group not in ALLOWED_GROUPS[element.kind]:
                        problems.append(
                            f"circuit {circuit.name}: {element.kind.value} {element.name!r} has no {group}"
                        )
        for channel in self.channels:
            binding = channel.binding
            if isinstance(binding, ParameterBinding):
                group = binding.parameter.partition(".")[0]
                for target in binding.targets:
                    if target in occurrences:
                        target = occurrences[target].element
                    element = definitions.get(target)
                    if element is None:
                        problems.append(f"{channel.name}: no definition or occurrence {target!r}")
                    elif group not in ALLOWED_GROUPS[element.kind]:
                        problems.append(
                            f"{channel.name}: {element.kind.value} {element.name!r} has no {group}"
                        )
            elif isinstance(binding, ObservableBinding):
                if binding.observable in (Observable.ORBIT_X, Observable.ORBIT_Y):
                    if binding.target not in occurrences:
                        problems.append(
                            f"{channel.name}: an orbit needs an occurrence, not {binding.target!r}"
                        )
                elif binding.target not in branches or not branches[binding.target].periodic:
                    problems.append(f"{channel.name}: a tune needs a periodic branch, not {binding.target!r}")
        return problems
