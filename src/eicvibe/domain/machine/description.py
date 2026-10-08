"""What a machine's devices are, as people describe them; the registry is assembled from it.

A :class:`MachineDescription` says how lattice elements are named (areas,
tokens, shared magnets, explicit names), which supplies drive which magnets
(power circuits, with each magnet's transfer function), and where the BPMs,
RF stations and tune monitors are. It is what the machine's device files
hold (``naming.yaml`` and ``devices/``; see :mod:`eicvibe.store`), so it
uses the lattice's ids and no derived names.

:func:`assemble_registry` turns it into a :class:`MachineRegistry`: every
element gets its EIC name by the naming rules unless the description names
it, devices and channels are named from their elements, and ids derive from
the description's keys and the lattice's ids, never from names. Renaming an
area therefore renames everything consistently and keeps every identity.
"""

import math
import re
from typing import Self

from pydantic import Field, model_validator

from ..lattice import Lattice
from ..lattice._base import DomainModel, Id, Label, Real
from ..lattice.ids import derived_id
from .naming import EicName, NamingArea, NamingScheme, non_lattice_name, slot_names
from .power import CircuitMagnet, PowerCircuit
from .registry import (
    Calibration,
    Channel,
    ChannelRole,
    CircuitBinding,
    Device,
    DeviceType,
    MachineRegistry,
    Observable,
    ObservableBinding,
    ParameterBinding,
    Slot,
)


class SharedName(DomainModel):
    """How a magnet that several branches pass through is named: once, at one traversal."""

    component: Id
    home: Label = Field(description="Role of the traversal whose position names it, e.g. circulation")
    abbreviation: Label = Field("sptm", description="Device abbreviation; sptm is a septum magnet")


class Naming(DomainModel):
    """The rules EIC names follow (EIC-SEG-RSI-009 Rev 02, :mod:`.naming`)."""

    areas: dict[Id, tuple[NamingArea, ...]] = Field(
        default_factory=dict, description="By branch id; a branch not listed is one area named after it"
    )
    supply: str = "ps"
    setpoint: str = "sp"
    readback: str = "rb"
    position_x: str = "rdx"
    position_y: str = "rdy"
    planes: tuple[str, str] = ("h", "v")
    rf_voltage: str = "volt"
    rf_phase: str = "phase"
    tune_monitor: str = "btun"
    tune_x: str = "rdx"
    tune_y: str = "rdy"
    shared: tuple[SharedName, ...] = ()
    names: dict[Id, EicName] = Field(
        default_factory=dict, description="Explicit element names replacing the rules', by occurrence id"
    )

    def scheme(self) -> NamingScheme:
        tokens = self.model_dump(exclude={"areas", "shared", "names"})
        return NamingScheme(**tokens)

    @classmethod
    def from_scheme(cls, scheme: NamingScheme, **fields) -> Self:
        tokens = {name: getattr(scheme, name) for name in cls.model_fields if hasattr(scheme, name)}
        return cls(**{**tokens, **fields})


class SupplyCircuit(DomainModel):
    """One circuit of a supply: its setting's range and the magnets it drives in series."""

    plane: str | None = Field(None, description="Signal token of a supply with several circuits, e.g. h")
    limits: tuple[Real, Real] = Field(description="Range of the setting, in engineering units")
    unit: str = "A"
    magnets: tuple[CircuitMagnet, ...] = Field(min_length=1)


class Supply(DomainModel):
    """A power supply: usually one circuit; a two-plane corrector has two."""

    key: str = Field(min_length=1, max_length=128, description="Its identity, e.g. a family or branch/magnet")
    name: EicName | None = Field(None, description="Its EIC name; by default its first magnet's, plus -ps")
    circuits: tuple[SupplyCircuit, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _planes(self) -> Self:
        planes = [c.plane for c in self.circuits]
        if len(planes) > 1 and (None in planes or len(set(planes)) != len(planes)):
            raise ValueError(f"supply {self.key}: several circuits need distinct planes")
        return self


class Bpm(DomainModel):
    occurrence: Id
    unit: str = "mm"
    scale: Real = Field(1e3, description="Channel units per metre")


class RfStation(DomainModel):
    occurrence: Id
    voltage: tuple[Real, Real] = Field(description="Range of the voltage setting, MV")
    phase: tuple[Real, Real] = Field((-180.0, 360.0), description="Range of the phase setting, deg")


class TuneMonitor(DomainModel):
    branch: Id


class MachineDescription(DomainModel):
    """Naming rules and devices of one machine, for one lattice."""

    lattice: Label
    naming: Naming = Field(default_factory=Naming)
    supplies: tuple[Supply, ...] = ()
    bpms: tuple[Bpm, ...] = ()
    rf: tuple[RfStation, ...] = ()
    tune_monitors: tuple[TuneMonitor, ...] = ()

    @model_validator(mode="after")
    def _unique_keys(self) -> Self:
        keys = [s.key for s in self.supplies]
        duplicates = sorted({k for k in keys if keys.count(k) > 1})
        if duplicates:
            raise ValueError(f"duplicate supply {', '.join(duplicates)}")
        return self


def default_area(lattice: Lattice, branch: str) -> NamingArea:
    """A whole branch as one area named after it, with positions from its start."""
    layout = lattice.layout(branch)
    prefix = re.sub(r"[^a-z0-9]", "", layout.branch.name.lower()).lstrip("0123456789") or "area"
    return NamingArea(prefix=prefix, start=0.0, end=layout.length + 1.0)


def element_names(lattice: Lattice, naming: Naming) -> tuple[Slot, ...]:
    """The EIC name of every element: by the rules, a shared magnet once, explicit names over both."""
    scheme = naming.scheme()
    slots: list[Slot] = []
    for branch in lattice.branches:
        areas = naming.areas.get(branch.id) or (default_area(lattice, branch.id),)
        placements = [p for p in lattice.layout(branch.id).placements if lattice.component_of(p.id) is None]
        names = slot_names(placements, areas, scheme.device)
        slots += [
            Slot(name=naming.names.get(p.id, names[p.id]), occurrences=(p.id,))
            for p in placements
            if p.id in names
        ]
    components = {c.id: c for c in lattice.shared_components}
    for shared in naming.shared:
        component = components.get(shared.component)
        if component is None:
            raise ValueError(f"naming: no shared component {shared.component!r}")
        home = component.traversal(shared.home).occurrence
        placed = lattice.placement(home)
        areas = naming.areas.get(placed.branch_id) or (default_area(lattice, placed.branch_id),)
        name = slot_names([placed], areas, lambda _, abbreviation=shared.abbreviation: abbreviation)[home]
        occurrences = tuple(t.occurrence for t in component.traversals)
        slots.append(Slot(name=naming.names.get(home, name), occurrences=occurrences))
    return tuple(slots)


def assemble_registry(lattice: Lattice, description: MachineDescription) -> MachineRegistry:
    """Element names, devices, channels and power circuits from a description."""
    if description.lattice != lattice.name:
        raise ValueError(f"the description is for lattice {description.lattice!r}, not {lattice.name!r}")
    naming, scheme = description.naming, description.naming.scheme()
    slots = element_names(lattice, naming)
    name_of = {o: s.name for s in slots for o in s.occurrences}
    devices: list[Device] = []
    channels: list[Channel] = []
    circuits: list[PowerCircuit] = []

    def device(name: str, kind: DeviceType, anchor: str, occurrences: tuple[str, ...] = ()) -> Device:
        made = Device(
            id=derived_id("dev", lattice.name, anchor, kind.value),
            name=name,
            type=kind,
            occurrences=occurrences,
        )
        devices.append(made)
        return made

    def setting(owner: Device, signals: tuple[str, ...], unit: str, limits, binding) -> None:
        name = ":".join((owner.name, scheme.setpoint, *signals))
        readback = ":".join((owner.name, scheme.readback, *signals))
        channels.extend(
            (
                Channel(
                    name=name,
                    device=owner.id,
                    role=ChannelRole.SETTING,
                    unit=unit,
                    control_limits=limits,
                    binding=binding,
                ),
                Channel(
                    name=readback, device=owner.id, role=ChannelRole.READBACK, unit=unit, readback_of=name
                ),
            )
        )

    # A shared magnet's hardware is at every traversal, including those its supply does not set.
    traversals = {t.occurrence: c for c in lattice.shared_components for t in c.traversals}

    def located(occurrence: str) -> tuple[str, ...]:
        component = traversals.get(occurrence)
        return tuple(t.occurrence for t in component.traversals) if component else (occurrence,)

    for supply in description.supplies:
        driven = (o for c in supply.circuits for m in c.magnets for o in m.occurrences)
        occurrences = tuple(dict.fromkeys(x for o in driven for x in located(o)))
        name = supply.name
        if name is None:
            first = supply.circuits[0].magnets[0].occurrences[0]
            if first not in name_of:
                raise ValueError(
                    f"supply {supply.key}: its first magnet has no EIC name; give the supply a name"
                )
            name = f"{name_of[first]}-{scheme.supply}"
        owner = device(name, DeviceType.POWER_SUPPLY, f"supply:{supply.key}", occurrences)
        for circuit in supply.circuits:
            label = supply.key if circuit.plane is None else f"{supply.key}:{circuit.plane}"
            made = PowerCircuit(
                id=derived_id("cir", lattice.name, supply.key, circuit.plane or ""),
                name=label,
                device=owner.id,
                magnets=circuit.magnets,
            )
            circuits.append(made)
            signals = () if circuit.plane is None else (circuit.plane,)
            setting(owner, signals, circuit.unit, circuit.limits, CircuitBinding(circuit=made.id))

    for bpm in description.bpms:
        name = _named(name_of, bpm.occurrence, "a BPM")
        owner = device(name, DeviceType.BPM, bpm.occurrence, (bpm.occurrence,))
        for signal, observable in (
            (scheme.position_x, Observable.ORBIT_X),
            (scheme.position_y, Observable.ORBIT_Y),
        ):
            binding = ObservableBinding(observable=observable, target=bpm.occurrence, scale=bpm.scale)
            channels.append(
                Channel(
                    name=f"{name}:{signal}",
                    device=owner.id,
                    role=ChannelRole.MEASUREMENT,
                    unit=bpm.unit,
                    binding=binding,
                )
            )

    for station in description.rf:
        owner = device(
            _named(name_of, station.occurrence, "an RF station"),
            DeviceType.RF_STATION,
            station.occurrence,
            (station.occurrence,),
        )
        volts = Calibration(coefficients=(0.0, 1e6), engineering_unit="MV", field_unit="V", normalized=False)
        degrees = Calibration(
            coefficients=(0.0, math.pi / 180), engineering_unit="deg", field_unit="rad", normalized=False
        )
        targets = (station.occurrence,)
        setting(
            owner,
            (scheme.rf_voltage,),
            "MV",
            station.voltage,
            ParameterBinding(targets=targets, parameter="RFP.voltage", calibration=volts),
        )
        setting(
            owner,
            (scheme.rf_phase,),
            "deg",
            station.phase,
            ParameterBinding(targets=targets, parameter="RFP.phase", calibration=degrees),
        )

    for monitor in description.tune_monitors:
        branch = lattice.layout(monitor.branch)
        areas = naming.areas.get(branch.id) or (default_area(lattice, branch.id),)
        machine = re.match(r"[a-z]+", areas[0].prefix)[0]
        owner = device(non_lattice_name(machine, scheme.tune_monitor), DeviceType.TUNE_MONITOR, branch.id)
        for signal, observable in ((scheme.tune_x, Observable.TUNE_X), (scheme.tune_y, Observable.TUNE_Y)):
            channels.append(
                Channel(
                    name=non_lattice_name(machine, scheme.tune_monitor, signals=(signal,)),
                    device=owner.id,
                    role=ChannelRole.MEASUREMENT,
                    binding=ObservableBinding(observable=observable, target=branch.id),
                )
            )

    return MachineRegistry(
        name=f"{lattice.name}-registry",
        lattice=lattice.name,
        slots=slots,
        devices=tuple(devices),
        channels=tuple(channels),
        circuits=tuple(circuits),
    )


def _named(name_of: dict[str, str], occurrence: str, what: str) -> str:
    if occurrence not in name_of:
        raise ValueError(f"{what} at occurrence {occurrence!r} has no EIC name")
    return name_of[occurrence]
