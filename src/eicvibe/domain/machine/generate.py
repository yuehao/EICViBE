"""A registry generated from a lattice, with EIC names (ADR 0003, plan step 2).

The virtual accelerator needs devices before the real device list exists.
Devices are named from their lattice slots (EIC-SEG-RSI-009 Rev 02, see
:mod:`.naming`), with the tokens of a :class:`~.naming.NamingScheme`. With
the default scheme:

- every quadrupole, sextupole, octupole and corrector has its own supply,
  ``<slot>-ps``, with ``:sp`` and ``:rb`` (as the standard's example
  "es6-qf50_2-ps, the supply powering the second focusing quadrupole"); a
  two-plane corrector has ``:sp:h``, ``:sp:v`` and their readbacks;
- every BPM (Instrument) reads ``<slot>:rdx`` and ``<slot>:rdy``, in mm;
- every RF cavity has ``<slot>:sp:volt`` (MV), ``<slot>:sp:phase`` (deg)
  and their readbacks;
- a periodic branch has a tune monitor, ``<machine>-btun-rdx`` and ``-rdy``.

Device ids derive from the lattice (the occurrence a device sits at, or the
branch of a tune monitor), not from names, so renaming areas or tokens keeps
every device's identity. :meth:`MachineRegistry.naming_problems` reports the
tokens the naming repository does not list yet. Calibrations are linear
placeholders with typical magnitudes and the polarity of the design value,
so supplies run at positive currents; replace them with measured transfer
functions.
"""

import math
import re
from collections.abc import Iterable, Mapping

from ..lattice import SPECIES, ElementKind, Lattice
from ..lattice.conventions import rigidity_from_P
from ..lattice.ids import derived_id
from .naming import NamingArea, NamingScheme, non_lattice_name, slot_names
from .registry import (
    Calibration,
    Channel,
    ChannelRole,
    Device,
    DeviceType,
    MachineRegistry,
    Observable,
    ObservableBinding,
    ParameterBinding,
)

#: Placeholder field per ampere: T/m for a quadrupole, T/m^2 for a sextupole, T/m^3 for an octupole.
PLACEHOLDER_GRADIENT_PER_AMPERE = {1: 0.1, 2: 1.0, 3: 10.0}
#: Placeholder integrated corrector field per ampere, T m.
PLACEHOLDER_CORRECTOR_PER_AMPERE = 1e-3
#: Placeholder current ranges, A: magnets run to twice their design current or at least this.
PLACEHOLDER_MAGNET_CURRENT = 100.0
PLACEHOLDER_CORRECTOR_CURRENT = 50.0

_ORDER = {ElementKind.QUADRUPOLE: 1, ElementKind.SEXTUPOLE: 2, ElementKind.OCTUPOLE: 3}


def _field_unit(power: int) -> str:
    """T/m^power, the field unit of a multipole of that many transverse derivatives."""
    return {0: "T", 1: "T/m"}.get(power, f"T/m^{power}")


def generate_registry(
    lattice: Lattice,
    branch: str | None = None,
    *,
    areas: Iterable[NamingArea] | None = None,
    scheme: NamingScheme | None = None,
) -> MachineRegistry:
    """Devices and channels for one branch of ``lattice`` (the root branch by default).

    ``areas`` assign EIC area prefixes and clock positions along the branch,
    e.g. :func:`~eicvibe.domain.machine.naming.ring_sectors` for the ESR. By
    default the whole branch is one area named after the branch, with
    positions in metres from its start. The branch's BeginningEle must carry
    the reference particle, which scales the placeholder calibrations.
    """
    name = branch or lattice.root_branch
    return generate_machine_registry(lattice, {name: areas} if areas is not None else [name], scheme=scheme)


def generate_machine_registry(
    lattice: Lattice,
    branches: Mapping[str, Iterable[NamingArea] | None] | Iterable[str] | None = None,
    *,
    scheme: NamingScheme | None = None,
) -> MachineRegistry:
    """Devices and channels for several branches, every branch by default, in one registry.

    ``branches`` maps branch names to their naming areas (None: one area
    named after the branch), or lists branch names. Traversals of a shared
    component (e.g. a septum seen from a ring and from its transfer lines)
    get no device here: one piece of hardware must not appear as several, so
    such devices are described explicitly.
    """
    if branches is None:
        branches = [b.name for b in lattice.branches]
    if not isinstance(branches, Mapping):
        branches = dict.fromkeys(branches)
    builder = _Builder(lattice.name, scheme or NamingScheme())
    for name, areas in branches.items():
        _add_branch(builder, lattice, lattice.layout(name), areas)
    return MachineRegistry(
        name=f"{lattice.name}-generated",
        lattice=lattice.name,
        devices=tuple(builder.devices),
        channels=tuple(builder.channels),
    )


def _add_branch(builder: _Builder, lattice: Lattice, layout, areas: Iterable[NamingArea] | None) -> None:
    conditions = lattice.initial_conditions(layout.id)
    reference = conditions.reference if conditions else None
    if reference is None or reference.pc_ref is None or reference.species_ref not in SPECIES:
        raise ValueError(
            f"branch {layout.branch.name!r} needs a reference particle with pc_ref and a known species"
        )
    rigidity = rigidity_from_P(reference.pc_ref, charge=SPECIES[reference.species_ref].charge)
    if areas is None:
        prefix = re.sub(r"[^a-z0-9]", "", layout.branch.name.lower()).lstrip("0123456789") or "area"
        areas = (NamingArea(prefix=prefix, start=0.0, end=layout.length + 1.0),)
    areas = tuple(areas)
    slots = slot_names(layout.placements, areas, builder.scheme.device)
    for placement in layout.placements:
        slot = slots.get(placement.id)
        kind = placement.element.kind
        if slot is None or lattice.component_of(placement.id) is not None:
            continue
        if kind in _ORDER:
            builder.magnet(slot, placement, _ORDER[kind], rigidity)
        elif kind is ElementKind.KICKER:
            builder.corrector(slot, placement)
        elif kind is ElementKind.INSTRUMENT:
            builder.bpm(slot, placement)
        elif kind is ElementKind.RFCAVITY:
            builder.rf_station(slot, placement)
    if layout.periodic:
        builder.tune_monitor(re.match(r"[a-z]+", areas[0].prefix)[0], layout.branch)


class _Builder:
    def __init__(self, namespace: str, scheme: NamingScheme) -> None:
        self.namespace = namespace
        self.scheme = scheme
        self.devices: list[Device] = []
        self.channels: list[Channel] = []

    def device(self, name: str, kind: DeviceType, anchor: str, occurrences: tuple[str, ...] = ()) -> Device:
        """A device whose id derives from ``anchor`` (an occurrence or branch id), not from its name."""
        device_id = derived_id("dev", self.namespace, anchor, kind.value)
        device = Device(id=device_id, name=name, type=kind, occurrences=occurrences)
        self.devices.append(device)
        return device

    def setting(
        self,
        device: Device,
        signals: tuple[str, ...],
        unit: str,
        limits: tuple[float, float],
        binding: ParameterBinding,
    ) -> None:
        """A setting ``<device>:<sp>[:...]`` and its readback ``<device>:<rb>[:...]``."""
        name = ":".join((device.name, self.scheme.setpoint, *signals))
        readback = ":".join((device.name, self.scheme.readback, *signals))
        self.channels += [
            Channel(
                name=name,
                device=device.id,
                role=ChannelRole.SETTING,
                unit=unit,
                control_limits=limits,
                binding=binding,
            ),
            Channel(name=readback, device=device.id, role=ChannelRole.READBACK, unit=unit, readback_of=name),
        ]

    def supply(self, slot: str, placement) -> Device:
        name = f"{slot}-{self.scheme.supply}"
        return self.device(name, DeviceType.POWER_SUPPLY, placement.id, (placement.id,))

    def magnet(self, slot: str, placement, order: int, rigidity: float) -> None:
        element = placement.element
        group = element.MagneticMultipoleP
        integrated = group is not None and getattr(group, f"Kn{order}L") != 0.0
        parameter = f"Kn{order}L" if integrated else f"Kn{order}"
        design = getattr(group, parameter) if group is not None else 0.0
        polarity = -1.0 if design < 0 else 1.0
        # An integrated strength takes an integrated field; a thin magnet counts as 1 m long here.
        length = (element.length or 1.0) if integrated else 1.0
        per_ampere = PLACEHOLDER_GRADIENT_PER_AMPERE[order] * length
        calibration = Calibration(
            coefficients=(0.0, polarity * per_ampere),
            engineering_unit="A",
            field_unit=_field_unit(order - 1 if integrated else order),
        )
        top = max(2.0 * abs(design) * rigidity / per_ampere, PLACEHOLDER_MAGNET_CURRENT)
        limits = (0.0, top) if design != 0.0 else (-top, top)
        binding = ParameterBinding(
            targets=(placement.id,), parameter=f"MagneticMultipoleP.{parameter}", calibration=calibration
        )
        self.setting(self.supply(slot, placement), (), "A", limits, binding)

    def corrector(self, slot: str, placement) -> None:
        supply = self.supply(slot, placement)
        tags = placement.element.tags
        horizontal, vertical = self.scheme.planes
        planes = [((), "Kn0L")] if "HCOR" in tags else [((), "Ks0L")] if "VCOR" in tags else []
        planes = planes or [((horizontal,), "Kn0L"), ((vertical,), "Ks0L")]
        calibration = Calibration(
            coefficients=(0.0, PLACEHOLDER_CORRECTOR_PER_AMPERE), engineering_unit="A", field_unit="T m"
        )
        limits = (-PLACEHOLDER_CORRECTOR_CURRENT, PLACEHOLDER_CORRECTOR_CURRENT)
        for signals, parameter in planes:
            binding = ParameterBinding(
                targets=(placement.id,), parameter=f"MagneticMultipoleP.{parameter}", calibration=calibration
            )
            self.setting(supply, signals, "A", limits, binding)

    def bpm(self, slot: str, placement) -> None:
        bpm = self.device(slot, DeviceType.BPM, placement.id, (placement.id,))
        for signal, observable in (
            (self.scheme.position_x, Observable.ORBIT_X),
            (self.scheme.position_y, Observable.ORBIT_Y),
        ):
            binding = ObservableBinding(observable=observable, target=placement.id, scale=1e3)
            self.channels.append(
                Channel(
                    name=f"{slot}:{signal}",
                    device=bpm.id,
                    role=ChannelRole.MEASUREMENT,
                    unit="mm",
                    binding=binding,
                )
            )

    def rf_station(self, slot: str, placement) -> None:
        cavity = self.device(slot, DeviceType.RF_STATION, placement.id, (placement.id,))
        volts = Calibration(coefficients=(0.0, 1e6), engineering_unit="MV", field_unit="V", normalized=False)
        degrees = Calibration(
            coefficients=(0.0, math.pi / 180), engineering_unit="deg", field_unit="rad", normalized=False
        )
        rf = placement.element.RFP
        top = max(2.0 * (rf.voltage if rf else 0.0) / 1e6, 1.0)
        targets = (placement.id,)
        voltage = ParameterBinding(targets=targets, parameter="RFP.voltage", calibration=volts)
        phase = ParameterBinding(targets=targets, parameter="RFP.phase", calibration=degrees)
        self.setting(cavity, (self.scheme.rf_voltage,), "MV", (0.0, top), voltage)
        self.setting(cavity, (self.scheme.rf_phase,), "deg", (-180.0, 360.0), phase)

    def tune_monitor(self, machine: str, branch) -> None:
        name = non_lattice_name(machine, self.scheme.tune_monitor)
        monitor = self.device(name, DeviceType.TUNE_MONITOR, branch.id)
        for signal, observable in (
            (self.scheme.tune_x, Observable.TUNE_X),
            (self.scheme.tune_y, Observable.TUNE_Y),
        ):
            binding = ObservableBinding(observable=observable, target=branch.id)
            self.channels.append(
                Channel(
                    name=non_lattice_name(machine, self.scheme.tune_monitor, signals=(signal,)),
                    device=monitor.id,
                    role=ChannelRole.MEASUREMENT,
                    binding=binding,
                )
            )
