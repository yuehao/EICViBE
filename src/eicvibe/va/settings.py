"""A machine's setpoints in engineering units, and the lattice values they make (ADR 0008).

The machine registry names every setting channel (``ltr-q1:1-ps:sp`` in A,
``bar-cav0:sp:volt`` in MV). Its original value is what the machine files give,
design and mode: the current at which a supply's magnets have their lattice (or
mode) strength, through their transfer functions. What users write is the live
layer on top. A channel's lattice values come from its power circuit (every
magnet of a string, trims added) or its calibration; channels nobody changed
keep the lattice's values exactly.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from eicvibe.domain.lattice import Lattice
from eicvibe.domain.machine import MachineRegistry
from eicvibe.domain.machine.power import design_value, magnet_current, rigidity, total_strengths
from eicvibe.domain.machine.registry import Channel, ChannelRole

Values = dict[tuple[str, str], float]

#: Where an original current is looked for when a channel gives no limits.
WIDE = (-1e7, 1e7)


@dataclass(frozen=True)
class Setting:
    """One setting channel: what it sets, its limits, and its original value."""

    name: str
    unit: str
    device: str
    kind: str
    """``circuit`` (a supply's current) or ``parameter`` (e.g. an RF voltage)."""
    limits: tuple[float, float] | None
    original: float
    keys: tuple[tuple[str, str], ...]
    """The (occurrence, parameter) pairs it sets."""
    source: str | None = None
    """Where a supply's transfer functions come from, the least sure of its magnets': placeholder,
    design or measured (ADR 0014); None for a channel that is not a supply's current."""

    @property
    def beyond(self) -> bool:
        """Whether its original value (its magnets' design current) is beyond its limits: a design its
        supply cannot run (ADR 0014)."""
        if self.limits is None:
            return False
        low, high = self.limits
        margin = 1e-9 * max(abs(low), abs(high), 1.0)
        return not low - margin <= self.original <= high + margin


class MachineSettings:
    """The setting channels of a machine on a lattice, with the mode's values as their origin."""

    def __init__(self, lattice: Lattice, registry: MachineRegistry, mode: Mapping[tuple[str, str], float]):
        self.lattice = lattice
        self.registry = registry
        self.mode = dict(mode)
        devices = {d.id: d.name for d in registry.devices}
        self._channels: dict[str, Channel] = {}
        self.settings: dict[str, Setting] = {}
        for channel in registry.channels:
            if channel.role is not ChannelRole.SETTING or channel.binding is None:
                continue
            keys = self._keys(channel)
            if not keys:
                continue
            self._channels[channel.name] = channel
            self.settings[channel.name] = Setting(
                name=channel.name,
                unit=channel.unit,
                device=devices.get(channel.device, channel.device),
                kind=channel.binding.kind,
                limits=tuple(channel.control_limits) if channel.control_limits else None,
                original=_clean(self._original(channel)),
                keys=keys,
                source=self._source(channel),
            )
        self._by_key: dict[tuple[str, str], list[str]] = {}
        for name, setting in self.settings.items():
            for key in setting.keys:
                self._by_key.setdefault(key, []).append(name)

    # -- reading the registry ----------------------------------------------------------------

    def _keys(self, channel: Channel) -> tuple[tuple[str, str], ...]:
        binding = channel.binding
        if binding.kind == "circuit":
            circuit = self.registry.circuit(binding.circuit)
            return tuple((o, m.parameter) for m in circuit.magnets for o in m.occurrences)
        if binding.kind == "parameter":
            return tuple((o, binding.parameter) for o in self._occurrences(binding.targets))
        return ()

    def _occurrences(self, targets: Iterable[str]) -> list[str]:
        """Occurrence ids of binding targets, which are occurrences or definitions."""
        out = []
        placements = self.lattice.topology.placements
        for target in targets:
            if target in placements:
                out.append(target)
            else:
                out += [p.id for p in placements.values() if p.occurrence.element == target]
        return out

    def base(self, key: tuple[str, str]) -> float:
        """The value the machine files give: the mode's, else the lattice's."""
        if key in self.mode:
            return self.mode[key]
        return design_value(self.lattice.placement(key[0]), key[1])

    def _source(self, channel: Channel) -> str | None:
        """The least sure source of a supply circuit's transfer functions."""
        if channel.binding.kind != "circuit":
            return None
        sources = {m.transfer.source.value for m in self.registry.circuit(channel.binding.circuit).magnets}
        return next(s for s in ("placeholder", "design", "measured") if s in sources)

    def _original(self, channel: Channel) -> float:
        binding = channel.binding
        if binding.kind == "parameter":
            occurrence = self._occurrences(binding.targets)[0]
            value = self.base((occurrence, binding.parameter))
            calibration = binding.calibration
            if calibration is None:
                return value
            rho = (
                rigidity(self.lattice, self.lattice.placement(occurrence).branch_id)
                if calibration.normalized
                else None
            )
            low, high = channel.control_limits or WIDE
            try:
                return calibration.to_engineering(value, low, high, rho)
            except ValueError:
                return calibration.to_engineering(value, *WIDE, rho)
        circuit = self.registry.circuit(binding.circuit)
        magnet = circuit.magnets[0]
        occurrence = magnet.occurrences[0]
        drivers = [c for c, m in self.registry.circuits_at(occurrence) if m.parameter == magnet.parameter]
        if len(drivers) > 1 and not circuit.series:
            return 0.0  # a trim on a string: the string carries the magnet's strength
        value = self.base((occurrence, magnet.parameter))
        for limits in (channel.control_limits or WIDE, WIDE):
            try:
                return magnet_current(self.lattice, magnet, value, tuple(limits), occurrence=occurrence)
            except ValueError:
                continue
        raise ValueError(f"{channel.name}: no current gives {magnet.parameter} = {value}")

    # -- values -----------------------------------------------------------------------------

    def value(self, name: str, live: Mapping[str, float]) -> float:
        """A channel's value now: the live one, else the original."""
        return live.get(name, self.settings[name].original)

    def check(self, name: str, value: float) -> str | None:
        """Why ``value`` cannot be written to channel ``name``, or None."""
        if name not in self.settings:
            return f"no setting channel {name}"
        limits = self.settings[name].limits
        if limits and not limits[0] <= value <= limits[1]:
            return (
                f"{value:g} {self.settings[name].unit} is outside the limits [{limits[0]:g}, {limits[1]:g}]"
            )
        return None

    def lattice_values(self, live: Mapping[str, float], names: Iterable[str] | None = None) -> Values:
        """Lattice values of the parameters the channels ``names`` set (all, when None), with ``live`` on top.

        A parameter that no live channel sets keeps its base value exactly; one driven by several
        circuits (a string and its trim) is their sum.
        """
        names = list(self.settings if names is None else names)
        keys = {k for n in names for k in self.settings[n].keys}
        out: Values = {}
        for key in keys:
            drivers = self._by_key.get(key, [])
            if not any(d in live for d in drivers):
                out[key] = self.base(key)
                continue
            out[key] = self._from_channels(key, drivers, live)
        return out

    def start_values(self, live: Mapping[str, float]) -> Values:
        """What a run starts with: the mode's values, and every live channel's."""
        return self.mode | self.lattice_values(live, live.keys())

    def _from_channels(self, key: tuple[str, str], drivers: list[str], live: Mapping[str, float]) -> float:
        circuits, currents, total = [], {}, 0.0
        for name in drivers:
            channel = self._channels[name]
            value = self.value(name, live)
            if channel.binding.kind == "circuit":
                circuit = self.registry.circuit(channel.binding.circuit)
                circuits.append(circuit)
                currents[circuit.id] = value
            else:
                calibration = channel.binding.calibration
                if calibration is None:
                    total += value
                else:
                    rho = (
                        rigidity(self.lattice, self.lattice.placement(key[0]).branch_id)
                        if calibration.normalized
                        else None
                    )
                    total += calibration.to_model(value, rho)
        if circuits:
            total += total_strengths(self.lattice, circuits, currents).get(key, 0.0)
        return total

    def readback(self, name: str, live: Mapping[str, float]) -> float:
        return self.value(name, live)


def _clean(value: float) -> float:
    """Round-off of a conversion (a phase of 0 rad in degrees: -1.6e-16) is zero."""
    return 0.0 if abs(value) < 1e-12 else value
