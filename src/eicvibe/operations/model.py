"""The machine as it is set, in one engine session: what changes of its setting channels do.

Operations work on setting channels (a corrector's current, a quadrupole
string's), not on lattice values: that is what is written to the machine, in
its units and limits, and a string or a trim moves every magnet it drives. The
model starts from the machine's values now (its mode's and the live ones) and
computes the orbit and optics with changes of some channels on top.
"""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np

from eicvibe.domain.machine.registry import DeviceType
from eicvibe.engines import engine as find_engine
from eicvibe.engines.contracts import Effect, OpticsRequest, OpticsResult, Options, OrbitRequest, OrbitResult
from eicvibe.engines.model import initial_conditions
from eicvibe.va.settings import MachineSettings

#: The parameters a corrector sets in each plane: a horizontal kick (Kn0L) or a vertical one (Ks0L).
CORRECTS = {"x": "MagneticMultipoleP.Kn0L", "y": "MagneticMultipoleP.Ks0L"}
#: The parameters of a quadrupole channel.
FOCUSES = ("MagneticMultipoleP.Kn1", "MagneticMultipoleP.Kn1L")


@dataclass(frozen=True)
class Monitor:
    """A BPM: its device name, the occurrence it reads at, and where."""

    name: str
    occurrence: str
    s: float


@dataclass(frozen=True)
class Knob:
    """A setting channel an operation may change: a corrector, or a quadrupole (string)."""

    channel: str
    device: str
    unit: str
    limits: tuple[float, float] | None
    occurrences: tuple[str, ...]
    """The occurrences it sets, in beam order."""
    s: float
    """Where the first of them is."""

    def step(self, value: float) -> float:
        """A change small enough to be linear and large enough to be well above round-off."""
        if self.limits:
            return 1e-4 * (self.limits[1] - self.limits[0])
        return 1e-4 * max(abs(value), 1.0)


class MachineModel:
    """One branch of a machine as it is set, in an engine session.

    ``live`` are the channels' values now (written on top of the mode's); ``changes`` in the
    methods are further channel values, taken back afterwards.
    """

    def __init__(
        self,
        settings: MachineSettings,
        branch: str,
        live: Mapping[str, float] | None = None,
        *,
        engine: str = "xsuite",
        options: Options | None = None,
    ) -> None:
        lattice = settings.lattice
        self.settings, self.lattice = settings, lattice
        self.layout = lattice.layout(branch)
        self.branch, self.name, self.periodic = self.layout.id, self.layout.branch.name, self.layout.periodic
        self.live = dict(live or {})
        # The model is the machine's deterministic response: closed orbits, optics, their changes. Quantum
        # excitation, being random, has no closed orbit (the engine's search fails), so it is left out; the
        # mean radiation stays, with the energy it takes.
        options = options or Options()
        self.options = Options(
            **(options.model_dump() | {"effects": options.effects - {Effect.QUANTUM_EXCITATION}})
        )
        here = {p.id for p in self.layout.placements}
        self._here = here
        self.position = {p.id: p.s_exit for p in self.layout.placements}
        self.order = {p.id: i for i, p in enumerate(self.layout.placements)}
        values = {k: v for k, v in settings.start_values(self.live).items() if k[0] in here}
        self._engine = find_engine(engine)
        self.engine = engine
        self._session = self._engine.open(lattice, self.branch, self.options, values)
        self.initial = None if self.periodic else initial_conditions(lattice, self.branch)
        self.bpms = sorted(
            (
                Monitor(d.name, d.occurrences[0], self.position[d.occurrences[0]])
                for d in settings.registry.devices
                if d.type is DeviceType.BPM and d.occurrences and d.occurrences[0] in here
            ),
            key=lambda m: m.s,
        )
        self.correctors = {plane: self._knobs((parameter,)) for plane, parameter in CORRECTS.items()}
        self.quadrupoles = self._knobs(FOCUSES)

    def _knobs(self, parameters: tuple[str, ...]) -> list[Knob]:
        out = []
        for name, setting in self.settings.settings.items():
            keys = [k for k in setting.keys if k[0] in self._here]
            if not keys or len(keys) != len(setting.keys) or any(p not in parameters for _, p in keys):
                continue
            occurrences = tuple(sorted({o for o, _ in keys}, key=self.order.__getitem__))
            out.append(
                Knob(
                    name,
                    setting.device,
                    setting.unit,
                    setting.limits,
                    occurrences,
                    self.position[occurrences[0]],
                )
            )
        return sorted(out, key=lambda k: k.s)

    # -- values ------------------------------------------------------------------------------------

    def value(self, channel: str) -> float:
        """A channel's value now."""
        return self.settings.value(channel, self.live)

    def knob(self, channel: str) -> Knob:
        for knob in (*self.correctors["x"], *self.correctors["y"], *self.quadrupoles):
            if knob.channel == channel:
                return knob
        raise KeyError(f"no corrector or quadrupole channel {channel} in {self.name}")

    @contextmanager
    def changed(self, changes: Mapping[str, float]) -> Iterator[None]:
        """The model with ``changes`` (channel -> value) on top of the values now, taken back after."""
        if not changes:
            yield
            return
        names = list(changes)
        self._session.update(self._branch_values(self.live | dict(changes), names))
        try:
            yield
        finally:
            self._session.update(self._branch_values(self.live, names))

    def _branch_values(self, live: Mapping[str, float], names: list[str]):
        values = self.settings.lattice_values(live, names)
        return {k: v for k, v in values.items() if k[0] in self._here}

    # -- what the beam does ----------------------------------------------------------------------

    def orbit_result(self, changes: Mapping[str, float] | None = None) -> OrbitResult:
        """The closed orbit (a ring) or the trajectory of a beam entering on axis (a line)."""
        initial = None if self.periodic else (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        with self.changed(changes or {}):
            return self._session.orbit(OrbitRequest(options=self.options, initial=initial))

    def orbit(self, changes: Mapping[str, float] | None = None) -> dict[str, np.ndarray]:
        """The orbit at the BPMs (m): ``x`` and ``y``, in the order of :attr:`bpms`."""
        result = self.orbit_result(changes)
        index = {o: i for i, o in enumerate(result.occurrences)}
        at = [index[m.occurrence] for m in self.bpms]
        return {p: np.asarray(getattr(result, p))[at] for p in ("x", "y")}

    def optics(self, changes: Mapping[str, float] | None = None) -> OpticsResult:
        """The optics: a ring's periodic solution, a line's from its initial conditions."""
        if not self.periodic and self.initial is None:
            raise ValueError(f"{self.name} is a line without initial conditions: no optics")
        with self.changed(changes or {}):
            return self._session.optics(OpticsRequest(options=self.options, initial=self.initial))

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> MachineModel:
        return self

    def __exit__(self, *_) -> None:
        self.close()
