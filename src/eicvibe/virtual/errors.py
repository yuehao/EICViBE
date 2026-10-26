"""A virtual machine's errors: what makes it differ from its model, as a real machine does (ADR 0011).

No magnet is where, or as strong as, the design says, no power supply holds its current exactly,
and no BPM reads the beam where it is. :class:`ErrorSpec` says which errors a machine has and how
large they are; :class:`MachineErrors` draws them once, from the spec's seed, for the branches a
run tracks:

- static errors of magnets chosen by kind, branch and name (:class:`MagnetErrors`): the body's
  offsets, roll, pitch and yaw (PALS ``BodyShiftP``, about its centre) and its main field's
  relative error (a bend's dipole field, a quadrupole's gradient, a corrector's calibration), each
  a Gaussian of the rms given, cut at a few sigma, as alignment and magnetic measurement leave them;
- power supplies whose current fluctuates (:class:`SupplyNoise`): noise of a bandwidth, and ripple
  lines, every magnet of one supply following the same current, turn by turn in a ring and shot
  by shot along a route;
- the BPMs' own errors (:class:`BpmModel`): offsets, gains and rolls, and the noise of their
  turn-by-turn and closed-orbit acquisitions.

The settings keep the design and what people set. The errors act only where values enter the
engine (:meth:`MachineErrors.engine_values`), so the settings, the operators' model and the log
never show them, as on the real machine; a run's ``hidden.json`` keeps what was drawn.
"""

import fnmatch
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Self

import numpy as np
from pydantic import Field

from eicvibe.diagnostics.bpm import NO_BEAM, VALID
from eicvibe.domain.lattice import ElementKind, Lattice, PlacedElement, strength_parameter
from eicvibe.domain.lattice._base import DomainModel
from eicvibe.domain.machine import MachineRegistry
from eicvibe.domain.machine.power import design_value

type Key = tuple[str, str]
type Values = Mapping[Key, float]


class _Spec(DomainModel):
    """Frozen and validated, as the lattice's data are; a changed copy is validated again."""

    def replaced(self, **changes: Any) -> Self:
        return type(self).model_validate(self.model_dump() | changes)


#: The kinds that are magnets: what errors are drawn for.
MAGNETS = (
    ElementKind.BEND,
    ElementKind.QUADRUPOLE,
    ElementKind.SEXTUPOLE,
    ElementKind.OCTUPOLE,
    ElementKind.MULTIPOLE,
    ElementKind.KICKER,
    ElementKind.SOLENOID,
)

#: The body's errors and the BodyShiftP parameter each moves.
SHIFTS = {
    "x_offset": "BodyShiftP.x_offset",
    "y_offset": "BodyShiftP.y_offset",
    "roll": "BodyShiftP.z_rot",
    "pitch": "BodyShiftP.x_rot",
    "yaw": "BodyShiftP.y_rot",
}

_ORDER = {ElementKind.BEND: 0, ElementKind.QUADRUPOLE: 1, ElementKind.SEXTUPOLE: 2, ElementKind.OCTUPOLE: 3}


def main_fields(placed: PlacedElement) -> tuple[str, ...]:
    """The parameters of a magnet's main field, which its field error scales and its supply drives.

    A bend's dipole field, a quadrupole's gradient, a sextupole's and an octupole's, each in the form
    the lattice gives it (normalized or field, per length or integrated); a corrector's two planes;
    a solenoid's strength or field. A thin multipole has no one main field.
    """
    group = placed.element.MagneticMultipoleP
    if placed.kind in _ORDER:
        return (f"MagneticMultipoleP.{strength_parameter(group, _ORDER[placed.kind])}",)
    if placed.kind is ElementKind.KICKER:
        normal, skew = strength_parameter(group, 0), strength_parameter(group, 0, normal=False)
        integrated = not normal.endswith("L")  # a corrector at zero: its kicks are integrated
        letter = normal[0]
        if integrated:
            normal, skew = f"{letter}n0L", f"{letter}s0L"
        return (f"MagneticMultipoleP.{normal}", f"MagneticMultipoleP.{skew}")
    if placed.kind is ElementKind.SOLENOID:
        solenoid = placed.element.SolenoidP
        return ("SolenoidP.Bsol" if solenoid is not None and solenoid.Bsol else "SolenoidP.Ksol",)
    return ()


class Selection(_Spec):
    """Which magnets: of these kinds, in these branches, with names matching these patterns; empty: any."""

    kinds: tuple[ElementKind, ...] = ()
    branches: tuple[str, ...] = Field((), description="Branch names")
    names: tuple[str, ...] = Field((), description="Element names, shell patterns: 'qf*', 'q?_1'")

    def describe(self) -> str:
        """'Quadrupole or Bend named qf* in rcs': what it chooses, in words."""
        what = " or ".join(k.value for k in self.kinds) or "magnet"
        named = f" named {' or '.join(self.names)}" if self.names else ""
        where = f" in {' or '.join(self.branches)}" if self.branches else ""
        return what + named + where

    def matches(self, lattice: Lattice, placed: PlacedElement) -> bool:
        if placed.kind not in MAGNETS or (self.kinds and placed.kind not in self.kinds):
            return False
        if self.branches and lattice.branch(placed.branch_id).name not in self.branches:
            return False
        return not self.names or any(fnmatch.fnmatchcase(placed.name.lower(), p.lower()) for p in self.names)


class MagnetErrors(_Spec):
    """Static errors of some magnets, each the rms of a Gaussian cut at ``cut`` sigma, drawn per magnet."""

    select: Selection = Field(default_factory=Selection)
    x_offset: float = Field(0.0, ge=0, description="Horizontal offset of the body's centre, m rms")
    y_offset: float = Field(0.0, ge=0, description="Vertical offset, m rms")
    roll: float = Field(0.0, ge=0, description="Roll about the body's axis (z_rot), rad rms")
    pitch: float = Field(0.0, ge=0, description="Pitch about its centre (x_rot), rad rms")
    yaw: float = Field(0.0, ge=0, description="Yaw about its centre (y_rot), rad rms")
    field: float = Field(0.0, ge=0, description="Relative error of the main field, rms")
    cut: float = Field(2.5, gt=0, description="Draws beyond this many sigma are drawn again")


class Ripple(_Spec):
    """A line of a power supply's ripple: its frequency and relative amplitude (of the current)."""

    frequency: float = Field(gt=0, description="Hz")
    amplitude: float = Field(ge=0, description="Relative, peak")


class SupplyNoise(_Spec):
    """Power supplies whose current fluctuates about its setting; every magnet of a supply follows it.

    Noise of ``rms`` (relative to the current), white, or of a first-order ``bandwidth``; and ripple
    lines. In a ring it moves turn by turn, at the ring's revolution period (sampled at least twenty
    times per period of its fastest part); along a route, one value per shot.
    """

    select: Selection = Field(default_factory=lambda: Selection(kinds=(ElementKind.BEND,)))
    rms: float = Field(0.0, ge=0, description="Relative rms of the current's noise")
    bandwidth: float | None = Field(
        None, gt=0, description="Hz: the noise's first-order low-pass corner; None: a new value every turn"
    )
    ripple: tuple[Ripple, ...] = ()


class BpmModel(_Spec):
    """How the virtual BPMs read the beam: a reading is gain x (roll of (centroid - offset)) + noise.

    Each BPM has its own hidden offset, roll and gain. Turn by turn it adds ``resolution`` noise (more
    as the beam thins); its closed-orbit acquisition, a narrow-band average of many turns, adds
    ``orbit_resolution`` to every sample instead.
    """

    resolution: float = Field(0.005, ge=0, description="Turn-by-turn noise, mm rms at full intensity")
    orbit_resolution: float = Field(0.0003, ge=0, description="Closed-orbit (slow acquisition) noise, mm rms")
    offset_rms: float = Field(0.1, ge=0, description="Hidden offsets of the BPMs, mm rms")
    gain_rms: float = Field(0.01, ge=0, description="Hidden relative gain errors, rms")
    roll_rms: float = Field(0.0, ge=0, description="Hidden rolls of the BPMs, rad rms")
    seed: int = 7


class ErrorSpec(_Spec):
    """Which errors a machine has, and how large: drawn from ``seed``, the same every time it starts."""

    seed: int | None = Field(
        None, description="Draws every error; None (a machine from before errors): the BPMs' own seed"
    )
    magnets: tuple[MagnetErrors, ...] = ()
    supplies: tuple[SupplyNoise, ...] = ()
    bpms: BpmModel = Field(default_factory=BpmModel)

    def bpm_model(self) -> BpmModel:
        """The BPMs' model with their errors drawn from the spec's seed."""
        if self.seed is None:
            return self.bpms
        return self.bpms.replaced(seed=int(np.random.SeedSequence([self.seed, 7]).generate_state(1)[0]))


def _cut_normal(rng: np.random.Generator, rms: float, cut: float, count: int) -> np.ndarray:
    out = rng.normal(0.0, rms, count)
    while (far := np.abs(out) > cut * rms).any():
        out[far] = rng.normal(0.0, rms, int(far.sum()))
    return out


@dataclass(frozen=True)
class Supply:
    """A fluctuating power supply: the settings it drives, and how its current moves."""

    name: str
    keys: tuple[Key, ...]
    noise: SupplyNoise
    seed: tuple[int, ...]


class MachineErrors:
    """The errors drawn for one machine on the branches a run tracks (``branches``: ids or names).

    ``registry`` groups magnets by the power supplies that drive them, for :class:`SupplyNoise`;
    without one, every magnet has a supply of its own.
    """

    def __init__(
        self,
        spec: ErrorSpec,
        lattice: Lattice,
        branches: Iterable[str],
        registry: MachineRegistry | None = None,
    ) -> None:
        self.spec = spec
        self.lattice = lattice
        seed = spec.seed or 0
        placed: dict[str, PlacedElement] = {}
        for branch in branches:
            placed |= {p.id: p for p in lattice.layout(branch).placements}
        self._placed = placed
        self.shifts: dict[Key, float] = {}
        """Added to the BodyShiftP parameters (m, rad)."""
        self.factors: dict[Key, float] = {}
        """Multiply the main fields' parameters: 1 + the field error."""
        self.chosen: dict[str, list[int]] = {"magnets": [], "supplies": []}
        """How many magnets each row of magnet errors chose here, and how many supplies each row of
        supply noise."""
        self.problems: list[str] = []
        """Rows that chose nothing here: a choice that names no magnet is a mistake, not no errors."""
        for i, errors in enumerate(spec.magnets):
            chosen = [p for p in placed.values() if errors.select.matches(lattice, p)]
            self.chosen["magnets"].append(len(chosen))
            if not chosen:
                self.problems.append(f"magnet errors, row {i + 1}: no {errors.select.describe()} here")
            for j, (name, path) in enumerate(SHIFTS.items()):
                rms = getattr(errors, name)
                if rms and chosen:
                    draws = _cut_normal(np.random.default_rng([seed, i, j]), rms, errors.cut, len(chosen))
                    for p, d in zip(chosen, draws, strict=True):
                        self.shifts[(p.id, path)] = self.shifts.get((p.id, path), 0.0) + float(d)
            keys = [(p.id, path) for p in chosen for path in main_fields(p)]
            if errors.field and keys:
                draws = _cut_normal(
                    np.random.default_rng([seed, i, len(SHIFTS)]), errors.field, errors.cut, len(keys)
                )
                for key, d in zip(keys, draws, strict=True):
                    self.factors[key] = self.factors.get(key, 1.0) * (1.0 + float(d))
        self.supplies: list[Supply] = []
        for i, noise in enumerate(spec.supplies):
            groups: dict[str, list[Key]] = {}
            for p in placed.values():
                if not noise.select.matches(lattice, p):
                    continue
                for path in main_fields(p):
                    circuits = [
                        c.name
                        for c, m in (registry.circuits_at(p.id) if registry else ())
                        if m.parameter == path
                    ]
                    groups.setdefault(circuits[0] if circuits else f"{p.name} {path}", []).append(
                        (p.id, path)
                    )
            self.chosen["supplies"].append(len(groups))
            if not groups:
                self.problems.append(f"supply noise, row {i + 1}: no {noise.select.describe()} here")
            if not (noise.rms or any(r.amplitude for r in noise.ripple)):
                continue
            for k, (name, keys) in enumerate(groups.items()):
                self.supplies.append(Supply(name, tuple(keys), noise, (seed, 1000 + i, k)))

    @property
    def empty(self) -> bool:
        return not (self.shifts or self.factors or self.supplies)

    def engine_values(self, values: Values, *, complete: bool = False) -> dict[Key, float]:
        """``values`` (settings) as the machine has them: field errors scale, body errors add.

        ``complete``: also every parameter with an error that ``values`` does not give, from the
        lattice's value (a session opened); else only those given (an update).
        """
        out = dict(values)
        for key, factor in self.factors.items():
            if key in out:
                out[key] *= factor
            elif complete:
                out[key] = design_value(self._placed[key[0]], key[1]) * factor
        for key, shift in self.shifts.items():
            if key in out:
                out[key] += shift
            elif complete:
                out[key] = design_value(self._placed[key[0]], key[1]) + shift
        return out

    def fluctuations(self, period: float) -> Fluctuations | None:
        """The supplies' noise for a run whose turns (or shots) are ``period`` seconds apart."""
        return Fluctuations(self.supplies, period) if self.supplies else None

    def hidden(self) -> dict[str, Any]:
        """What was drawn, by magnet and supply: the truth a run keeps hidden."""
        magnets: dict[str, dict[str, Any]] = {}
        for (occurrence, path), shift in self.shifts.items():
            entry = magnets.setdefault(occurrence, {"name": self._placed[occurrence].name})
            entry[next(k for k, v in SHIFTS.items() if v == path)] = shift
        for (occurrence, path), factor in self.factors.items():
            entry = magnets.setdefault(occurrence, {"name": self._placed[occurrence].name})
            entry.setdefault("field", {})[path.split(".", 1)[1]] = factor - 1.0
        return {
            "seed": self.spec.seed,
            "magnets": magnets,
            "supplies": [{"supply": s.name, "settings": len(s.keys)} for s in self.supplies],
        }


class Fluctuations:
    """The supplies' relative deviations from their settings, turn after turn (or shot after shot).

    Each supply's current is sampled every ``step`` turns, at least twenty times per period of its
    fastest part (every turn for white noise), and held in between; the samples follow on from call
    to call, so a run's turns may be asked for in any chunks, in order.
    """

    def __init__(self, supplies: list[Supply], period: float) -> None:
        self.supplies = supplies
        self.period = period
        self._state: list[dict[str, Any]] = []
        for supply in supplies:
            noise = supply.noise
            fastest = max([noise.bandwidth or 0.0] + [r.frequency for r in noise.ripple])
            if noise.rms and noise.bandwidth is None:
                step = 1
            else:
                step = max(1, int(1.0 / (20.0 * fastest * period))) if fastest and period > 0 else 1
            rng = np.random.default_rng(list(supply.seed))
            self._state.append(
                {
                    "rng": rng,
                    "step": step,
                    "phases": rng.uniform(0.0, 2 * math.pi, len(noise.ripple)),
                    "decay": math.exp(-2 * math.pi * noise.bandwidth * step * period)
                    if noise.bandwidth
                    else 0.0,
                    "samples": np.zeros(0),
                    "first": 0,
                    "noise": None,
                }
            )

    def turns(self, first: int, count: int) -> list[tuple[Supply, np.ndarray]]:
        """Each supply and its relative deviation before each turn of ``[first, first + count)``."""
        out = []
        turns = np.arange(first, first + count)
        for supply, state in zip(self.supplies, self._state, strict=True):
            index = turns // state["step"]
            self._extend(supply, state, int(index[-1]) + 1 if count else 0)
            out.append((supply, state["samples"][index - state["first"]]))
            keep = (
                int(index[-1]) - state["first"] if count else 0
            )  # the last sample is held into the next call
            state["samples"], state["first"] = state["samples"][keep:], state["first"] + keep
        return out

    def _extend(self, supply: Supply, state: dict[str, Any], end: int) -> None:
        """Samples up to index ``end`` (exclusive): the noise's next values, plus the ripple's."""
        have = state["first"] + len(state["samples"])
        if end <= have:
            return
        noise, rng, n = supply.noise, state["rng"], end - have
        values = np.empty(n)
        x = state["noise"]
        for k in range(n):
            z = rng.standard_normal()
            if x is None or not noise.bandwidth:
                x = noise.rms * z
            else:
                x = state["decay"] * x + math.sqrt(1 - state["decay"] ** 2) * noise.rms * z
            values[k] = x
        state["noise"] = x
        times = np.arange(have, end) * state["step"] * self.period
        for line, phase in zip(noise.ripple, state["phases"], strict=True):
            values += line.amplitude * np.sin(2 * math.pi * line.frequency * times + phase)
        state["samples"] = np.concatenate([state["samples"], values])


class VirtualBpms:
    """The BPMs of a run: each with a hidden offset, roll and gain error, reading with noise.

    ``names`` (one per reading) gives a BPM read more than once (a route passing it on two turns)
    one offset, roll and gain: a reading per pass, the errors of the one BPM.
    """

    def __init__(
        self, model: BpmModel, count: int, particles: int, charge: float, names: list[str] | None = None
    ) -> None:
        names = names if names is not None else [str(k) for k in range(count)]
        unique = list(dict.fromkeys(names))
        self._slot = np.array([unique.index(n) for n in names], dtype=int)
        self._unique = len(unique)
        self.particles = particles
        self.charge = charge
        self.earlier: list[dict[str, Any]] = []
        """The errors read with before each :meth:`remodel`, and the turn (or shot) they ended."""
        self._draw(model)
        self._noise = np.random.default_rng(model.seed + 1)
        self._orbit = np.random.default_rng(model.seed + 2)

    def _draw(self, model: BpmModel) -> None:
        rng = np.random.default_rng(model.seed)
        self.model = model
        self.offsets = rng.normal(0.0, model.offset_rms, (2, self._unique))[:, self._slot]
        self.gains = (1.0 + rng.normal(0.0, model.gain_rms, (2, self._unique)))[:, self._slot]
        self.rolls = rng.normal(0.0, model.roll_rms, self._unique)[self._slot]

    def remodel(self, model: BpmModel, until: int) -> None:
        """Read with ``model`` from now on, the errors so far ending at turn (or shot) ``until``: an
        administrator's change on a running machine, whose beam it leaves alone. Drawn from the model's
        seed as before, so with the same seed each BPM keeps its offset, gain and roll, scaled to the
        new rms."""
        self.earlier.append(self._drawn() | {"until": until})
        self._draw(model)

    def _drawn(self) -> dict[str, Any]:
        return {
            "offsets_mm": self.offsets.tolist(),
            "gains": self.gains.tolist(),
            "rolls_rad": self.rolls.tolist(),
        }

    def hidden(self, names: list[str]) -> dict[str, Any]:
        """The errors the BPMs read with now, and those they read with before (``earlier``), if changed."""
        return self._drawn() | {"bpms": names} | ({"earlier": self.earlier} if self.earlier else {})

    def systematic(self, x: np.ndarray, y: np.ndarray, count: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """What the BPMs read of centroids ``[bpm, n]`` (m) without noise: mm, through their errors."""
        u = 1e3 * x - self.offsets[0][:, None]
        v = 1e3 * y - self.offsets[1][:, None]
        c, s = np.cos(self.rolls)[:, None], np.sin(self.rolls)[:, None]
        rx, ry = self.gains[0][:, None] * (c * u + s * v), self.gains[1][:, None] * (c * v - s * u)
        return np.where(count > 0, rx, np.nan), np.where(count > 0, ry, np.nan)

    def read(self, x: np.ndarray, y: np.ndarray, count: np.ndarray) -> tuple[np.ndarray, ...]:
        """Turn-by-turn readings of centroids ``[bpm, n]`` (m): mm, through the BPMs' errors, noisier as
        the beam thins; with the intensity (nC) and status of each."""
        with np.errstate(divide="ignore", invalid="ignore"):
            noise = self.model.resolution * np.sqrt(self.particles / count)
        readings = [
            value + noise * self._noise.standard_normal(value.shape) for value in self.systematic(x, y, count)
        ]
        intensity = self.charge * 1e9 * count / self.particles
        status = np.where(count > 0, VALID, NO_BEAM)
        return readings[0], readings[1], intensity, status

    def orbit_noise(self, bpms: int) -> np.ndarray:
        """The closed-orbit acquisition's noise of one sample, ``[plane, bpm]`` (mm)."""
        return self.model.orbit_resolution * self._orbit.standard_normal((2, bpms))
