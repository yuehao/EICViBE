"""A synchrotron's ramp: its energy and RF in time, as its designers give them (ADR 0013).

The EIN's RCS takes the beam at 750 MeV and ramps it to 9 GeV in 108 ms (the RCS parameter
database's sheets "9GeV Lat" and "RF Ramp 9GeV Lat"). A :class:`RampProfile` holds what its
designers decided:

- the energy program (:class:`EnergyRamp`), their waveform in turns: a smooth start, quartic in
  the turn, up to their largest ramp rate; a linear ramp at that rate through the intrinsic spin
  resonances; a smooth end, quartic again, into the extraction energy. The joins are continuous to
  the second derivative. It is kept as their parameters and evaluated turn by turn exactly as their
  sheet does, not as samples of it;
- the RF program (:class:`RfProgram`): the cavities' total voltage and the synchronous phase in
  time from the ramp's start, as they computed them (with the radiation loss), interpolated
  linearly and held at its ends;
- the injection porch and the flattop.

Energies are total energies (eV). A turn is the revolution period at the speed of light, as in the
designers' sheet (electrons: within 2e-7 at 750 MeV). Magnets keep their normalized strengths
through the ramp (perfect tracking), so their fields, and their supplies' currents, follow the
energy.
"""

import bisect
import math
from dataclasses import dataclass
from typing import Any, Literal, Self

from pydantic import Field, model_validator

from ..lattice import SPECIES, ElementKind, Lattice, PlacedElement
from ..lattice._base import DomainModel, Label, PositiveInt, PositiveReal, Real

C_LIGHT = 299_792_458.0  # m/s


@dataclass(frozen=True)
class _Shape:
    """The waveform's coefficients for one revolution period (the designers' names in brackets)."""

    a: float  # the injection energy over the extraction energy [a]
    rate: float  # the linear part's slope, per turn, in units of the extraction energy [R]
    b: float  # the smooth start's cubic and quartic coefficients [b, c]
    c: float
    a1: float  # where the linear part starts [a1]
    start: float  # turns of the smooth start [N1]
    linear: float  # turns of the linear part [dT2]
    end: float  # turns of the smooth end [dT3]
    b2: float  # the smooth end's coefficients [b2, c2]
    c2: float

    @property
    def turns(self) -> float:
        return self.start + self.linear + self.end


class EnergyRamp(DomainModel):
    """The designers' energy waveform: a smooth start, a linear ramp, a smooth end (see the module)."""

    waveform: Literal["smooth-linear-smooth"] = "smooth-linear-smooth"
    injection: PositiveReal = Field(description="Total energy at injection, eV [E0]")
    extraction: PositiveReal = Field(description="Total energy at extraction, eV [Ef]")
    linear_end: PositiveReal = Field(
        description="Where the linear ramp ends, eV: past the last intrinsic spin resonance [EendLR]"
    )
    rate: PositiveReal = Field(description="The linear ramp's rate, eV/s [RE]")
    smooth_turns: PositiveInt = Field(description="Turns of the smooth start up to the linear rate [N1]")

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if not self.injection < self.linear_end < self.extraction:
            raise ValueError("energies must rise: injection < linear_end < extraction")
        return self

    def shape(self, period: float) -> _Shape:
        """The coefficients for a revolution period (s), as the designers' sheet derives them."""
        ef, n1 = self.extraction, float(self.smooth_turns)
        a = self.injection / ef
        rate = self.rate * period / ef
        b = rate / n1**2
        c = -b / (2.0 * n1)  # no curvature where the linear part begins
        a1 = a + b * n1**3 + c * n1**4
        linear = (self.linear_end / ef - a1) / rate
        if linear <= 0.0:
            raise ValueError(
                f"the smooth start ends at {a1 * ef / 1e9:.4g} GeV, beyond the linear ramp's end "
                f"({self.linear_end / 1e9:.4g} GeV): fewer smooth_turns, or a slower rate"
            )
        end = 2.0 * (1.0 - self.linear_end / ef) / rate
        return _Shape(a, rate, b, c, a1, n1, linear, end, -rate / end**2, rate / (2.0 * end**3))

    def turns(self, period: float) -> float:
        """The ramp's length in turns, from its start to the extraction energy."""
        return self.shape(period).turns

    def energy(self, turn: float, period: float) -> float:
        """Total energy (eV) at ``turn`` (turns from the ramp's start), held outside the ramp."""
        s = self.shape(period)
        n = float(turn)
        m = n - s.start - s.linear  # into the smooth end
        if n < 0.0:
            fraction = s.a
        elif n < s.start:
            fraction = s.a + s.b * n**3 + s.c * n**4
        elif n <= s.start + s.linear:
            fraction = s.a1 + s.rate * (n - s.start)
        elif m <= s.end:
            fraction = 1.0 + s.b2 * (s.end - m) ** 3 + s.c2 * (s.end - m) ** 4
        else:
            fraction = 1.0
        return fraction * self.extraction


class RfProgram(DomainModel):
    """The RF in time from the ramp's start: the cavities' total voltage and the synchronous phase.

    The synchronous phase is from the zero crossing, positive where the bunch gains energy
    (|q| V sin(phi_s) per turn: the energy gain and the radiation loss). Values are interpolated
    linearly between the points and held beyond the ends.
    """

    time: tuple[Real, ...] = Field(description="s from the ramp's start")
    voltage: tuple[Real, ...] = Field(description="V, the sum over the cavities")
    phase: tuple[Real, ...] = Field(description="Synchronous phase, rad")
    energy: tuple[Real, ...] = Field((), description="The designers' energy at each time, eV (to check)")

    @model_validator(mode="after")
    def _columns(self) -> Self:
        n = len(self.time)
        if n == 0:
            raise ValueError("an RF program needs points")
        for name in ("voltage", "phase", "energy"):
            values = getattr(self, name)
            if values and len(values) != n:
                raise ValueError(f"RF program: {len(values)} {name} values for {n} times")
        if any(t1 <= t0 for t0, t1 in zip(self.time, self.time[1:], strict=False)):
            raise ValueError("RF program: times must increase")
        if any(v < 0 for v in self.voltage):
            raise ValueError("RF program: voltages are not negative")
        return self

    def at(self, time: float) -> tuple[float, float]:
        """Total voltage (V) and synchronous phase (rad) at ``time`` (s from the ramp's start)."""
        times = self.time
        k = bisect.bisect_right(times, time)
        if k == 0:
            return self.voltage[0], self.phase[0]
        if k == len(times):
            return self.voltage[-1], self.phase[-1]
        f = (time - times[k - 1]) / (times[k] - times[k - 1])
        return (
            self.voltage[k - 1] + f * (self.voltage[k] - self.voltage[k - 1]),
            self.phase[k - 1] + f * (self.phase[k] - self.phase[k - 1]),
        )


class RampProfile(DomainModel):
    """A ring's ramp: its energy and RF programs, the injection porch and the flattop (see the module)."""

    name: Label
    branch: Label = Field(description="The ring's branch name")
    energy: EnergyRamp
    rf: RfProgram
    cavities: tuple[Label, ...] = Field(
        (), description="Cavities sharing the voltage equally, by occurrence name; none: all of the ring's"
    )
    porch: Real = Field(0.0, ge=0.0, description="s at injection before the ramp, in the designers' cycle")
    flattop: Real = Field(0.0, ge=0.0, description="s at the extraction energy before extraction")
    cycle: Real = Field(0.0, ge=0.0, description="s, the whole cycle (ramp down and reset included)")
    source: dict[str, Any] = Field(
        default_factory=dict, description="Where it came from: the file, its SHA-256, the sheets"
    )

    def ramp_turns(self, period: float) -> int:
        """Turns from the ramp's start to the flattop (the first at the extraction energy)."""
        return math.ceil(self.energy.turns(period))

    def extraction_turn(self, period: float) -> int:
        """Turns from the ramp's start to extraction: the ramp, then the flattop."""
        return self.ramp_turns(period) + round(self.flattop / period)

    def at(self, turn: float, period: float, mass: float) -> dict[str, float]:
        """The programs at ``turn`` (turns from the ramp's start): total energy and momentum ``pc``
        (eV), total RF ``voltage`` (V) and synchronous ``phase`` (rad)."""
        energy = self.energy.energy(turn, period)
        voltage, phase = self.rf.at(turn * period)
        return {"energy": energy, "pc": math.sqrt(energy**2 - mass**2), "voltage": voltage, "phase": phase}


def ramp_period(profile: RampProfile, lattice: Lattice) -> float:
    """The ring's revolution period at the speed of light (s): the ramp's turn."""
    return lattice.layout(profile.branch).length / C_LIGHT


def ramp_cavities(profile: RampProfile, lattice: Lattice) -> list[PlacedElement]:
    """The cavities that share the RF program: those listed, or all of the ring's."""
    cavities = [p for p in lattice.layout(profile.branch).placements if p.kind is ElementKind.RFCAVITY]
    if profile.cavities:
        cavities = [p for p in cavities if p.name in profile.cavities]
    return cavities


def ramp_problems(profile: RampProfile, lattice: Lattice) -> list[str]:
    """What keeps a ramp from running on its ring: the ring, its cavities, its reference, and whether
    the RF program covers the energy program and was computed for it."""
    where = f"ramp {profile.name}"
    if profile.branch not in {b.name for b in lattice.branches}:
        return [f"{where}: no branch {profile.branch}"]
    layout = lattice.layout(profile.branch)
    if not layout.periodic:
        return [f"{where}: {profile.branch} is not a ring"]
    problems = []
    cavities = ramp_cavities(profile, lattice)
    if missing := sorted(set(profile.cavities) - {p.name for p in cavities}):
        problems.append(f"{where}: no RF cavities named {', '.join(missing)} in {profile.branch}")
    if not cavities:
        problems.append(f"{where}: {profile.branch} has no RF cavities to follow its RF program")
    conditions = lattice.initial_conditions(layout.id)
    reference = conditions.reference if conditions else None
    if reference is not None and reference.pc_ref and reference.species_ref in SPECIES:
        energy = math.hypot(reference.pc_ref, SPECIES[reference.species_ref].mass)
        if abs(energy - profile.energy.injection) > 1e-6 * energy:
            problems.append(
                f"{where}: {profile.branch}'s reference ({energy / 1e9:.6g} GeV) is not the injection "
                f"energy ({profile.energy.injection / 1e9:.6g} GeV)"
            )
    period = ramp_period(profile, lattice)
    try:
        turns = profile.energy.turns(period)
    except ValueError as error:
        return [*problems, f"{where}: {error}"]
    if profile.rf.time[-1] < 0.99 * turns * period:
        problems.append(
            f"{where}: its RF program ends at {profile.rf.time[-1] * 1e3:.4g} ms, before the energy "
            f"program reaches extraction ({turns * period * 1e3:.4g} ms)"
        )
    if profile.rf.energy:
        mine = [profile.energy.energy(t / period, period) for t in profile.rf.time]
        worst = max(range(len(mine)), key=lambda k: abs(mine[k] - profile.rf.energy[k]) / mine[k])
        if abs(mine[worst] - profile.rf.energy[worst]) > 1e-3 * mine[worst]:
            at = f"{profile.rf.time[worst] * 1e3:.4g} ms"
            problems.append(
                f"{where}: its RF program was computed for {profile.rf.energy[worst] / 1e9:.5g} GeV at {at}, "
                f"where its energy program gives {mine[worst] / 1e9:.5g} GeV"
            )
    return problems
