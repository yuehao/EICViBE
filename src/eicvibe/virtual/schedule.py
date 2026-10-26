"""What a ring's settings and reference are on every turn: the one place its ramps are made (ADR 0007, 0013).

Settings reach the beam turn by turn, never as steps at the boundaries of engine calls. Three kinds of
source move them:

- a **change** (an operator's write, an operation's proposal applied): each setting goes from its
  value at the change's turn to the new one over ``turns`` turns, a half cosine, flat at both ends, as a
  magnet follows its supply; a newer change of a setting takes over from where it is, so nothing jumps;
- a **program** (a synchrotron's ramp, ADR 0013): the reference momentum and some settings (its RF's
  voltage and phase) as values turn by turn; while it runs it owns its settings, and a change of one
  goes on underneath and applies when the program ends;
- **noise**: the power supplies' fluctuations about whatever their settings are (ADR 0011).

:class:`Schedule` keeps them and answers what a run asks: a setting's value at a turn, and over an
engine call's turns the reference momentum and the settings that move, as the engines' ``Ramp``s. It is
plain numpy: no engine, no run.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np

from eicvibe.engines.contracts import Ramp

from .errors import Fluctuations

type Key = tuple[str, str]
"""A setting: (occurrence id, parameter)."""


def half_cosine(u):
    """A change's progress from 0 to 1 as its fraction ``u`` of the change's turns goes from 0 to 1: a half
    cosine, flat at both ends, so the change starts and stops smoothly; held outside."""
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(u, 0.0, 1.0))


@dataclass
class Change:
    """Settings going from ``start`` to ``end`` over ``turns`` turns from turn ``first``."""

    first: int
    turns: int
    start: dict[Key, float]
    end: dict[Key, float]

    def progress(self, turns) -> np.ndarray:
        return half_cosine((np.asarray(turns) - self.first) / self.turns)

    def value(self, key: Key, turn: int) -> float:
        if turn - self.first >= self.turns:
            return self.end[key]
        f = float(self.progress(turn))
        return self.start[key] + f * (self.end[key] - self.start[key])

    def done(self, turn: int) -> bool:
        return turn - self.first >= self.turns


@dataclass
class Program:
    """Settings, and the reference momentum, as values turn by turn while it runs (a synchrotron's ramp).

    ``values`` takes the turns of an engine call and gives the reference momentum on each (pc, eV; or
    None to leave the reference alone) and the values of the settings it sets, ``keys``, on each.
    """

    keys: frozenset[Key]
    values: Callable[[np.ndarray], tuple[np.ndarray | None, dict[Key, np.ndarray]]]


@dataclass
class Schedule:
    """A ring's settings turn by turn: the run's values, the changes ramping in, a program, the noise.

    ``values`` are the run's settings as of the turn it last advanced to (the dict is the run's own,
    updated in place); ``design`` gives a setting the run has not set; ``to_engine`` puts settings as the
    engine takes them (with the machine's errors).
    """

    values: dict[Key, float]
    design: Callable[[Key], float]
    to_engine: Callable[[Mapping[Key, float]], dict]
    noise: Fluctuations | None = None
    changes: list[Change] = field(default_factory=list)
    program: Program | None = None

    def change(self, turn: int, values: Mapping[Key, float], turns: int) -> bool:
        """Settings changed at ``turn``, ramping in over ``turns`` turns (a newer change of a setting takes
        over from where it is). With ``turns`` 0 they step: the run's values change at once, and the
        answer is True for the run to put them into its engine session."""
        begin = {key: self.value(key, turn) for key in values}
        for active in self.changes:
            for key in values:
                active.start.pop(key, None)
                active.end.pop(key, None)
        self.changes = [c for c in self.changes if c.start]
        if turns > 0:
            self.changes.append(Change(turn, turns, begin, dict(values)))
            return False
        self.values.update(values)
        return True

    def value(self, key: Key, turn: int) -> float:
        """A setting's value before ``turn``: its change's, the run's, or the design's (a program's values
        are the beam's while it runs, not the setting's)."""
        for active in self.changes:
            if key in active.start:
                return active.value(key, turn)
        if key in self.values:
            return self.values[key]
        return self.design(key)

    def owned(self, keys) -> list[Key]:
        """Those of ``keys`` a running program sets, whose changes apply when it ends."""
        return [k for k in keys if self.program is not None and k in self.program.keys]

    def window(self, first: int, turns: int) -> tuple[np.ndarray | None, list[Ramp]]:
        """Over turns ``[first, first + turns)``: the reference momentum (None: fixed) and the settings that
        move, as the engine takes them: the changes ramping in (not those a program owns), the supplies'
        noise about them, and the program's settings, last."""
        owned = self.program.keys if self.program is not None else frozenset()
        span = np.arange(first, first + turns)
        ramps = []
        for active in self.changes:
            start = {k: v for k, v in active.start.items() if k not in owned}
            end = {k: v for k, v in active.end.items() if k not in owned}
            if start:
                ramps.append(Ramp(self.to_engine(start), self.to_engine(end), active.progress(span)))
        if self.noise is not None:
            for supply, deviation in self.noise.turns(first, turns):
                steady = {}
                for key in supply.keys:
                    ramping = next((r for r in ramps if key in r.start), None)
                    if ramping is None:
                        steady[key] = self.to_engine({key: self.value(key, first)})[key]
                        continue
                    # Ramping and fluctuating: a line of its own, through its values turn by turn.
                    start, end = ramping.start.pop(key), ramping.end.pop(key)
                    values = (start + np.asarray(ramping.fraction) * (end - start)) * (1.0 + deviation)
                    ramps.append(Ramp({key: 0.0}, {key: 1.0}, values))
                if steady := {key: value for key, value in steady.items() if value}:
                    ramps.append(Ramp(steady, {key: 2.0 * value for key, value in steady.items()}, deviation))
        ramps = [r for r in ramps if r.start]
        momentum = None
        if self.program is not None:
            momentum, values = self.program.values(span)
            # Each setting through its values turn by turn: a ramp from 0 to 1 whose fraction is the value.
            ramps += [Ramp({key: 0.0}, {key: 1.0}, v) for key, v in values.items()]
        return momentum, ramps

    def advance(self, turn: int) -> None:
        """After the turns up to ``turn``: the run's values where the changes are; finished changes done."""
        for active in list(self.changes):
            self.values.update({key: self.value(key, turn) for key in active.start})
            if active.done(turn):
                self.changes.remove(active)

    def end_program(self) -> frozenset[Key]:
        """The program ends: its settings are the run's again (those changed meanwhile, as changed). The
        settings it set, for the run to put back into its engine session."""
        keys = self.program.keys if self.program is not None else frozenset()
        self.program = None
        return keys
