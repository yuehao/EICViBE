"""A ring's settings turn by turn (ADR 0007, 0013): changes ramping in, a program that owns its settings,
the supplies' noise, as the engines' ramps. Plain numpy: no engine."""

from types import SimpleNamespace

import numpy as np
import pytest

from eicvibe.virtual.schedule import Program, Schedule, half_cosine

Q = ("q1", "MagneticMultipoleP.Kn1")
C = ("c1", "MagneticMultipoleP.Kn0L")
V = ("rf", "RFP.voltage")


def schedule(**values) -> Schedule:
    return Schedule(dict(values), design=lambda key: 0.0, to_engine=dict)


def test_a_change_ramps_in_as_a_half_cosine_and_a_newer_one_takes_over_from_where_it_is():
    s = schedule()
    s.values[Q] = 1.0
    assert s.change(100, {Q: 2.0}, 100) is False  # a ramp, not a step
    assert s.value(Q, 100) == 1.0 and s.value(Q, 150) == pytest.approx(1.5) and s.value(Q, 200) == 2.0
    _, [ramp] = s.window(140, 20)
    assert ramp.start == {Q: 1.0} and ramp.end == {Q: 2.0}
    assert np.asarray(ramp.fraction) == pytest.approx(half_cosine((np.arange(140, 160) - 100) / 100))
    s.advance(150)
    assert s.values[Q] == pytest.approx(1.5)  # the run's value is where the ramp is
    s.change(150, {Q: 0.5}, 100)  # changed again mid-ramp: from 1.5, not from 1.0 or 2.0
    assert len(s.changes) == 1 and s.value(Q, 150) == pytest.approx(1.5) and s.value(Q, 250) == 0.5
    s.advance(260)
    assert s.changes == [] and s.values[Q] == 0.5


def test_a_step_changes_the_values_at_once():
    s = schedule()
    assert s.change(10, {C: 1e-4}, 0) is True and s.values[C] == 1e-4 and s.window(10, 5)[1] == []


def test_a_program_owns_its_settings_and_a_change_of_one_applies_when_it_ends():
    s = schedule()
    s.values[V] = 1e5
    turns_seen = []

    def ramp(turns):
        turns_seen.append(turns.copy())
        return 7.5e8 + turns * 1e3, {V: 2e5 + turns}

    s.program = Program(frozenset({V}), ramp)
    s.change(0, {V: 3e5, Q: 1.0}, 10)
    assert s.owned([V, Q]) == [V]
    momentum, ramps = s.window(0, 4)
    assert momentum == pytest.approx(7.5e8 + np.arange(4) * 1e3)
    assert [r.start for r in ramps] == [{Q: 0.0}, {V: 0.0}]  # the change of V waits; the program's comes last
    assert np.asarray(ramps[-1].fraction) == pytest.approx(2e5 + np.arange(4))
    assert np.array_equal(turns_seen[0], np.arange(4))
    s.advance(20)
    assert s.end_program() == {V} and s.program is None
    assert s.value(V, 20) == 3e5 and s.window(20, 3)[0] is None  # its setting, as changed meanwhile


def test_the_supplies_noise_moves_settings_about_their_values_and_rides_on_a_ramp():
    deviation = np.array([0.01, -0.01, 0.0])

    class Noise:
        def turns(self, first, count):
            return [(SimpleNamespace(keys=(Q, C)), deviation[:count])]

    s = schedule()
    s.values[C] = 2e-4
    s.noise = Noise()
    s.change(0, {Q: 1.0}, 3)
    _, ramps = s.window(0, 3)
    along, steady = ramps[0], ramps[1]
    # Q ramping and fluctuating: its own line through its values; C steady: about its value.
    expected = 1.0 * half_cosine(np.arange(3) / 3) * (1 + deviation)
    assert along.start == {Q: 0.0} and np.asarray(along.fraction) == pytest.approx(expected)
    assert steady.start == {C: 2e-4} and steady.end == {C: 4e-4}
    assert np.asarray(steady.fraction) == pytest.approx(deviation)
