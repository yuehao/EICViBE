"""A ring's ramp profile (ADR 0013): the designers' energy waveform, their RF table, and the checks."""

import math

import numpy as np
import pytest

from eicvibe.domain.lattice import LatticeBuilder, SequentialIds
from eicvibe.domain.machine import EnergyRamp, RampProfile, RfProgram, ramp_period, ramp_problems

MASS = 510998.95069
#: The RCS parameter database's 9 GeV ramp ("9GeV Lat"), and its turn: 1506.934 m at the speed of light.
RCS = EnergyRamp(injection=0.75e9, extraction=9e9, linear_end=8.813e9, rate=86.25e9, smooth_turns=4000)
PERIOD = 1506.934 / 299_792_458.0


def test_the_waveform_is_the_designers_sheet():
    shape = RCS.shape(PERIOD)
    # The sheet's own numbers ("9GeV Lat", cells I3 to I9 and B11 to B13).
    assert (shape.a1, shape.rate, shape.b, shape.c) == pytest.approx(
        (0.17967632294472197, 4.8171494805694326e-05, 3.010718425355895e-12, -3.763398031694869e-16),
        rel=1e-12,
    )
    assert (shape.linear, shape.end) == pytest.approx((16597.905099324144, 862.6586267080731), rel=1e-12)
    assert RCS.turns(PERIOD) * PERIOD == pytest.approx(0.1078734713797391, rel=1e-12)  # its t2, in s
    ends = [RCS.energy(n, PERIOD) for n in (-5.0, 0.0, RCS.turns(PERIOD), 1e6)]
    assert ends == pytest.approx([0.75e9, 0.75e9, 9e9, 9e9])


def test_its_joins_are_smooth_to_the_second_derivative():
    shape = RCS.shape(PERIOD)
    for join in (shape.start, shape.start + shape.linear):
        e = [RCS.energy(join + k * 1e-2, PERIOD) for k in (-2.0, -1.0, 0.0, 1.0, 2.0)]
        slopes = np.diff(e)
        assert slopes[0] == pytest.approx(slopes[-1], rel=1e-6)  # no kink
    linear = [RCS.energy(shape.start + n, PERIOD) for n in (10.0, 11.0)]
    assert np.diff(linear)[0] == pytest.approx(RCS.rate * PERIOD, rel=1e-12)  # 86.25 GeV/s


def test_a_waveform_whose_smooth_start_overruns_its_linear_end_is_refused():
    with pytest.raises(ValueError, match="energies must rise"):
        EnergyRamp(injection=1e9, extraction=2e9, linear_end=3e9, rate=1e10, smooth_turns=10)
    slow = EnergyRamp(injection=1e9, extraction=2e9, linear_end=1.001e9, rate=1e12, smooth_turns=10_000)
    with pytest.raises(ValueError, match="beyond the linear ramp's end"):
        slow.turns(PERIOD)


def test_the_rf_program_interpolates_and_holds_its_ends():
    rf = RfProgram(time=(0.0, 1.0), voltage=(1e6, 3e6), phase=(0.0, 0.5))
    assert [rf.at(t) for t in (-1.0, 0.5, 2.0)] == [(1e6, 0.0), (2e6, 0.25), (3e6, 0.5)]
    with pytest.raises(ValueError, match="times must increase"):
        RfProgram(time=(0.0, 0.0), voltage=(1e6, 1e6), phase=(0.0, 0.0))
    with pytest.raises(ValueError, match="2 phase values for 3 times"):
        RfProgram(time=(0.0, 1.0, 2.0), voltage=(1.0, 1.0, 1.0), phase=(0.0, 0.0))


def ring(pc: float = math.sqrt(0.75e9**2 - MASS**2), cavity: bool = True):
    b = LatticeBuilder("r", ids=SequentialIds())
    b.define("BEG", "BeginningEle", ReferenceP={"species_ref": "electron", "pc_ref": pc})
    b.define("D", "Drift", length=1506.934)
    b.define("CAV", "RFCavity", RFP={"harmon": 2971, "voltage": 1e6})
    b.add_branch("rcs", "periodic", ["BEG", "D", *(["CAV"] if cavity else [])])
    b.add_branch("line", "linac", ["BEG", "D"])
    return b.build()


def profile(**changes) -> RampProfile:
    turns = RCS.turns(PERIOD)
    times = np.linspace(0.0, turns * PERIOD, 50)
    energies = [RCS.energy(t / PERIOD, PERIOD) for t in times]
    fields = {
        "name": "rcs",
        "branch": "rcs",
        "energy": RCS,
        "rf": RfProgram(time=tuple(times), voltage=(1e6,) * 50, phase=(0.1,) * 50, energy=tuple(energies)),
        "flattop": 0.01,
    }
    return RampProfile(**(fields | changes))


def test_a_ramp_is_checked_against_its_ring():
    lattice = ring()
    good = profile()
    assert ramp_problems(good, lattice) == []
    assert ramp_period(good, lattice) == pytest.approx(PERIOD)
    assert good.extraction_turn(PERIOD) == math.ceil(RCS.turns(PERIOD)) + round(0.01 / PERIOD)
    at = good.at(0, PERIOD, MASS)
    assert at["pc"] == pytest.approx(math.sqrt(0.75e9**2 - MASS**2)) and at["voltage"] == 1e6
    assert "no branch elsewhere" in ramp_problems(profile(branch="elsewhere"), lattice)[0]
    assert "line is not a ring" in ramp_problems(profile(branch="line"), lattice)[0]
    assert "no RF cavities" in ramp_problems(good, ring(cavity=False))[0]
    assert "not the injection energy" in ramp_problems(good, ring(pc=1e9))[0]
    short = RfProgram(time=(0.0, 0.05), voltage=(1e6, 1e6), phase=(0.0, 0.0))
    assert "before the energy program reaches extraction" in ramp_problems(profile(rf=short), lattice)[0]
    other = good.rf.model_dump() | {"energy": tuple(1.01 * e for e in good.rf.energy)}
    assert "was computed for" in ramp_problems(profile(rf=RfProgram(**other)), lattice)[0]
