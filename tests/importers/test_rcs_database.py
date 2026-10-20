"""The RCS parameter database's ramp (ADR 0013), read from the workbook in the EIN folder."""

import math
import os
from pathlib import Path

import numpy as np
import pytest

from eicvibe.importers.rcs_database import read_rcs_ramp
from eicvibe.machines.ein import RCS_DATABASE, add_ein_ramp
from eicvibe.store import read_ramps, write_machine
from tests.domain.machine.test_ramps import ring

FOLDER = Path(os.environ.get("EICVIBE_EIN_FOLDER", "/Users/haoyue/Research/EIC/virtual_accelerator/EIN"))
pytestmark = pytest.mark.skipif(not (FOLDER / RCS_DATABASE).is_file(), reason="no RCS parameter database")


@pytest.fixture(scope="module")
def ramp():
    return read_rcs_ramp(FOLDER / RCS_DATABASE)


def test_the_9gev_ramp_is_the_sheets_waveform_and_rf_table(ramp):
    energy = ramp.energy
    assert (energy.injection, energy.extraction, energy.linear_end, energy.rate, energy.smooth_turns) == (
        0.75e9,
        9e9,
        8.813e9,
        86.25e9,
        4000,
    )
    assert (ramp.porch, ramp.flattop, ramp.cycle) == pytest.approx((0.01, 0.01, 1.0))  # a 1 Hz cycle
    rf = ramp.rf
    assert len(rf.time) == 144 and rf.time[-1] == pytest.approx(0.10782, abs=1e-5)
    # 1.698 MV at injection (the VA's pinned value) to 15.9 MV, the synchronous phase to 52.5 degrees
    assert (rf.voltage[0], rf.voltage[-1]) == pytest.approx((1.6976e6, 15.917e6), rel=1e-4)
    assert (math.degrees(max(rf.phase)), math.degrees(rf.phase[-1])) == pytest.approx(
        (52.48, 52.09), abs=0.01
    )
    # The RF table was computed for this waveform: its energies agree to 2e-6.
    period = 1506.934 / 299_792_458.0
    mine = np.array([energy.energy(t / period, period) for t in rf.time])
    assert np.max(np.abs(mine - rf.energy) / mine) < 2e-6
    assert ramp.source["sheets"] == ["9GeV Lat", "RF Ramp 9GeV Lat"] and len(ramp.source["sha256"]) == 64


def test_another_sheet_must_be_there(ramp):
    with pytest.raises(ValueError, match="no sheet 'RF Ramp 7GeV'"):
        read_rcs_ramp(FOLDER / RCS_DATABASE, rf_sheet="RF Ramp 7GeV")


def test_the_ramp_joins_a_machine_folder_and_nothing_else_changes(tmp_path, ramp):
    folder = tmp_path / "machine"
    write_machine(folder, ring())
    before = {p: p.read_bytes() for p in folder.rglob("*") if p.is_file()}
    written = add_ein_ramp(folder, FOLDER)
    after = {p: p.read_bytes() for p in folder.rglob("*") if p.is_file()}
    assert set(after) - set(before) == set(written) and all(after[p] == data for p, data in before.items())
    assert read_ramps(folder) == {"rcs": ramp}
