"""BPM data from real systems and its analysis: the reference the virtual rings' data must match."""

import math
import os
from datetime import UTC, datetime
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pytest

from eicvibe.diagnostics import (
    NO_BEAM,
    VALID,
    PlaneData,
    TurnByTurn,
    amplitudes,
    closed_orbit,
    read_nsls2,
    read_rhic,
    tune,
    tune_of,
    write_rhic,
)

RHIC = Path(
    os.environ.get(
        "EICVIBE_RHIC_TBT",
        "/Users/haoyue/Research/studentwork/Fung/RHIC_BPM/Error analysis/TBT_Data/"
        "data_Wed_Sep__4_19_08_18_2024",
    )
)
NSLS2 = Path(
    os.environ.get(
        "EICVIBE_NSLS2_TBT",
        "/Users/haoyue/Research/AI/Bayesian_Inference/NSLSII_BPM_MCMC/TbT20191024/op20191024_01.h5",
    )
)


def oscillation(turns: int, tune: float, amplitude: float = 1.0, phase: float = 0.0) -> np.ndarray:
    return amplitude * np.cos(2 * math.pi * tune * np.arange(turns) + phase)


def synthetic(turns: int = 1000) -> TurnByTurn:
    rng = np.random.default_rng(3)
    x = np.array([oscillation(turns, 0.31, 1.0, k) + 0.2 * k for k in range(6)]) + rng.normal(
        0, 0.01, (6, turns)
    )
    y = np.array([oscillation(turns, 0.18, 0.5, k) for k in range(5)])
    x[2, 100:] = np.nan  # one BPM loses the beam
    status = np.where(np.isnan(x), NO_BEAM, VALID)
    return TurnByTurn(
        ring="test",
        x=PlaneData(tuple(f"b{k}" for k in range(6)), np.arange(6.0), x, np.ones_like(x), status),
        y=PlaneData(
            tuple(f"b{k}" for k in range(5)), np.arange(5.0), y
        ),  # a single-plane BPM: b5 reads x only
        first_turn=0,
        time=datetime(2026, 10, 8, 12, 0, 0, 250000, tzinfo=UTC),
        source="synthetic",
    )


def test_a_tune_is_found_to_1e4_through_a_drift_and_noise():
    rng = np.random.default_rng(1)
    signal = oscillation(1024, 0.2345, 0.3) + np.linspace(0, 2, 1024) + rng.normal(0, 0.02, 1024)
    assert tune_of(signal) == pytest.approx(0.2345, abs=1e-4)
    # A strong low line (a synchrotron sideband) wins unless it is excluded.
    sideband = signal + oscillation(1024, 0.012, 1.0)
    assert tune_of(sideband) == pytest.approx(0.012, abs=1e-3)
    assert tune_of(sideband, low=0.05) == pytest.approx(0.2345, abs=1e-4)
    assert math.isnan(tune_of(np.full(64, np.nan)))


def test_analysis_reads_any_acquisition():
    tbt = synthetic()
    assert tune(tbt, "x") == pytest.approx(0.31, abs=2e-4) and tune(tbt, "y") == pytest.approx(0.18, abs=2e-4)
    x, _ = closed_orbit(tbt)
    assert x[4] == pytest.approx(0.8, abs=0.01)
    assert amplitudes(tbt.y)[0] == pytest.approx(0.5, rel=0.01)
    assert tbt.select(["b5", "b0"]).y.bpms == ("b0",)  # b5 has no vertical reading
    assert tbt.window(10, 100).first_turn == 10 and tbt.window(10, 100).turns == 100
    with pytest.raises(ValueError, match="2 BPMs, 3 s values"):
        PlaneData(("a", "b"), np.zeros(3), np.zeros((2, 4)))


def test_the_rhic_layout_round_trips(tmp_path):
    tbt = synthetic()
    write_rhic(tbt, tmp_path / "tbt.sdds")
    back = read_rhic(tmp_path / "tbt.sdds")
    assert back.x.bpms == tbt.x.bpms and back.y.bpms == tbt.y.bpms
    assert back.time == tbt.time
    valid = tbt.x.status == VALID
    assert back.x.position[valid] == pytest.approx(tbt.x.position[valid], abs=1e-9)
    assert np.array_equal(back.x.status, tbt.x.status)  # the lost readings say so
    assert back.y.position == pytest.approx(tbt.y.position, abs=1e-9)


@pytest.mark.skipif(not RHIC.exists(), reason="no RHIC turn-by-turn file here")
def test_a_real_rhic_acquisition(tmp_path):
    tbt = read_rhic(RHIC)
    assert tbt.ring == "Blue" and tbt.time.year == 2024
    assert (len(tbt.x.bpms), tbt.x.turns, len(tbt.y.bpms)) == (168, 1024, 167)
    assert tbt.x.bpms[:2] == ("g6-bx", "bo6-b1") and tbt.x.s[1] == pytest.approx(25.0237827)
    assert tune(tbt, "x") == pytest.approx(0.3005, abs=1e-3)  # RHIC's .695, seen as 1 - q
    write_rhic(tbt, tmp_path / "again.sdds")
    again = read_rhic(tmp_path / "again.sdds")
    assert again.x.position == pytest.approx(tbt.x.position, abs=1e-6)
    assert np.array_equal(again.y.status, tbt.y.status)


@pytest.mark.skipif(not NSLS2.exists() or find_spec("h5py") is None, reason="no NSLS-II file or h5py here")
def test_a_real_nsls2_acquisition():
    tbt = read_nsls2(NSLS2)
    assert tbt.x.position.shape == (180, 2000)
    assert "pinger" in tbt.metadata["comment"]
    assert (tune(tbt, "x"), tune(tbt, "y")) == pytest.approx(
        (0.2185, 0.2603), abs=2e-3
    )  # NSLS-II: 33.22, 16.26
