"""A run's BPM history: by turn, across chunks, kept within its size, readable while written."""

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from eicvibe.virtual import BpmHistory, RunInfo


def info(**changes) -> RunInfo:
    base = dict(
        id="ring-test",
        lattice="demo",
        lattice_digest="0" * 64,
        branch="ring",
        branch_id="br_1",
        engine="xsuite",
        fidelity="fast",
        effects=["longitudinal"],
        start=datetime(2026, 10, 8, tzinfo=UTC),
        revolution_period=1e-6,
        chunk=100,
        bpms=["r-b1", "r-b2", "r-b3"],
        occurrences=["o1", "o2", "o3"],
        s=[1.0, 2.0, 3.0],
        beam={"particles": 1},
        resolution=0.005,
        retention_bytes=1 << 30,
    )
    return RunInfo(**(base | changes))


def chunk(first: int, turns: int = 100) -> tuple[np.ndarray, ...]:
    turn = np.arange(first, first + turns, dtype=float)
    x = np.vstack([turn + 0.1 * k for k in range(3)])
    return x, -x, np.ones_like(x), np.zeros(x.shape, dtype=int)


def test_turns_are_read_across_chunks_by_their_numbers(tmp_path):
    writer = BpmHistory.create(tmp_path / "run", info())
    for first in (0, 100, 200):
        writer.append(first, *chunk(first), wall=0.0)
    writer.event(100, "kick", x=1e-4)
    reader = BpmHistory.open(tmp_path / "run")  # another reader, as a notebook would open it
    assert reader.span() == (0, 300) and reader.turn == 300
    tbt = reader.turn_by_turn(150, 100)
    assert tbt.first_turn == 150 and tbt.x.position[0] == pytest.approx(np.arange(150, 250))
    assert tbt.y.position[2][0] == pytest.approx(-150.2)
    assert tbt.x.bpms == ("r-b1", "r-b2", "r-b3") and tbt.time == info().start + timedelta(seconds=150e-6)
    assert reader.turn_by_turn(turns=50).first_turn == 250  # the latest turns by default
    assert reader.turn_by_turn(250, 500).turns == 50  # only what exists
    assert [e["kind"] for e in reader.events()] == ["kick"] and reader.events(0, 100) == []
    assert reader.turn_by_turn(bpms=["r-b3"]).x.bpms == ("r-b3",)
    orbit = reader.orbit()
    assert list(orbit.turn) == [0, 100, 200] and orbit.x[1, 0] == pytest.approx(149.5)
    writer.append(300, *chunk(300), wall=0.0)  # the run goes on; the reader sees it
    assert reader.span() == (0, 400)
    with pytest.raises(ValueError, match="not kept"):
        reader.turn_by_turn(1000, 10)
    with pytest.raises(PermissionError):
        reader.append(400, *chunk(400), wall=0.0)
    with pytest.raises(FileExistsError):
        BpmHistory.create(tmp_path / "run", info())


def test_old_turns_go_first_beyond_the_size_kept(tmp_path):
    writer = BpmHistory.create(tmp_path / "run", info(retention_bytes=24_000))
    for first in range(0, 1000, 100):
        writer.append(first, *chunk(first), wall=0.0)
    first, end = writer.span()
    assert end == 1000 and 0 < first < 900
    with pytest.raises(ValueError, match="not kept"):
        writer.turn_by_turn(0, 50)
    assert len(writer.orbit().turn) == 10  # the slow orbit is kept whole


def test_a_rings_buffer_keeps_its_latest_turns(tmp_path):
    history = BpmHistory.create(tmp_path / "run", info(buffer_turns=300))
    for first in range(0, 1000, 100):
        history.append(first, *chunk(first), wall=(first / 1000, (first + 100) / 1000))
    assert history.span() == (700, 1000)  # the latest 300 turns, as a BPM's circular buffer holds them
    history.protect_from = 800  # a capture waits for turns from 800 on
    for first in range(1000, 1300, 100):
        history.append(first, *chunk(first), wall=(1.0, 1.1))
    assert history.span() == (800, 1300)
    history.protect_from = None
    history.append(1300, *chunk(1300), wall=(1.3, 1.4))
    assert history.span() == (1100, 1400)


def test_the_slow_orbit_averages_each_second_of_the_run(tmp_path):
    history = BpmHistory.create(tmp_path / "run", info(slow_period=1.0, buffer_turns=1000))
    history.append(0, *chunk(0), wall=(10.0, 10.5))  # the turns are spread over the time they took
    history.append(100, *chunk(100), wall=(10.5, 11.5))
    history.append(200, *chunk(200), wall=(11.5, 11.9))
    history.flush()
    orbit = history.orbit()
    assert list(orbit.turn) == [0, 150] and list(orbit.turns) == [150, 150]  # seconds 10 and 11
    assert orbit.x[:, 0] == pytest.approx([74.5, 224.5])  # the mean of each second's turns
    assert orbit.y[0, 2] == pytest.approx(-74.7)


def test_a_capture_freezes_the_turns_around_its_trigger(tmp_path):
    history = BpmHistory.create(tmp_path / "run", info(buffer_turns=300))
    for first in range(0, 400, 100):
        history.append(first, *chunk(first), wall=(0.0, 0.1))
    meta = history.save_capture("kick", 250, 200, 150, x=1e-4)
    assert meta["id"] == "000000000250-kick" and (meta["first"], meta["turns"], meta["x"]) == (200, 150, 1e-4)
    for first in range(400, 1000, 100):  # the buffer goes on; the capture stays
        history.append(first, *chunk(first), wall=(0.0, 0.1))
    reader = BpmHistory.open(tmp_path / "run")
    assert [c["id"] for c in reader.captures()] == ["000000000250-kick"]
    tbt = reader.capture("000000000250-kick")
    assert (tbt.first_turn, tbt.turns) == (200, 150) and tbt.x.position[0, 0] == pytest.approx(200.0)
    assert [(e["turn"], e["kind"], e["id"]) for e in reader.events()] == [
        (250, "capture", "000000000250-kick")
    ]
    with pytest.raises(KeyError):
        reader.capture("nope")
