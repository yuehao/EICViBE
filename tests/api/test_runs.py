"""Virtual rings' runs in the viewer API: orbits every N turns, live and from history."""

from datetime import UTC, datetime

import numpy as np
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from eicvibe.api import Catalog, create_app  # noqa: E402
from eicvibe.api.runs import RunCatalog  # noqa: E402
from eicvibe.virtual import BpmHistory, RunInfo  # noqa: E402


def write_run(folder, turns: int = 300, chunk: int = 100) -> BpmHistory:
    info = RunInfo(
        id="ring-test",
        lattice="demo",
        lattice_digest="sha256:" + "a" * 64,
        branch="ring",
        branch_id="br_1",
        engine="xsuite",
        fidelity="fast",
        effects=["longitudinal"],
        start=datetime(2026, 10, 8, tzinfo=UTC),
        revolution_period=1e-6,
        chunk=chunk,
        bpms=["r-b1", "r-b2", "r-b3"],
        occurrences=["o1", "o2", "o3"],
        s=[1.0, 2.0, 3.0],
        beam={"particles": 1},
        resolution=0.005,
        retention_bytes=1 << 30,
    )
    history = BpmHistory.create(folder, info)
    history.event(0, "inject", particles=1)
    for first in range(0, turns, chunk):
        turn = np.arange(first, first + chunk, dtype=float)
        x = np.vstack([turn + 0.1 * k for k in range(3)])
        x[1, :] = np.nan  # a BPM that reads nothing
        history.append(first, x, -x, np.ones_like(x), np.zeros(x.shape, int), wall=0.0)
    history.event(250, "kick", x=1e-4)
    history.set_state("running", turns)
    return history


@pytest.fixture
def client(tmp_path):
    write_run(tmp_path / "runs" / "one")
    app = create_app(Catalog([]), static_dir=None, runs=RunCatalog([tmp_path / "runs"]))
    return TestClient(app), tmp_path / "runs" / "one"


def test_runs_are_listed_with_how_far_they_are(client):
    http, _ = client
    [run] = http.get("/api/v1/runs").json()
    assert run["id"] == "ring-test" and run["lattice_digest"] == "a" * 64  # as the catalog writes digests
    assert (run["first_turn"], run["end_turn"], run["state"], run["bpms"]) == (0, 300, "running", 3)
    assert (run["kind"], run["unit"]) == ("ring", "turn")


def test_the_latest_orbits_every_n_turns(client):
    http, folder = client
    body = http.get("/api/v1/runs/ring-test/orbits", params={"every": 5, "count": 5}).json()
    assert body["turns"] == [279, 284, 289, 294, 299] and body["unit"] == "turn"  # oldest first
    assert body["bpms"] == ["r-b1", "r-b2", "r-b3"] and body["s"] == [1.0, 2.0, 3.0]
    assert body["x"][-1][0] == pytest.approx(299.0) and body["y"][-1][2] == pytest.approx(-299.2)
    assert body["x"][0][1] is None  # no reading
    assert body["end_turn"] == 300 and body["state"] == "running"
    assert body["events"] == []  # the kick at turn 250 came before these orbits
    around = http.get("/api/v1/runs/ring-test/orbits", params={"every": 5, "count": 5, "end": 260}).json()
    assert [(e["turn"], e["kind"]) for e in around["events"]] == [(250, "kick")]
    # The run goes on: the next poll shows its new turns.
    turn = np.arange(300, 400, dtype=float)
    BpmHistory(folder, BpmHistory.open(folder).info, writable=True).append(
        300,
        np.vstack([turn] * 3),
        np.vstack([turn] * 3),
        np.ones((3, 100)),
        np.zeros((3, 100), int),
        wall=0.0,
    )
    assert http.get("/api/v1/runs/ring-test/orbits").json()["turns"][-1] == 399


def test_orbits_from_the_history_are_kept_and_revalidated(client):
    http, _ = client
    first = http.get("/api/v1/runs/ring-test/orbits", params={"every": 20, "count": 3, "end": 101})
    assert first.json()["turns"] == [60, 80, 100]
    again = http.get(first.url, headers={"If-None-Match": first.headers["etag"]})
    assert again.status_code == 304
    clipped = http.get("/api/v1/runs/ring-test/orbits", params={"every": 50, "count": 10, "end": 120}).json()
    assert clipped["turns"] == [19, 69, 119]  # only what the run has
    assert http.get("/api/v1/runs/nope/orbits").status_code == 404
    assert http.get("/api/v1/runs/ring-test/orbits", params={"count": 500}).status_code == 422


@pytest.fixture
def ring_run(tmp_path):
    """A ring's run as RingRun writes it: a 300-turn buffer, a slow orbit per second, a capture."""
    info = write_run(tmp_path / "runs" / "one", turns=0).info
    folder = tmp_path / "runs" / "two"
    info = info.model_copy(update={"id": "ring-buffered", "buffer_turns": 300, "slow_period": 1.0})
    history = BpmHistory.create(folder, info)
    for first in range(0, 1000, 100):
        turn = np.arange(first, first + 100, dtype=float)
        x = np.vstack([turn + 0.1 * k for k in range(3)])
        history.append(
            first, x, -x, np.ones_like(x), np.zeros(x.shape, int), wall=(first / 200, (first + 100) / 200)
        )
        if first == 600:
            history.event(650, "kick", x=1e-4)
            history.save_capture("kick", 650, 600, 100, x=1e-4)
    history.flush()
    history.set_state("running", 1000)
    app = create_app(Catalog([]), static_dir=None, runs=RunCatalog([tmp_path / "runs"]))
    return TestClient(app)


def test_a_rings_buffer_its_slow_orbit_and_its_captures(ring_run):
    http = ring_run
    entry = next(r for r in http.get("/api/v1/runs").json() if r["id"] == "ring-buffered")
    assert (entry["buffer_turns"], entry["slow_period"], entry["captures"]) == (300, 1.0, 1)
    assert (entry["first_turn"], entry["end_turn"]) == (700, 1000)  # only the buffer, turn by turn
    live = http.get("/api/v1/runs/ring-buffered/orbits", params={"every": 100, "count": 3}).json()
    assert (live["source"], live["unit"], live["turns"]) == ("buffer", "turn", [799, 899, 999])
    query = {"source": "slow", "every": 1, "count": 5}
    slow = http.get("/api/v1/runs/ring-buffered/orbits", params=query).json()
    assert (slow["source"], slow["unit"], slow["sample_period"]) == ("slow", "sample", 1.0)
    assert slow["turns"] == [0, 1, 2, 3, 4] and slow["x"][0][0] == pytest.approx(
        99.5
    )  # second 0: turns 0-199
    assert slow["times"] == [0.0, 1.0, 2.0, 3.0, 4.0]
    (capture,) = http.get("/api/v1/runs/ring-buffered/captures").json()
    assert (capture["kind"], capture["trigger"], capture["first"], capture["turns"]) == (
        "kick",
        650,
        600,
        100,
    )
    assert capture["detail"] == {"x": 1e-4} and not capture["partial"]
    frames = http.get(
        "/api/v1/runs/ring-buffered/orbits",
        params={"source": f"capture:{capture['id']}", "every": 10, "count": 3},
    ).json()
    assert frames["turns"] == [679, 689, 699] and frames["state"] == "capture"
    assert http.get("/api/v1/runs/ring-buffered/orbits", params={"source": "capture:nope"}).status_code == 404
    assert http.get("/api/v1/runs/ring-buffered/orbits", params={"source": "../x"}).status_code == 422
