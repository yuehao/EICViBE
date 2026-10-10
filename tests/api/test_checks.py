"""A ring run's tracking checked against the model, as the viewer's Check tab asks for it."""

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("xtrack")

from fastapi.testclient import TestClient  # noqa: E402

from eicvibe.api import Catalog, create_app  # noqa: E402
from eicvibe.api.demo import demo_lattice, write_demo  # noqa: E402
from eicvibe.api.runs import RunCatalog  # noqa: E402
from eicvibe.engines.contracts import Fidelity  # noqa: E402
from eicvibe.virtual import Beam, BpmModel, RingRun  # noqa: E402


@pytest.fixture(scope="module")
def checked(tmp_path_factory):
    """The demo ring, kicked by 0.1 mrad (about 1 mm at the BPMs) after 256 turns."""
    folder = tmp_path_factory.mktemp("check")
    lattice = demo_lattice()
    corrector = next(p.id for p in lattice.layout("ring").placements if p.kind.value == "Kicker")
    values = {(corrector, "MagneticMultipoleP.Kn0L"): 2e-5}  # a closed orbit for the model to give
    run = RingRun(
        lattice,
        "ring",
        folder / "runs" / "kicked",
        engine="xsuite",
        fidelity=Fidelity.FAST,
        beam=Beam(particles=50),
        bpm=BpmModel(resolution=0.001, gain_rms=0.0),
        values=values,
        turns=2048,
        buffer_turns=1024,
        run_id="kicked",
    ).start()
    run.wait(256)
    run.kick(x=1e-4)
    run.wait()
    run.stop()
    app = create_app(
        Catalog([write_demo(folder / "lattice")]), static_dir=None, runs=RunCatalog([folder / "runs"])
    )
    with TestClient(app) as http:
        yield http, run


def test_a_small_kick_gives_the_models_tune_phase_beta_and_orbit(checked):
    http, run = checked
    response = http.get("/api/v1/runs/kicked/check", params={"turns": 512})
    assert response.status_code == 200, response.text
    check = response.json()
    assert check["source"].startswith("capture:") and check["first_turn"] == check["trigger"]
    assert check["settings"] == f"the run's at turn {check['trigger']}: 1 beyond the lattice's"
    x = check["x"]
    assert x["measured_full"] == pytest.approx(x["model_tune"], abs=2e-3)  # the integer part from the model
    assert x["map_tune"] == pytest.approx(x["model_tune"] % 1, abs=1e-4)  # the run's own one-turn map agrees
    assert x["phase_rms"] < 2.0 and x["beta_beating_rms"] < 0.03  # degrees; no BPM gain errors here
    assert len(x["phase_error"]) == len(check["bpms"]) == len(x["model_orbit"])
    # The mean readings are the model's closed orbit plus the BPMs' hidden offsets (0.1 mm rms).
    offsets = [m - o for m, o in zip(x["orbit"], x["model_orbit"], strict=True)]
    assert max(abs(v) for v in offsets) < 0.4 and max(abs(v) for v in x["model_orbit"]) > 0.05
    assert check["y"]["measured_tune"] is None and any("hardly oscillates in y" in n for n in check["notes"])
    assert len(x["spectrum"]) == len(check["frequencies"]) and len(x["drift_tune"]) == len(
        check["drift_turns"]
    )


def test_the_detuning_curve_starts_at_the_models_tune(checked):
    http, _ = checked
    check = http.get("/api/v1/runs/kicked/check", params={"turns": 512, "detuning": True}).json()
    detuning = check["detuning"]
    assert detuning["plane"] == "x" and len(detuning["amplitude"]) == 8
    assert detuning["amplitude"] == sorted(detuning["amplitude"])
    assert detuning["tune_x"][0] == pytest.approx(
        min(check["x"]["model_tune"] % 1, 1 - check["x"]["model_tune"] % 1), abs=2e-3
    )


def test_the_buffer_and_errors(checked):
    http, _ = checked
    buffer = http.get("/api/v1/runs/kicked/check", params={"source": "buffer", "turns": 256}).json()
    assert (buffer["source"], buffer["turns"], buffer["trigger"]) == ("buffer", 256, None)
    assert http.get("/api/v1/runs/nope/check").status_code == 404
    assert http.get("/api/v1/runs/kicked/check", params={"source": "capture:nope"}).status_code == 422
    assert http.get("/api/v1/runs/kicked/check", params={"turns": 10}).status_code == 422
