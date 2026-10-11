"""The viewer's optional AI features (ADR 0009), against a stand-in model server."""

import socket

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("xtrack")

from fastapi.testclient import TestClient  # noqa: E402

from eicvibe.ai import AiConfig, LocalModel  # noqa: E402
from eicvibe.api import (  # noqa: E402
    Catalog,
    create_app,
    schemas,  # noqa: E402
)
from eicvibe.api.assist import Assistant, Busy, _verdict  # noqa: E402
from eicvibe.api.demo import demo_lattice, write_demo  # noqa: E402
from eicvibe.api.runs import RunCatalog  # noqa: E402
from eicvibe.engines.contracts import Fidelity  # noqa: E402
from eicvibe.virtual import Beam, RingRun  # noqa: E402
from tests.ai.fake_ollama import FakeOllama  # noqa: E402


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    folder = tmp_path_factory.mktemp("assist")
    run = RingRun(
        demo_lattice(),
        "ring",
        folder / "runs" / "kicked",
        engine="xsuite",
        fidelity=Fidelity.FAST,
        beam=Beam(particles=20),
        turns=1024,
        run_id="kicked",
    )
    run.kick(x=1e-4)
    run.start()
    run.wait()
    run.stop()
    fake = FakeOllama(reply="The measured tunes agree with the model within the resolution.")
    app = create_app(
        Catalog([write_demo(folder / "lattice")]),
        static_dir=None,
        runs=RunCatalog([folder / "runs"]),
        ai=AiConfig(model="fake:1b", url=fake.url),
    )
    with TestClient(app) as http:
        yield http, fake
    fake.close()


def test_the_status_says_which_model(setup):
    http, fake = setup
    assert http.get("/api/v1/ai").json() == {
        "enabled": True,
        "available": True,
        "provider": "local",
        "model": "fake:1b",
        "where": "on this server",
        "reason": f"fake:1b at {fake.url}",
    }


def test_explaining_a_check_hands_the_model_eicvibes_numbers(setup):
    http, fake = setup
    response = http.post("/api/v1/ai/explain-check", json={"run": "kicked", "turns": 512})
    assert response.status_code == 200 and response.headers["content-type"].startswith("text/plain")
    assert response.text.strip() == "The measured tunes agree with the model within the resolution."
    request = fake.requests[-1]
    assert request["think"] is False and request["options"]["num_predict"] == 220
    system, prompt = (m["content"] for m in request["messages"])
    assert "computed every number" in system
    assert "Run kicked: a ring tracked with the xsuite engine" in prompt
    assert "The kick: x 0.1 mrad." in prompt
    assert "(within 0.001)" in prompt  # the comparison is made in code, not by the model
    assert "EICViBE's reading of these numbers: In x the measured tune agrees with the model's" in prompt
    assert "recommend one of the next steps offered (no other)" in prompt


def test_asking_about_the_view_gathers_the_facts_on_the_server(setup):
    http, fake = setup
    summary = http.get("/api/v1/lattices").json()[0]
    digest = summary["digest"]
    lattice = http.get(f"/api/v1/lattices/{digest}").json()
    ring = next(b for b in lattice["branches"] if b["name"] == "ring")
    table = http.get(f"/api/v1/lattices/{digest}/branches/{ring['id']}").json()["table"]
    quad = table["id"][table["name"].index("QF_1")]
    body = {
        "question": "What is this element?",
        "digest": digest,
        "branch": ring["id"],
        "occurrence": quad,
        "tab": "optics",
    }
    response = http.post("/api/v1/ai/ask", json=body)
    assert response.status_code == 200
    prompt = fake.requests[-1]["messages"][1]["content"]
    assert "Shown: branch ring, a ring" in prompt and "The optics tab is open." in prompt
    assert (
        "Selected: QF_1" in prompt and "MagneticMultipoleP.Kn1 = 0.4 1/m^2" in prompt and "dr-q4-ps" in prompt
    )
    assert "Optics: not computed yet" in prompt  # nothing is started to answer a question
    assert prompt.endswith("Question: What is this element?\nAnswer in at most 80 words.")
    assert http.post("/api/v1/ai/ask", json=body | {"question": ""}).status_code == 422
    assert http.post("/api/v1/ai/ask", json=body | {"branch": "nope"}).status_code == 404


def test_off_unreachable_and_busy(tmp_path):
    off = TestClient(create_app(Catalog([]), static_dir=None, ai=AiConfig(enabled=False)))
    assert off.get("/api/v1/ai").json()["enabled"] is False
    assert off.post("/api/v1/ai/explain-check", json={"run": "x"}).status_code == 404
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    gone = TestClient(
        create_app(
            Catalog([write_demo(tmp_path)]),
            static_dir=None,
            ai=AiConfig(model="fake:1b", url=f"http://127.0.0.1:{port}"),
        )
    )
    status = gone.get("/api/v1/ai").json()
    assert status["enabled"] is True and status["available"] is False
    digest = gone.get("/api/v1/lattices").json()[0]["digest"]
    response = gone.post("/api/v1/ai/ask", json={"question": "?", "digest": digest, "branch": "ring"})
    assert response.status_code == 503 and "no model server" in response.json()["detail"]
    fake = FakeOllama()
    try:
        assistant = Assistant(lambda: LocalModel("fake:1b", fake.url))
        first = assistant.stream([{"role": "user", "content": "?"}], 10)
        next(first)  # answering: it holds the model
        with pytest.raises(Busy):
            assistant.stream([{"role": "user", "content": "?"}], 10)
        list(first)
        assert "".join(assistant.stream([{"role": "user", "content": "?"}], 10))  # free again
    finally:
        fake.close()


def plane(**values):
    fields = dict.fromkeys(schemas.PlaneCheck.model_fields) | {
        "spectrum": [],
        "phase_error": [],
        "beta_beating": [],
    }
    return schemas.PlaneCheck.model_construct(
        **(fields | {"orbit": [], "model_orbit": [], "drift_tune": []} | values)
    )


def test_the_verdict_is_decided_in_code():
    # The RCS run kicked by 1 mrad: 57 beam sizes, its tune where the model puts it at that amplitude.
    x = plane(measured_full=28.2525, measured_tune=0.2525, model_tune=28.2002, kick_sigma=56.8)
    y = plane(model_tune=27.4375, kick_sigma=0.06)
    detuning = schemas.Detuning(
        plane="x",
        amplitude=[2.3, 16.3],
        tune_x=[0.2014, 0.2578],
        tune_y=[0.4384, 0.4713],
        turns=512,
        expected_x=0.2525,
        expected_y=0.4688,
    )
    check = schemas.RunCheck.model_construct(
        x=x, y=y, detuning=detuning, kick={"x": 1e-3, "y": 0.0}, notes=[]
    )
    findings, steps = _verdict(check)
    assert "is amplitude-dependent detuning from a large kick, so the tracking agrees" in findings[0]
    assert steps[0].startswith("Kick again in x by about 0.053 mrad (three beam sizes instead of 56.8)")
    # Without the curve, the same numbers ask for it; a small difference agrees.
    findings, steps = _verdict(check.model_copy(update={"detuning": None}))
    assert (
        "may explain it" in findings[0]
        and steps[0] == "Tick 'tune vs amplitude' on the Check tab and check again."
    )
    agreeing = check.model_copy(
        update={"x": plane(measured_full=28.2003, measured_tune=0.2003, model_tune=28.2002)}
    )
    assert _verdict(agreeing)[0] == [
        "In x the measured tune agrees with the model's: the tracking is consistent."
    ]
