"""The AI features' model, chosen by administrators (ADR 0009): on by default, local; keys kept unseen."""

import json
import os
import stat

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from eicvibe.ai import AiConfig  # noqa: E402
from eicvibe.api import Catalog, create_app  # noqa: E402
from eicvibe.api.demo import write_demo  # noqa: E402
from eicvibe.api.runs import RunCatalog  # noqa: E402
from eicvibe.va import VirtualAccelerator, load_source  # noqa: E402
from tests.ai.fake_ollama import KEY, FakeOllama  # noqa: E402


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    fake = FakeOllama(models=("fake:1b", "claude-test"), reply="Ready.")
    lattice_file = write_demo(tmp_path / "files")
    va = VirtualAccelerator(load_source(lattice_file), tmp_path / "state")
    password = va.first_admin()
    app = create_app(
        Catalog([tmp_path / "files"]),
        static_dir=None,
        runs=RunCatalog([va.runs]),
        va=va,
        ai=AiConfig(url=fake.url),  # the server's default: local, its first model
    )
    with TestClient(app) as http:
        admin = TestClient(app)
        assert admin.post("/api/v1/va/login", json={"name": "admin", "password": password}).status_code == 200
        made = admin.post("/api/v1/va/users", json={"name": "ann", "role": "physicist"}).json()
        ann = TestClient(app)
        ann.post("/api/v1/va/login", json={"name": "ann", "password": made["password"]})
        yield http, admin, ann, va, fake
    fake.close()


def test_on_by_default_with_the_local_model_and_only_administrators_see_its_settings(setup):
    http, admin, ann, _, _ = setup
    status = http.get("/api/v1/ai").json()
    assert (status["enabled"], status["available"], status["provider"], status["model"]) == (
        True,
        True,
        "local",
        "fake:1b",
    )
    assert status["where"] == "on this server"
    assert http.get("/api/v1/va/ai").status_code == 401
    assert ann.get("/api/v1/va/ai").status_code == 403
    settings = admin.get("/api/v1/va/ai").json()
    assert (settings["provider"], settings["saved"], settings["key"]) == ("local", False, None)
    assert settings["providers"]["anthropic"]["model"] == "claude-opus-5-5"
    assert settings["providers"]["openai"]["key_variable"] == "OPENAI_API_KEY"


def test_an_administrator_tries_then_chooses_claude_and_the_key_is_kept_unseen(setup):
    http, admin, _, va, fake = setup
    choice = {
        "enabled": True,
        "provider": "anthropic",
        "model": "claude-test",
        "url": fake.url,
        "api_key": KEY,
    }
    tried = admin.post("/api/v1/va/ai/test", json=choice).json()
    assert (
        tried["available"] and tried["answer"] == "Ready." and tried["models"] == ["fake:1b", "claude-test"]
    )
    assert http.get("/api/v1/ai").json()["provider"] == "local"  # trying saves nothing
    settings = admin.put("/api/v1/va/ai", json=choice | {"reason": "faster answers"}).json()
    assert (settings["provider"], settings["key"], settings["key_hint"], settings["saved"]) == (
        "anthropic",
        "stored",
        "…1234",
        True,
    )
    status = http.get("/api/v1/ai").json()
    assert (status["provider"], status["model"], status["available"]) == ("anthropic", "claude-test", True)
    # The viewer's question now goes to the provider, with the key.
    digest = http.get("/api/v1/lattices").json()[0]["digest"]
    assert (
        http.post("/api/v1/ai/ask", json={"question": "?", "digest": digest, "branch": "ring"}).text
        == "Ready."
    )
    assert fake.headers[-1]["X-Api-Key"] == KEY
    # The key is kept for its owner only, logged as changed, and sent back nowhere.
    path = va.state / "ai.json"
    assert (
        stat.S_IMODE(os.stat(path).st_mode) == 0o600
        and json.loads(path.read_text())["keys"]["anthropic"] == KEY
    )
    actions = admin.get("/api/v1/va/actions").json()
    entry = next(a for a in actions if a["action"] == "ai settings")
    assert entry["detail"]["key"] == "changed" and entry["reason"] == "faster answers"
    for page in ("/api/v1/va/ai", "/api/v1/va/actions", "/api/v1/ai"):
        assert KEY not in admin.get(page).text


def test_switching_providers_keeps_each_key_and_off_turns_the_features_off(setup):
    http, admin, _, _, fake = setup
    claude = {"provider": "anthropic", "model": "claude-test", "url": fake.url, "api_key": KEY}
    admin.put("/api/v1/va/ai", json=claude)
    openai = admin.put(
        "/api/v1/va/ai", json={"provider": "openai", "model": "fake:1b", "url": f"{fake.url}/v1"}
    ).json()
    assert openai["key"] is None  # the Claude key is Claude's
    status = http.get("/api/v1/ai").json()
    assert status["available"] is False and "no API key" in status["reason"]
    back = admin.put("/api/v1/va/ai", json=claude | {"api_key": None}).json()
    assert back["key"] == "stored"  # kept
    assert (
        admin.put("/api/v1/va/ai", json=claude | {"api_key": None, "clear_key": True}).json()["key"] is None
    )
    admin.put("/api/v1/va/ai", json={"enabled": False, "provider": "local"})
    assert http.get("/api/v1/ai").json()["enabled"] is False
    digest = http.get("/api/v1/lattices").json()[0]["digest"]
    assert (
        http.post("/api/v1/ai/ask", json={"question": "?", "digest": digest, "branch": "ring"}).status_code
        == 404
    )
