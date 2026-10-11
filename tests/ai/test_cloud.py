"""OpenAI's and Anthropic's APIs, and the configuration that chooses among the models, against stand-ins."""

import os
import stat

import pytest

from eicvibe.ai import (
    AiConfig,
    AiUnavailable,
    AnthropicModel,
    LocalModel,
    OpenAIModel,
    load_config,
    save_config,
)
from tests.ai.fake_ollama import KEY, FakeOllama

MESSAGES = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Why?"}]


@pytest.fixture
def fake():
    server = FakeOllama(models=("model-a", "model-b"), reply="Amplitude detuning explains it.")
    yield server
    server.close()


def test_openai_streams_with_its_key(fake):
    model = OpenAIModel("model-a", KEY, url=f"{fake.url}/v1")
    assert model.status() == (True, "model-a on 127.0.0.1") and model.models() == ["model-a", "model-b"]
    assert "".join(model.stream(MESSAGES, max_tokens=100)) == "Amplitude detuning explains it."
    request, headers = fake.requests[-1], fake.headers[-1]
    assert headers["Authorization"] == f"Bearer {KEY}" and request["messages"] == MESSAGES
    assert request["max_completion_tokens"] == 1024 and request["stream"] is True  # room for reasoning models
    assert OpenAIModel("model-a", KEY).where == "by OpenAI"


def test_anthropic_streams_with_its_key_and_a_separate_system_prompt(fake):
    model = AnthropicModel("model-b", KEY, url=fake.url)
    assert model.status() == (True, "model-b on 127.0.0.1")
    assert "".join(model.stream(MESSAGES, max_tokens=100)) == "Amplitude detuning explains it."
    request, headers = fake.requests[-1], fake.headers[-1]
    assert headers["X-Api-Key"] == KEY and headers["Anthropic-Version"] == "2023-06-01"
    assert request["system"] == "Be brief." and request["messages"] == [{"role": "user", "content": "Why?"}]
    assert AnthropicModel("claude-opus-5-5", KEY).where == "by Anthropic"


def test_a_wrong_key_a_missing_key_and_an_unknown_model_are_said(fake):
    wrong = AnthropicModel("model-b", "sk-wrong", url=fake.url)
    available, reason = wrong.status()
    assert not available and reason.startswith("Anthropic refused the API key")
    with pytest.raises(AiUnavailable, match="refused the API key"):
        list(OpenAIModel("model-a", "sk-wrong", url=f"{fake.url}/v1").stream(MESSAGES))
    assert OpenAIModel("model-a", "", url=f"{fake.url}/v1").status() == (
        False,
        "no API key: give one on the administration page, or set OPENAI_API_KEY",
    )
    assert OpenAIModel("", KEY, url=f"{fake.url}/v1").status() == (False, "no model chosen")
    assert AnthropicModel("model-z", KEY, url=fake.url).status() == (False, "the API does not offer model-z")


def test_the_configuration_chooses_the_model(monkeypatch, tmp_path):
    for variable in (
        "EICVIBE_AI_OFF",
        "EICVIBE_AI_PROVIDER",
        "EICVIBE_AI_MODEL",
        "EICVIBE_AI_URL",
        "ANTHROPIC_API_KEY",
    ):
        monkeypatch.delenv(variable, raising=False)
    assert AiConfig.from_environment().make() == LocalModel()  # on, local, its first model: the default
    monkeypatch.setenv("EICVIBE_AI_OFF", "1")
    assert AiConfig.from_environment().make() is None
    claude = AiConfig(provider="anthropic").make()
    assert claude == AnthropicModel("claude-opus-5-5", "") and AiConfig(provider="anthropic").key() == (
        "",
        None,
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-env")
    assert AiConfig(provider="anthropic").key() == ("sk-from-env", "ANTHROPIC_API_KEY")
    stored = AiConfig(provider="openai", model="model-a", keys={"openai": KEY, "anthropic": "sk-other"})
    assert stored.key() == (KEY, "stored") and stored.make() == OpenAIModel("model-a", KEY)
    assert KEY not in repr(stored) and KEY not in repr(stored.make())
    path = tmp_path / "state" / "ai.json"
    save_config(path, stored)
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600  # it holds keys
    assert load_config(path) == stored and load_config(tmp_path / "none.json") is None
