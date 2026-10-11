"""The local model client, against a stand-in Ollama server."""

import socket

import pytest

from eicvibe.ai import AiUnavailable, LocalModel
from tests.ai.fake_ollama import FakeOllama


@pytest.fixture
def fake():
    server = FakeOllama(reply="Amplitude detuning explains it.")
    yield server
    server.close()


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_status_says_whether_the_model_is_there(fake):
    assert LocalModel("fake:1b", fake.url).status() == (True, f"fake:1b at {fake.url}")
    available, reason = LocalModel("other:27b", fake.url).status()
    assert not available and "does not have other:27b (it has fake:1b)" in reason
    available, reason = LocalModel("fake:1b", f"http://127.0.0.1:{free_port()}").status()
    assert not available and reason.startswith("no model server at")


def test_answers_stream_short_and_without_thinking(fake):
    model = LocalModel("fake:1b", fake.url)
    pieces = list(model.stream([{"role": "user", "content": "why?"}], max_tokens=50))
    assert "".join(pieces).strip() == "Amplitude detuning explains it." and len(pieces) == 4
    [request] = fake.requests
    assert request["think"] is False and request["stream"] is True and request["keep_alive"] == "30m"
    assert request["options"]["num_predict"] == 50


def test_a_structured_answer_and_errors(fake):
    model = LocalModel("fake:1b", fake.url)
    schema = {"type": "object", "properties": {"kind": {"enum": ["Quadrupole", "Bend"]}}}
    assert (
        model.answer([{"role": "user", "content": "which?"}], schema=schema)
        == "Amplitude detuning explains it."
    )
    assert fake.requests[-1]["format"] == schema and fake.requests[-1]["stream"] is False
    fake.error = "model failed to load"
    with pytest.raises(AiUnavailable, match="model failed to load"):
        list(model.stream([{"role": "user", "content": "why?"}]))
    with pytest.raises(AiUnavailable, match="no model server"):
        list(
            LocalModel("fake:1b", f"http://127.0.0.1:{free_port()}").stream(
                [{"role": "user", "content": "?"}]
            )
        )


def test_without_a_model_named_it_takes_the_first_the_server_has(fake):
    model = LocalModel(url=fake.url)
    assert model.status() == (True, f"fake:1b at {fake.url}") and model.resolved() == "fake:1b"
    assert (
        "".join(model.stream([{"role": "user", "content": "?"}])).strip() == "Amplitude detuning explains it."
    )
    assert fake.requests[-1]["model"] == "fake:1b"
    assert model.where == "on this server" and LocalModel(url="http://gpu.lab.example:11434").where == (
        "on gpu.lab.example"
    )


@pytest.mark.parametrize(
    ("pieces", "answer"),
    [
        (["The tune ", "is 0.2", "002."], "The tune is 0.2002."),
        (["Answer.", "</th", "ink>Answer again"], "Answer."),  # a leaked tag ends it; the repeat is dropped
        (["<thi", "nk>reasoning", "</think>", "\nAnswer."], "Answer."),  # reasoning first is left out
        (["x <", " y"], "x < y"),
    ],
)
def test_reasoning_text_is_left_out(pieces, answer):
    from eicvibe.ai.local import _answer_only

    assert "".join(_answer_only(iter(pieces))).strip() == answer


def test_a_streamed_answer_stops_at_a_leaked_tag():
    fake = FakeOllama(reply="It is a quadrupole. </think> It is a quadrupole.")
    try:
        assert "".join(
            LocalModel("fake:1b", fake.url).stream([{"role": "user", "content": "?"}])
        ).strip() == ("It is a quadrupole.")
    finally:
        fake.close()
