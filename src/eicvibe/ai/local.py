"""A local language model: Ollama's HTTP API, on this machine or a server of the lab's.

The model is a helper (ADR 0009). It reads what EICViBE computed and puts it
into words; it never acts. Thinking is off and answers are capped, so a task
takes seconds: a 27B model on an M3 Max reads about 190 tokens a second and
writes about 20, and a prompt it has seen before costs nothing to read again.
Nothing leaves the host the URL names.
"""

import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

DEFAULT_URL = "http://127.0.0.1:11434"

type Message = dict[str, str]


class AiUnavailable(RuntimeError):
    """The model server cannot be reached, or does not have the model."""


@dataclass(frozen=True)
class LocalModel:
    """A model served by Ollama at ``url``; with no model named, the first one it has."""

    model: str = ""
    url: str = DEFAULT_URL
    timeout: float = 60.0
    """Seconds to connect, and to wait for each piece of an answer."""
    context: int = 8192
    keep_alive: str = "30m"
    """How long the server keeps the model loaded after a request: a cold start costs seconds."""

    provider = "local"
    concurrency = 1
    """A local model answers one request at a time."""

    @property
    def where(self) -> str:
        host = urllib.parse.urlparse(self.url).hostname or self.url
        return "on this server" if host in ("127.0.0.1", "localhost", "::1") else f"on {host}"

    def models(self) -> list[str]:
        """The models the server has."""
        try:
            with urllib.request.urlopen(f"{self.url}/api/tags", timeout=min(self.timeout, 5.0)) as response:
                return [m.get("name") for m in json.load(response).get("models", [])]
        except (OSError, ValueError) as error:
            raise AiUnavailable(f"no model server at {self.url} ({error})") from None

    def resolved(self) -> str:
        """The model's name: the one given, or the first the server has."""
        if self.model:
            return self.model
        models = self.models()
        if not models:
            raise AiUnavailable(f"the model server at {self.url} has no models (ollama pull MODEL)")
        return models[0]

    def status(self) -> tuple[bool, str]:
        """Whether the server answers and has the model, and if not, why."""
        try:
            models = self.models()
        except AiUnavailable as error:
            return False, str(error)
        if not self.model:
            if not models:
                return False, f"the model server at {self.url} has no models (ollama pull MODEL)"
            return True, f"{models[0]} at {self.url}"
        if self.model not in models and f"{self.model}:latest" not in models:
            return False, f"{self.url} does not have {self.model} (it has {', '.join(models) or 'none'})"
        return True, f"{self.model} at {self.url}"

    def stream(
        self, messages: Sequence[Message], *, max_tokens: int = 300, temperature: float = 0.2
    ) -> Iterator[str]:
        """The answer to ``messages``, piece by piece as the model writes it, at most ``max_tokens``."""
        pieces = self._chat(messages, stream=True, max_tokens=max_tokens, temperature=temperature)
        yield from _answer_only(p.get("message", {}).get("content") or "" for p in pieces)

    def answer(
        self,
        messages: Sequence[Message],
        *,
        schema: dict | None = None,
        max_tokens: int = 300,
        temperature: float = 0.0,
    ) -> str:
        """The whole answer at once; with ``schema`` (JSON Schema) it is JSON of that shape."""
        [reply] = list(
            self._chat(messages, stream=False, max_tokens=max_tokens, temperature=temperature, schema=schema)
        )
        return reply.get("message", {}).get("content", "")

    def _chat(
        self, messages, *, stream: bool, max_tokens: int, temperature: float, schema: dict | None = None
    ) -> Iterator[dict]:
        body = {
            "model": self.resolved(),
            "messages": list(messages),
            "stream": stream,
            "think": False,
            "keep_alive": self.keep_alive,
            "options": {"num_predict": max_tokens, "temperature": temperature, "num_ctx": self.context},
        }
        if schema is not None:
            body["format"] = schema
        request = urllib.request.Request(
            f"{self.url}/api/chat",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            response = urllib.request.urlopen(request, timeout=self.timeout)
        except urllib.error.HTTPError as error:
            raise AiUnavailable(f"{self.model}: {error.read().decode(errors='replace')[:300]}") from None
        except OSError as error:
            raise AiUnavailable(f"no model server at {self.url} ({error})") from None
        with response:
            for line in response:
                if not line.strip():
                    continue
                piece = json.loads(line)
                if "error" in piece:
                    raise AiUnavailable(f"{self.model}: {piece['error']}")
                yield piece
                if piece.get("done"):
                    return


def _visible(text: str, final: bool = False) -> tuple[str, bool]:
    """The answer in ``text`` so far, and whether it has ended.

    Some models write ``<think>...</think>`` even with thinking off: reasoning they
    begin with is left out, and a closing tag after the answer ends it (what
    follows repeats it). Until ``final``, a possible start of a tag is held back.
    """
    answer = text.lstrip()
    if answer.startswith("<think>"):
        if "</think>" not in answer:
            return "", False
        answer = answer.split("</think>", 1)[1].lstrip()
    elif not final and "<think>".startswith(answer):
        return "", False
    if "</think>" in answer:
        return answer.split("</think>", 1)[0].rstrip(), True
    if not final:
        for k in range(len("</think>") - 1, 0, -1):
            if answer.endswith("</think>"[:k]):
                return answer[:-k], False
    return answer, False


def _answer_only(pieces: Iterator[str]) -> Iterator[str]:
    """The answer, piece by piece, without reasoning text (see :func:`_visible`)."""
    text, sent = "", 0
    for piece in pieces:
        text += piece
        answer, ended = _visible(text)
        if len(answer) > sent:
            yield answer[sent:]
            sent = len(answer)
        if ended:
            return
    answer, _ = _visible(text, final=True)
    if len(answer) > sent:
        yield answer[sent:]
