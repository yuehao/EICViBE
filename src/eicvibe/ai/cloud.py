"""Models served by OpenAI and Anthropic, through their APIs, for servers whose administrator chose one.

They do what the local model does (ADR 0009), with the same few features; what
differs is where the question goes: to the provider, with the facts EICViBE
gathered for it. They need an API key from the provider's developer platform
(platform.openai.com, console.anthropic.com): a ChatGPT or Claude subscription
does not let other applications use its models. ``OpenAIModel`` also talks to
servers with an OpenAI-compatible API, at their own base URL.
"""

import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

from .local import AiUnavailable, Message

OPENAI_URL = "https://api.openai.com/v1"
ANTHROPIC_URL = "https://api.anthropic.com"
ANTHROPIC_VERSION = "2023-06-01"


def _open(request: urllib.request.Request, timeout: float, who: str):
    try:
        return urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as error:
        body = error.read().decode(errors="replace")
        try:
            message = json.loads(body).get("error", {}).get("message") or body
        except ValueError:
            message = body
        if error.code in (401, 403):
            raise AiUnavailable(f"{who} refused the API key ({message[:200]})") from None
        raise AiUnavailable(f"{who}: {error.code} {message[:300]}") from None
    except OSError as error:
        raise AiUnavailable(f"{who} cannot be reached at {request.full_url} ({error})") from None


def _events(response) -> Iterator[dict]:
    """The JSON of each ``data:`` line of a server-sent event stream; stops at ``[DONE]``."""
    with response:
        for raw in response:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                return
            yield json.loads(data)


@dataclass(frozen=True)
class OpenAIModel:
    """A model of OpenAI's Chat Completions API, or of a server with the same API at ``url``."""

    model: str
    api_key: str = field(repr=False)
    url: str = OPENAI_URL
    timeout: float = 60.0
    provider = "openai"
    concurrency = 4

    @property
    def where(self) -> str:
        host = urllib.parse.urlparse(self.url).hostname or self.url
        return "by OpenAI" if host == "api.openai.com" else f"on {host}"

    def _request(self, path: str, body: dict | None = None) -> urllib.request.Request:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        data = None if body is None else json.dumps(body).encode()
        return urllib.request.Request(f"{self.url.rstrip('/')}{path}", data=data, headers=headers)

    def models(self) -> list[str]:
        with _open(self._request("/models"), min(self.timeout, 10.0), "OpenAI") as response:
            return sorted(m["id"] for m in json.load(response).get("data", []))

    def status(self) -> tuple[bool, str]:
        if not self.api_key:
            return False, "no API key: give one on the administration page, or set OPENAI_API_KEY"
        if not self.model:
            return False, "no model chosen"
        try:
            models = self.models()
        except AiUnavailable as error:
            return False, str(error)
        if models and self.model not in models:
            return False, f"the API does not offer {self.model}"
        return True, f"{self.model} {self.where}"

    def stream(self, messages: Sequence[Message], *, max_tokens: int = 300) -> Iterator[str]:
        # Reasoning models spend part of the cap thinking; the prompt bounds the answer's length.
        body = {
            "model": self.model,
            "messages": list(messages),
            "stream": True,
            "max_completion_tokens": max(4 * max_tokens, 1024),
        }
        for event in _events(_open(self._request("/chat/completions", body), self.timeout, "OpenAI")):
            if "error" in event:
                raise AiUnavailable(f"OpenAI: {event['error'].get('message', event['error'])}")
            for choice in event.get("choices", []):
                if text := (choice.get("delta") or {}).get("content"):
                    yield text


@dataclass(frozen=True)
class AnthropicModel:
    """A Claude model of Anthropic's Messages API."""

    model: str
    api_key: str = field(repr=False)
    url: str = ANTHROPIC_URL
    timeout: float = 60.0
    provider = "anthropic"
    concurrency = 4

    @property
    def where(self) -> str:
        host = urllib.parse.urlparse(self.url).hostname or self.url
        return "by Anthropic" if host == "api.anthropic.com" else f"on {host}"

    def _request(self, path: str, body: dict | None = None) -> urllib.request.Request:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        data = None if body is None else json.dumps(body).encode()
        return urllib.request.Request(f"{self.url.rstrip('/')}{path}", data=data, headers=headers)

    def models(self) -> list[str]:
        with _open(self._request("/v1/models?limit=100"), min(self.timeout, 10.0), "Anthropic") as response:
            return [m["id"] for m in json.load(response).get("data", [])]

    def status(self) -> tuple[bool, str]:
        if not self.api_key:
            return False, "no API key: give one on the administration page, or set ANTHROPIC_API_KEY"
        if not self.model:
            return False, "no model chosen"
        try:
            models = self.models()
        except AiUnavailable as error:
            return False, str(error)
        if models and self.model not in models:
            return False, f"the API does not offer {self.model}"
        return True, f"{self.model} {self.where}"

    def stream(self, messages: Sequence[Message], *, max_tokens: int = 300) -> Iterator[str]:
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        body = {
            "model": self.model,
            "max_tokens": max(2 * max_tokens, 512),
            "messages": [m for m in messages if m["role"] != "system"],
            "stream": True,
        }
        if system:
            body["system"] = system
        for event in _events(_open(self._request("/v1/messages", body), self.timeout, "Anthropic")):
            kind = event.get("type")
            if kind == "error":
                raise AiUnavailable(f"Anthropic: {event.get('error', {}).get('message', event)}")
            if kind == "content_block_delta" and event.get("delta", {}).get("type") == "text_delta":
                yield event["delta"]["text"]
            elif kind == "message_stop":
                return
