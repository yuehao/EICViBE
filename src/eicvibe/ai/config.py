"""Which model the AI features use: a local one by default, or OpenAI's or Anthropic's (ADR 0009).

With nothing configured the features are on and ask the local Ollama server for
the first model it has. An administrator chooses otherwise on the administration
page, where the choice is kept in the virtual accelerator's state folder
(``ai.json``, readable by its owner only); a server without one takes it from
its command line or environment. A provider's API key is the one stored there,
or else its usual environment variable (``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``).
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from .cloud import ANTHROPIC_URL, OPENAI_URL, AnthropicModel, OpenAIModel
from .local import DEFAULT_URL, LocalModel

type Model = LocalModel | OpenAIModel | AnthropicModel

#: Each provider: what it is called, its URL, where else its key may be, and its model if none is chosen.
PROVIDERS = {
    "local": {"label": "Local model (Ollama)", "url": DEFAULT_URL, "key_variable": None, "model": ""},
    "openai": {"label": "OpenAI", "url": OPENAI_URL, "key_variable": "OPENAI_API_KEY", "model": ""},
    "anthropic": {
        "label": "Anthropic Claude",
        "url": ANTHROPIC_URL,
        "key_variable": "ANTHROPIC_API_KEY",
        "model": "claude-opus-5-5",
    },
}

#: For a server without a virtual accelerator: EICVIBE_AI_OFF=1 turns the features off.
OFF_VARIABLE = "EICVIBE_AI_OFF"
PROVIDER_VARIABLE = "EICVIBE_AI_PROVIDER"
MODEL_VARIABLE = "EICVIBE_AI_MODEL"
URL_VARIABLE = "EICVIBE_AI_URL"


class AiConfig(BaseModel):
    """Whether the AI features are on, and which model answers."""

    enabled: bool = True
    provider: Literal["local", "openai", "anthropic"] = "local"
    model: str = Field("", description="Empty: the local server's first model, or the provider's default")
    url: str = Field("", description="Empty: the provider's own")
    keys: dict[str, str] = Field(
        default_factory=dict, repr=False, description="API keys by provider; none: the provider's variable"
    )

    @classmethod
    def from_environment(cls) -> AiConfig:
        """On, with the local model, unless the environment says otherwise."""
        off = os.environ.get(OFF_VARIABLE, "").strip().lower() not in ("", "0", "false", "no")
        provider = os.environ.get(PROVIDER_VARIABLE, "").strip() or "local"
        return cls(
            enabled=not off,
            provider=provider,
            model=os.environ.get(MODEL_VARIABLE, "").strip(),
            url=os.environ.get(URL_VARIABLE, "").strip(),
        )

    def key(self) -> tuple[str, str | None]:
        """The API key, and where it comes from: ``stored``, an environment variable's name, or None."""
        if self.keys.get(self.provider):
            return self.keys[self.provider], "stored"
        variable = PROVIDERS[self.provider]["key_variable"]
        if variable and os.environ.get(variable):
            return os.environ[variable], variable
        return "", None

    def make(self) -> Model | None:
        """The model to ask, or None when the features are off."""
        if not self.enabled:
            return None
        url = self.url or PROVIDERS[self.provider]["url"]
        model = self.model or PROVIDERS[self.provider]["model"]
        if self.provider == "local":
            return LocalModel(model=model, url=url)
        key, _ = self.key()
        if self.provider == "openai":
            return OpenAIModel(model=model, api_key=key, url=url)
        return AnthropicModel(model=model, api_key=key, url=url)


def load_config(path: Path) -> AiConfig | None:
    """The configuration kept at ``path``, or None if there is none."""
    if not path.is_file():
        return None
    return AiConfig.model_validate_json(path.read_text())


def save_config(path: Path, config: AiConfig) -> None:
    """Keep ``config`` at ``path``, readable by its owner only (it may hold an API key)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=".ai-", suffix=".json")
    try:
        os.fchmod(handle, 0o600)
        with os.fdopen(handle, "w") as file:
            json.dump(config.model_dump(), file, indent=1)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
