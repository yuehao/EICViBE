"""The AI features' models (ADR 0009): local by default, OpenAI or Anthropic if an administrator chooses.

EICViBE works without them, and they never act. :class:`AiConfig` says which model
answers; :mod:`eicvibe.api.assist` holds the features (an explanation of a run's
check, a question about what the viewer shows).
"""

from .cloud import AnthropicModel, OpenAIModel
from .config import PROVIDERS, AiConfig, Model, load_config, save_config
from .local import DEFAULT_URL, AiUnavailable, LocalModel, Message

__all__ = [
    "DEFAULT_URL",
    "PROVIDERS",
    "AiConfig",
    "AiUnavailable",
    "AnthropicModel",
    "LocalModel",
    "Message",
    "Model",
    "OpenAIModel",
    "load_config",
    "save_config",
]
