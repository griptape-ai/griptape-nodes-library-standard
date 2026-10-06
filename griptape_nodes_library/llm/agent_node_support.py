from __future__ import annotations

import json
from typing import Any

from griptape_nodes.retained_mode.events.agent_events import ProviderConfig

from griptape_nodes_library.llm.model_config import ModelConfig, ModelProvider

DEFAULT_CLOUD_MODEL = "claude-sonnet-5"


def default_cloud_model_config() -> ModelConfig:
    return ModelConfig(provider=ModelProvider.GRIPTAPE_CLOUD, model=DEFAULT_CLOUD_MODEL)


def model_config_for_provider(provider_config: ProviderConfig, model: str) -> ModelConfig:
    """Describe `model` on an engine-configured third-party provider. Holds the secret's name, not its value."""
    match provider_config.type:
        case ModelProvider.OLLAMA:
            provider = ModelProvider.OLLAMA
        case ModelProvider.LMSTUDIO:
            provider = ModelProvider.LMSTUDIO
        case _:
            provider = ModelProvider.OPENAI_COMPATIBLE
    return ModelConfig(
        provider=provider,
        model=model,
        base_url=provider_config.base_url or None,
        api_key_secret=provider_config.api_key_secret_name or None,
    )


def parse_agent_memory(memory_data: Any) -> dict[str, Any] | None:
    if isinstance(memory_data, str):
        if not memory_data.strip():
            return None
        try:
            memory_data = json.loads(memory_data)
        except json.JSONDecodeError:
            return None
    if not isinstance(memory_data, dict) or not memory_data:
        return None
    return memory_data
