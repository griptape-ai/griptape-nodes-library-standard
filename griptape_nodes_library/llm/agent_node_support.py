from __future__ import annotations

import json
from typing import Any

from griptape_nodes_library.llm.model_config import ModelConfig, cloud_model_config

DEFAULT_CLOUD_MODEL = "claude-sonnet-5"


def default_cloud_model_config() -> ModelConfig:
    return cloud_model_config(DEFAULT_CLOUD_MODEL)


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
