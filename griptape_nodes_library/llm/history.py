"""Drop the oldest conversation runs so replayed history fits the model's context.

Replaces griptape's `ConversationMemory(autoprune=True)`. Token counts are estimated
at four characters per token, as griptape's `SimpleTokenizer` did.
"""

from __future__ import annotations

import logging

from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter, ModelRequest, UserPromptPart

from griptape_nodes_library.llm.model_config import ModelConfig, ModelProvider

CHARS_PER_TOKEN = 4

logger = logging.getLogger("griptape_nodes")

# History budgets in estimated tokens. Local servers keep griptape's Ollama limit; hosted
# and OpenAI-compatible providers get headroom below their smallest context.
_LOCAL_BUDGET = 2_000
_HOSTED_BUDGET = 100_000
_CLOUD_BUDGET = 500_000
_HISTORY_TOKEN_BUDGETS: dict[ModelProvider, int] = {
    ModelProvider.GRIPTAPE_CLOUD: _CLOUD_BUDGET,
    ModelProvider.OLLAMA: _LOCAL_BUDGET,
    ModelProvider.LMSTUDIO: _LOCAL_BUDGET,
}


def history_token_budget(config: ModelConfig) -> int:
    return _HISTORY_TOKEN_BUDGETS.get(config.provider, _HOSTED_BUDGET)


def _estimate_tokens(messages: list[ModelMessage]) -> int:
    return len(ModelMessagesTypeAdapter.dump_json(messages)) // CHARS_PER_TOKEN


def _starts_run(message: ModelMessage) -> bool:
    return isinstance(message, ModelRequest) and any(isinstance(p, UserPromptPart) for p in message.parts)


def prune_history(messages: list[ModelMessage], budget: int) -> list[ModelMessage]:
    """Drop whole runs, oldest first, until `messages` fit `budget`. The latest run is always kept."""
    if _estimate_tokens(messages) <= budget:
        return messages
    starts = [i for i, m in enumerate(messages) if i > 0 and _starts_run(m)]
    start = next((i for i in starts if _estimate_tokens(messages[i:]) <= budget), starts[-1] if starts else 0)
    if start:
        logger.warning(
            "Replaying %d of %d history messages to fit the model's context.", len(messages) - start, len(messages)
        )
    return messages[start:]
