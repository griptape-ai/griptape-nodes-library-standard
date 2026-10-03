"""Budget-gate and meter model calls that go straight to a provider rather than through Griptape Cloud.

Whether a call is direct depends on the driver actually installed, not on the node: an Agent
on the Griptape Cloud provider goes through the proxy, which enforces budgets itself, while the
same Agent with a connected Anthropic config calls Anthropic directly. `require_driver_access`
picks the right gate for the driver, and the returned meter reports the call's cost afterwards.

Not vendored: the price table and driver classes are this library's business. The vendorable
contract lives in `model_invocation.py`.
"""

from __future__ import annotations

import logging
from decimal import ROUND_HALF_UP, Decimal
from typing import TYPE_CHECKING, Any, TypeVar

from griptape.drivers.image_generation.griptape_cloud import GriptapeCloudImageGenerationDriver
from griptape.drivers.prompt.amazon_bedrock import AmazonBedrockPromptDriver
from griptape.drivers.prompt.anthropic import AnthropicPromptDriver
from griptape.drivers.prompt.cohere import CoherePromptDriver
from griptape.drivers.prompt.griptape_cloud import GriptapeCloudPromptDriver
from griptape.drivers.prompt.grok import GrokPromptDriver
from griptape.drivers.prompt.ollama import OllamaPromptDriver
from griptape.drivers.prompt.openai import OpenAiChatPromptDriver
from griptape.events import EventListener, FinishPromptEvent

from griptape_nodes_library.utils.model_invocation import (
    report_model_usage_sync,
    require_model_access_sync,
    require_model_invocation_sync,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from griptape_nodes.exe_types.node_types import BaseNode

logger = logging.getLogger("griptape_nodes")

T = TypeVar("T")

# Cloud-proxied drivers are budgeted server-side; Ollama is local and costs nothing.
_NOT_DIRECT = (GriptapeCloudPromptDriver, GriptapeCloudImageGenerationDriver, OllamaPromptDriver)

# Most specific first: GrokPromptDriver subclasses OpenAiChatPromptDriver. Values are the provider
# strings Cloud groups usage dashboards on, so keep them stable and lowercase.
_PROVIDERS: tuple[tuple[type, str], ...] = (
    (GrokPromptDriver, "xai"),
    (OpenAiChatPromptDriver, "openai"),
    (AnthropicPromptDriver, "anthropic"),
    (AmazonBedrockPromptDriver, "amazon_bedrock"),
    (CoherePromptDriver, "cohere"),
)

# Provider list prices in USD per 1,000,000 (input, output) tokens -- which is also micro-USD per
# token. No markup: this is the customer's own spend. Taken from griptape-cloud's
# `credits/chat_token_pricing.py` (fetched there 2026-07-31 / 2026-08-26). Direct Anthropic is that
# file's Bedrock regional rate without its 1.1x regional premium. Providers missing here are
# budget-checked but not reported.
DIRECT_PROVIDER_PRICES: dict[str, dict[str, tuple[Decimal, Decimal]]] = {
    "openai": {
        "gpt-5.6-sol": (Decimal("5.00"), Decimal("30.00")),
        "gpt-5.6-terra": (Decimal("2.00"), Decimal("12.00")),
        "gpt-5.6-luna": (Decimal("0.20"), Decimal("1.20")),
        "gpt-5.5": (Decimal("5.00"), Decimal("30.00")),
        "gpt-5.4": (Decimal("2.50"), Decimal("15.00")),
        "gpt-5.2": (Decimal("1.75"), Decimal("14.00")),
        "gpt-5.1": (Decimal("1.25"), Decimal("10.00")),
        "gpt-5": (Decimal("1.25"), Decimal("10.00")),
        "gpt-5-mini": (Decimal("0.25"), Decimal("2.00")),
        "gpt-5-nano": (Decimal("0.05"), Decimal("0.40")),
        "gpt-4.1": (Decimal("2.00"), Decimal("8.00")),
        "gpt-4.1-mini": (Decimal("0.40"), Decimal("1.60")),
        "gpt-4.1-nano": (Decimal("0.10"), Decimal("0.40")),
        "gpt-4o": (Decimal("2.50"), Decimal("10.00")),
        "o4-mini": (Decimal("1.10"), Decimal("4.40")),
        "o3": (Decimal("2.00"), Decimal("8.00")),
        "o3-mini": (Decimal("1.10"), Decimal("4.40")),
        "o1": (Decimal("15.00"), Decimal("60.00")),
    },
    "anthropic": {
        "claude-opus-4-7": (Decimal("5.00"), Decimal("25.00")),
        "claude-sonnet-4-6": (Decimal("3.00"), Decimal("15.00")),
        "claude-haiku-4-5": (Decimal("1.00"), Decimal("5.00")),
    },
    "amazon_bedrock": {
        "us.anthropic.claude-opus-4-7": (Decimal("5.50"), Decimal("27.50")),
        "us.anthropic.claude-sonnet-4-6": (Decimal("3.30"), Decimal("16.50")),
        "us.anthropic.claude-sonnet-4-5-20250929-v1:0": (Decimal("3.30"), Decimal("16.50")),
        "us.anthropic.claude-haiku-4-5-20251001-v1:0": (Decimal("1.10"), Decimal("5.50")),
        "deepseek.v3.2": (Decimal("0.62"), Decimal("1.85")),
        "us.deepseek.r1-v1:0": (Decimal("1.35"), Decimal("5.40")),
        "us.meta.llama3-3-70b-instruct-v1:0": (Decimal("0.72"), Decimal("0.72")),
        "us.meta.llama3-1-70b-instruct-v1:0": (Decimal("0.72"), Decimal("0.72")),
    },
}


def is_direct_provider(driver: Any) -> bool:
    return not isinstance(driver, _NOT_DIRECT)


def provider_name(driver: Any) -> str | None:
    return next((name for cls, name in _PROVIDERS if isinstance(driver, cls)), None)


def cost_micro_usd(provider: str | None, model: str, input_tokens: int, output_tokens: int) -> int | None:
    """Price a call in micro-USD, or None when the model has no price here."""
    rates = DIRECT_PROVIDER_PRICES.get(provider or "", {}).get(model)
    if rates is None:
        return None
    input_rate, output_rate = rates
    return int((input_tokens * input_rate + output_tokens * output_rate).to_integral_value(ROUND_HALF_UP))


class ModelCallMeter:
    """Counts the tokens a direct call spends and reports their cost. A no-op for non-direct calls."""

    def __init__(self, node: BaseNode, driver: Any, correlation_id: str | None, *, direct: bool) -> None:
        self._node = node
        self._driver = driver
        self._correlation_id = correlation_id
        self._direct = direct

    def run(self, fn: Callable[[], T]) -> T:
        """Run the model call, then report its cost -- even when it fails partway, since tokens were spent."""
        if not self._direct:
            return fn()
        model = self._driver.model
        tokens = [0, 0]
        seen = False

        def _count(event: FinishPromptEvent) -> None:
            nonlocal seen
            # Only this driver's calls: a tool may run its own sub-agent on another model.
            if event.model == model:
                seen = True
                tokens[0] += int(event.input_token_count or 0)
                tokens[1] += int(event.output_token_count or 0)

        try:
            # Listeners are context-local; `run_stream` copies the context into its worker thread.
            with EventListener(_count, event_types=[FinishPromptEvent]):
                return fn()
        finally:
            self._report(model, *tokens, seen=seen)

    def _report(self, model: str, input_tokens: int, output_tokens: int, *, seen: bool) -> None:
        node_type = type(self._node).__name__
        if not seen:
            # e.g. an image driver, which publishes no token counts.
            logger.warning("%s: no token usage seen for direct call to '%s'; usage not reported.", node_type, model)
            return
        provider = provider_name(self._driver)
        cost = cost_micro_usd(provider, model, input_tokens, output_tokens)
        if cost is None:
            logger.warning("%s: no price for '%s' (%s); usage not reported.", node_type, model, provider)
            return
        report_model_usage_sync(
            self._node,
            declared_cost_micro_usd=cost,
            provider=provider,
            model=model,
            correlation_id=self._correlation_id,
        )


def require_driver_access(node: BaseNode, driver: Any, *, purpose: str | None = None) -> ModelCallMeter:
    """Gate a call on `driver`: permission always, plus the budget check when it is a direct call.

    Run the call through the returned meter's `run` so a direct call's cost gets reported.
    """
    if not is_direct_provider(driver):
        require_model_invocation_sync(node, driver.model, purpose=purpose)
        return ModelCallMeter(node, driver, None, direct=False)
    correlation_id = require_model_access_sync(node, driver.model, purpose=purpose)
    return ModelCallMeter(node, driver, correlation_id, direct=True)
