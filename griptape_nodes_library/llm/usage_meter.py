"""Budget-gate and meter model calls that go straight to a provider rather than through Griptape Cloud.

Whether a call is direct depends on the model config actually in use, not on the node: an Agent
on the Griptape Cloud provider goes through the proxy, which enforces budgets itself, while the
same Agent with a connected Anthropic config calls Anthropic directly. `require_model_access`
picks the right gate for the config, and the returned meter reports the call's cost afterwards.

The meter counts only the node's own top-level run. `run_agent` adds each run's usage to the
active meter, including the requests a run made before it raised. An agent called as a tool runs
through `run_agent_async` inside that run and is not added to it. It is gated and metered on its
own model instead, by `run_tool_model_async`, and charged to the node whose run called it: a tool on
Griptape Cloud gets the permission check only, and a tool on a direct provider is budget-checked
and reported under its own provider and model.

Not vendored: the price table and provider names are this library's business. The vendorable
contract lives in `utils/model_invocation.py`.
"""

from __future__ import annotations

import asyncio
import logging
from contextvars import ContextVar
from decimal import ROUND_HALF_UP, Decimal
from typing import TYPE_CHECKING

from pydantic_ai.usage import RunUsage

from griptape_nodes_library.llm.image_generation import ImageGenerationConfig, ImageProvider
from griptape_nodes_library.llm.model_config import ModelConfig, ModelProvider
from griptape_nodes_library.utils.model_invocation import (
    report_model_usage_sync,
    require_model_access_sync,
    require_model_invocation_sync,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from griptape_nodes.exe_types.node_types import BaseNode

logger = logging.getLogger("griptape_nodes")

# Griptape Cloud is budgeted server-side; local servers cost nothing.
_NOT_DIRECT = {
    ModelProvider.GRIPTAPE_CLOUD,
    ModelProvider.OLLAMA,
    ModelProvider.LMSTUDIO,
    ImageProvider.GRIPTAPE_CLOUD,
}

# Provider list prices in USD per 1,000,000 (input, output) tokens -- which is also micro-USD per
# token. No markup: this is the customer's own spend. Taken from griptape-cloud's
# `credits/chat_token_pricing.py` (fetched there 2026-07-31 / 2026-08-26). Direct Anthropic is that
# file's Bedrock regional rate without its 1.1x regional premium. Keys are `ModelProvider` values,
# which are also the provider names Cloud groups usage on. Providers missing here are
# budget-checked but not reported.
DIRECT_PROVIDER_PRICES: dict[str, dict[str, tuple[Decimal, Decimal]]] = {
    ModelProvider.OPENAI: {
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
    ModelProvider.ANTHROPIC: {
        "claude-opus-4-7": (Decimal("5.00"), Decimal("25.00")),
        "claude-sonnet-4-6": (Decimal("3.00"), Decimal("15.00")),
        "claude-haiku-4-5": (Decimal("1.00"), Decimal("5.00")),
    },
    ModelProvider.BEDROCK: {
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

_active_meter: ContextVar[ModelCallMeter | None] = ContextVar("active_model_call_meter", default=None)


def is_direct_provider(config: ModelConfig | ImageGenerationConfig) -> bool:
    return config.provider not in _NOT_DIRECT


def cost_micro_usd(provider: str, model: str, input_tokens: int, output_tokens: int) -> int | None:
    """Price a call in micro-USD, or None when the model has no price here."""
    rates = DIRECT_PROVIDER_PRICES.get(provider, {}).get(model)
    if rates is None:
        return None
    input_rate, output_rate = rates
    return int((input_tokens * input_rate + output_tokens * output_rate).to_integral_value(ROUND_HALF_UP))


def record_run_usage(usage: RunUsage) -> None:
    """Add a finished run's usage to the meter wrapping it, if any. Called by `run_agent`."""
    meter = _active_meter.get()
    if meter is not None:
        meter.add(usage)


async def run_tool_model_async[T](config: ModelConfig, run: Callable[[RunUsage], Awaitable[T]]) -> T:
    """Gate and meter a model an agent tool runs, charged to the node whose run called the tool.

    `run` receives the usage counter to pass to `run_agent_async`, so the tool's spend is counted
    even when its run raises. Outside a node's metered run (no meter active), `run` runs ungated.
    """
    parent = _active_meter.get()
    if parent is None:
        return await run(RunUsage())
    # The gate makes a blocking engine round trip; keep it off the loop.
    meter = await asyncio.to_thread(require_model_access, parent.node, config)
    usage = RunUsage()
    try:
        return await run(usage)
    finally:
        if meter.direct:
            meter.add(usage)
            await asyncio.to_thread(meter.report)


class ModelCallMeter:
    """Counts the tokens a direct call spends and reports their cost. A no-op for non-direct calls."""

    def __init__(
        self, node: BaseNode, config: ModelConfig | ImageGenerationConfig, correlation_id: str | None, *, direct: bool
    ) -> None:
        self.node = node
        self._config = config
        self._correlation_id = correlation_id
        self.direct = direct
        self._input_tokens = 0
        self._output_tokens = 0
        self._runs = 0

    def add(self, usage: RunUsage) -> None:
        self._runs += 1
        self._input_tokens += usage.input_tokens
        self._output_tokens += usage.output_tokens

    def run[T](self, fn: Callable[[], T]) -> T:
        """Run the model call, then report its cost."""
        token = _active_meter.set(self)
        try:
            return fn()
        finally:
            _active_meter.reset(token)
            if self.direct:
                self.report()

    def report(self) -> None:
        node_type = type(self.node).__name__
        model = self._config.model
        if not self._runs:
            # Image generation carries no token usage.
            logger.warning("%s: no token usage for direct call to '%s'; usage not reported.", node_type, model)
            return
        provider = str(self._config.provider)
        cost = cost_micro_usd(provider, model, self._input_tokens, self._output_tokens)
        if cost is None:
            logger.warning("%s: no price for '%s' (%s); usage not reported.", node_type, model, provider)
            return
        report_model_usage_sync(
            self.node,
            declared_cost_micro_usd=cost,
            provider=provider,
            model=model,
            correlation_id=self._correlation_id,
        )


def require_model_access(
    node: BaseNode, config: ModelConfig | ImageGenerationConfig, *, purpose: str | None = None
) -> ModelCallMeter:
    """Gate a call on `config`: permission always, plus the budget check when it is a direct call.

    Run the call through the returned meter's `run` so a direct call's cost gets reported.
    """
    if not is_direct_provider(config):
        require_model_invocation_sync(node, config.model, purpose=purpose)
        return ModelCallMeter(node, config, None, direct=False)
    correlation_id = require_model_access_sync(node, config.model, purpose=purpose)
    return ModelCallMeter(node, config, correlation_id, direct=True)
