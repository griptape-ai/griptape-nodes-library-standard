"""Gate model calls that go straight to a provider on Griptape Cloud budgets.

Whether a call is direct depends on the model config actually in use, not on the node: an Agent
on the Griptape Cloud provider goes through the proxy, which enforces budgets itself, while the
same Agent with a connected Anthropic config calls Anthropic directly. A node runs
`check_model_permission` in `process()`, then `run_gated` on the worker thread, which adds the
budget check for a direct call.

After a direct call, its token counts are reported and Griptape Cloud prices them; the library
holds no prices. `run_gated` counts a call's tokens: `run_agent` adds each run's usage to the
gated call it runs in, including runs that raise partway.

A tool that calls a model is gated on that model, charged to the node whose call ran it.
`run_tool_model` gates, runs and reports an agent called as a tool, on its own model, so its
tokens are never added to the calling node's. `gate_tool_direct_call` gates a tool that calls a
provider API itself and returns no tokens (audio transcription), so it is checked but not reported.
"""

from __future__ import annotations

import asyncio
from contextvars import ContextVar
from typing import TYPE_CHECKING

from pydantic_ai.usage import RunUsage

from griptape_nodes_library.llm.image_generation import ImageGenerationConfig, ImageProvider
from griptape_nodes_library.llm.model_config import ModelConfig, ModelProvider
from griptape_nodes_library.utils.model_invocation import (
    report_model_usage_sync,
    require_budget_access_sync,
    require_model_access_sync,
    require_model_invocation_sync,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from griptape_nodes.exe_types.node_types import BaseNode

# Griptape Cloud is budgeted server-side; local servers cost nothing.
_NOT_DIRECT = {
    ModelProvider.GRIPTAPE_CLOUD,
    ModelProvider.OLLAMA,
    ModelProvider.LMSTUDIO,
    ImageProvider.GRIPTAPE_CLOUD,
}

_active: ContextVar[GatedCall | None] = ContextVar("budget_gate_active_call", default=None)


def is_direct_provider(config: ModelConfig | ImageGenerationConfig) -> bool:
    return config.provider not in _NOT_DIRECT


class GatedCall:
    """One gated model call: the node it is charged to, and for a direct call, the tokens it used."""

    def __init__(
        self, node: BaseNode, config: ModelConfig | ImageGenerationConfig, correlation_id: str | None, *, direct: bool
    ) -> None:
        self.node = node
        self._config = config
        self._correlation_id = correlation_id
        self._direct = direct
        self._usage = RunUsage()
        self._runs = 0

    def add(self, usage: RunUsage) -> None:
        self._runs += 1
        self._usage = self._usage + usage

    def report(self) -> None:
        """Report a direct call's token counts. A call with none (image generation) reports nothing."""
        if not self._direct or not self._runs:
            return
        report_model_usage_sync(
            self.node,
            provider=str(self._config.provider),
            model=self._config.model,
            input_tokens=self._usage.input_tokens,
            output_tokens=self._usage.output_tokens,
            cache_read_tokens=self._usage.cache_read_tokens,
            cache_write_tokens=self._usage.cache_write_tokens,
            correlation_id=self._correlation_id,
        )


def check_model_permission(
    node: BaseNode, config: ModelConfig | ImageGenerationConfig, *, purpose: str | None = None
) -> None:
    """The permission half of the gate: local and fast, so a node runs it in `process()`."""
    require_model_invocation_sync(node, config.model, purpose=purpose)


def gate_call(node: BaseNode, config: ModelConfig | ImageGenerationConfig, *, purpose: str | None = None) -> GatedCall:
    """Gate a call on `config`: permission always, plus the budget check when it is a direct call.

    The budget check is a round trip to Griptape Cloud; call this from a worker thread.
    """
    check_model_permission(node, config, purpose=purpose)
    return _open_call(node, config)


def _open_call(node: BaseNode, config: ModelConfig | ImageGenerationConfig) -> GatedCall:
    """Budget-check a direct call, and start counting its tokens."""
    if not is_direct_provider(config):
        return GatedCall(node, config, None, direct=False)
    correlation_id = require_budget_access_sync(node, config.model)
    return GatedCall(node, config, correlation_id, direct=True)


def run_gated[T](node: BaseNode, config: ModelConfig | ImageGenerationConfig, fn: Callable[[], T]) -> T:
    """Budget-check a call on `config`, run it, and report its tokens. Call from the worker thread.

    The budget check is a round trip to Griptape Cloud, so a node runs this inside the function it
    yields, never in `process()` before the yield, which is on the engine's event loop. The node runs
    `check_model_permission` in `process()` first. The budget check runs before `fn`, so a refused
    call never reaches the provider. Reports even when the call raises partway,
    since the tokens it already used were spent; the agent tools it calls are gated as `node`.
    """
    call = _open_call(node, config)
    token = _active.set(call)
    try:
        return fn()
    finally:
        _active.reset(token)
        call.report()


def record_run_usage(usage: RunUsage) -> None:
    """Add a finished run's usage to the gated call it ran in, if any. Called by `run_agent`."""
    call = _active.get()
    if call is not None:
        call.add(usage)


def gate_tool_direct_call(api_model_id: str) -> None:
    """Gate a tool's own call to a provider API (e.g. audio transcription on the studio's key).

    Always a direct call, so always budget-checked; it returns no token counts, so it is not
    reported. For a sync tool, which runs off the loop. Outside a gated call there is no node
    to charge, and the tool runs ungated.
    """
    call = _active.get()
    if call is not None:
        require_model_access_sync(call.node, api_model_id)


async def run_tool_model[T](config: ModelConfig, run: Callable[[RunUsage], Awaitable[T]]) -> T:
    """Gate, run and report a model an agent tool runs, charged to the node whose call ran the tool.

    `run` receives the usage counter to pass to `run_agent_async`, so the tool's tokens are counted
    even when its run raises. Outside a gated call there is no node to charge, and `run` runs ungated.
    """
    parent = _active.get()
    if parent is None:
        return await run(RunUsage())
    # The gate and the report make blocking engine round trips; keep them off the loop.
    call = await asyncio.to_thread(gate_call, parent.node, config)
    usage = RunUsage()
    try:
        return await run(usage)
    finally:
        call.add(usage)
        await asyncio.to_thread(call.report)
