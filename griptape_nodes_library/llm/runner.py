"""Build and run pydantic-ai agents from node code.

Node `process()` bodies run in a worker thread with no event loop, so
:func:`run_agent` owns one per call and streams events back through plain callbacks.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
from collections.abc import Callable, Coroutine, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic_ai import Agent, CancellationToken, StructuredDict
from pydantic_ai.exceptions import RunCancelled
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    UserContent,
)
from pydantic_ai.run import AgentRunResultEvent

from griptape_nodes_library.llm.budget import raise_budget_halt
from griptape_nodes_library.llm.history import history_token_budget, prune_history
from griptape_nodes_library.llm.model_config import USE_NATIVE_TOOLS_OPTION
from griptape_nodes_library.llm.models import build_model
from griptape_nodes_library.llm.rulesets import render_rulesets

if TYPE_CHECKING:
    from pydantic_ai.agent import AgentRunResult
    from pydantic_ai.messages import ModelMessage
    from pydantic_ai.toolsets import AbstractToolset
    from pydantic_ai.usage import UsageLimits

    from griptape_nodes_library.llm.model_config import ModelConfig

Prompt = str | Sequence[UserContent]

CANCEL_POLL_SECONDS = 0.25


class AgentRunCancelledError(Exception):
    """Raised when cancellation is requested during a run."""


@dataclass
class RunCallbacks:
    on_text: Callable[[str], None] | None = None
    on_tool_call: Callable[[str, str], None] | None = None
    on_tool_result: Callable[[str, str], None] | None = None
    is_cancelled: Callable[[], bool] | None = None


class HistoryBudgetAgent(Agent[None, Any]):
    """An agent that knows how much replayed history its model can take."""

    history_token_budget: int | None = None


def output_type_from_schema(schema: dict[str, Any], *, name: str | None = None) -> Any:
    return StructuredDict(schema, name=name or schema.get("title") or "output")


def build_agent(
    model_config: ModelConfig,
    *,
    instructions: str | None = None,
    rulesets: Sequence[dict] = (),
    toolsets: Sequence[AbstractToolset[Any]] = (),
    output_type: Any = str,
) -> Agent[None, Any]:
    """Build an agent for `model_config`.

    Raises:
        ValueError: Tools are attached but the config opted out of native tool calling.
    """
    if toolsets and model_config.options.get(USE_NATIVE_TOOLS_OPTION) is False:
        msg = (
            f"Model '{model_config.model}' has use_native_tools set to False, but prompted tool use is not "
            "supported. Remove the tools, or use a model with native tool calling and set use_native_tools to True."
        )
        raise ValueError(msg)
    parts = [p for p in (instructions, render_rulesets(list(rulesets))) if p]
    agent = HistoryBudgetAgent(
        build_model(model_config),
        instructions="\n\n".join(parts) or None,
        toolsets=list(toolsets) or None,
        output_type=output_type,
        deps_type=type(None),
    )
    agent.history_token_budget = history_token_budget(model_config)
    return agent


def run_coroutine_sync[T](coro: Coroutine[Any, Any, T]) -> T:
    """Run `coro` to completion from sync code, even if this thread already has a running loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):
        return str(value)


async def run_agent_async(
    agent: Agent[None, Any],
    prompt: Prompt | None,
    *,
    message_history: list[ModelMessage] | None = None,
    callbacks: RunCallbacks | None = None,
    usage_limits: UsageLimits | None = None,
) -> AgentRunResult[Any]:
    callbacks = callbacks or RunCallbacks()
    result: AgentRunResult[Any] | None = None
    # Prune here, not with a history processor: pydantic-ai writes processed history back
    # into the run, which would drop the pruned runs from saved memory too.
    budget = getattr(agent, "history_token_budget", None)
    if message_history and budget is not None:
        message_history = prune_history(message_history, budget)
    token = CancellationToken()
    watcher = asyncio.create_task(_watch_cancel(callbacks.is_cancelled, token)) if callbacks.is_cancelled else None
    try:
        async with agent.run_stream_events(
            prompt, message_history=message_history or None, usage_limits=usage_limits, cancellation_token=token
        ) as events:
            async for event in events:
                _dispatch(event, callbacks)
                if isinstance(event, AgentRunResultEvent):
                    result = event.result
    except RunCancelled as exc:
        raise AgentRunCancelledError from exc
    except Exception as exc:
        raise_budget_halt(exc)
        raise
    finally:
        if watcher is not None:
            watcher.cancel()
    if result is None:
        msg = "Agent run ended without a result."
        raise RuntimeError(msg)
    return result


async def _watch_cancel(is_cancelled: Callable[[], bool], token: CancellationToken) -> None:
    """Cancel `token` once `is_cancelled` reports True, so Cancel stops in-flight model and tool calls."""
    while not is_cancelled():
        await asyncio.sleep(CANCEL_POLL_SECONDS)
    token.cancel()


def _dispatch(event: Any, callbacks: RunCallbacks) -> None:
    match event:
        case PartStartEvent(part=TextPart(content=content)) if content and callbacks.on_text:
            callbacks.on_text(content)
        case PartDeltaEvent(delta=TextPartDelta(content_delta=delta)) if delta and callbacks.on_text:
            callbacks.on_text(delta)
        case FunctionToolCallEvent(part=part) if callbacks.on_tool_call:
            callbacks.on_tool_call(part.tool_name, part.args_as_json_str())
        case FunctionToolResultEvent(part=tool_result) if callbacks.on_tool_result:
            callbacks.on_tool_result(tool_result.tool_name or "", _stringify(tool_result.content))
        case _:
            pass


def run_agent(
    agent: Agent[None, Any],
    prompt: Prompt | None,
    *,
    message_history: list[ModelMessage] | None = None,
    callbacks: RunCallbacks | None = None,
    usage_limits: UsageLimits | None = None,
) -> AgentRunResult[Any]:
    """Run `agent` synchronously, streaming text and tool events to `callbacks`.

    Raises:
        AgentRunCancelledError: `callbacks.is_cancelled` returned True.
    """
    return run_coroutine_sync(
        run_agent_async(agent, prompt, message_history=message_history, callbacks=callbacks, usage_limits=usage_limits)
    )


def output_to_text(output: Any) -> str:
    if isinstance(output, str):
        return output
    if hasattr(output, "model_dump_json"):
        return output.model_dump_json()
    return json.dumps(output, default=str)


def prompt_model(
    model_config: ModelConfig,
    prompt: Prompt,
    *,
    instructions: str | None = None,
    rulesets: Sequence[dict] = (),
    toolsets: Sequence[AbstractToolset[Any]] = (),
    output_type: Any = str,
    message_history: list[ModelMessage] | None = None,
    callbacks: RunCallbacks | None = None,
) -> Any:
    agent = build_agent(
        model_config, instructions=instructions, rulesets=rulesets, toolsets=toolsets, output_type=output_type
    )
    return run_agent(agent, prompt, message_history=message_history, callbacks=callbacks).output
