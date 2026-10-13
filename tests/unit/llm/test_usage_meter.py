"""Direct-provider budget gating and usage metering.

Only a call that goes straight to a provider is budget-checked and reported; a call through
the Griptape proxy is budgeted server-side and must not be counted twice.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
from griptape_nodes.exe_types.node_types import BaseNode
from pydantic_ai import Agent
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel
from pydantic_ai.usage import RunUsage, UsageLimits

import griptape_nodes_library.llm.usage_meter as meter_module
from griptape_nodes_library.config.prompt import amazon_bedrock_prompt, anthropic_prompt, openai_prompt
from griptape_nodes_library.llm.image_generation import ImageGenerationConfig, ImageProvider
from griptape_nodes_library.llm.model_config import ModelConfig, ModelProvider
from griptape_nodes_library.llm.runner import run_agent, run_agent_async
from griptape_nodes_library.llm.usage_meter import (
    DIRECT_PROVIDER_PRICES,
    cost_micro_usd,
    is_direct_provider,
    require_model_access,
)

OPENAI_GPT_4O = ModelConfig(provider=ModelProvider.OPENAI, model="gpt-4o")


class _StubNode(BaseNode):
    def process(self) -> None:
        return


@pytest.fixture
def node() -> _StubNode:
    return _StubNode(name="StubNode")


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict[str, Any]]]:
    """Record which gate ran and every usage report, without touching the engine."""
    seen: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(
        meter_module,
        "require_model_invocation_sync",
        lambda _node, model, **kw: seen.append(("permission", {"model": model, **kw})),
    )

    def _access(_node: Any, model: str, **kw: Any) -> str:
        seen.append(("access", {"model": model, **kw}))
        return "corr-1"

    monkeypatch.setattr(meter_module, "require_model_access_sync", _access)
    monkeypatch.setattr(meter_module, "report_model_usage_sync", lambda _node, **kw: seen.append(("report", kw)))
    return seen


async def _say(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:  # noqa: ARG001
    yield "hello there"


def _agent() -> Agent[None, str]:
    return Agent(FunctionModel(stream_function=_say))


def _cost(usage: RunUsage) -> int | None:
    return cost_micro_usd("openai", "gpt-4o", usage.input_tokens, usage.output_tokens)


@pytest.mark.parametrize(
    ("config", "direct"),
    [
        (ModelConfig(provider=ModelProvider.GRIPTAPE_CLOUD, model="gpt-4o"), False),
        (ModelConfig(provider=ModelProvider.OLLAMA, model="llama3"), False),
        (ModelConfig(provider=ModelProvider.LMSTUDIO, model="qwen"), False),
        (ImageGenerationConfig(provider=ImageProvider.GRIPTAPE_CLOUD, model="gpt-image-1"), False),
        (OPENAI_GPT_4O, True),
        (ModelConfig(provider=ModelProvider.ANTHROPIC, model="claude-sonnet-4-6"), True),
        (ImageGenerationConfig(provider=ImageProvider.OPENAI, model="gpt-image-1"), True),
    ],
)
def test_is_direct_provider(config: Any, *, direct: bool) -> None:
    assert is_direct_provider(config) is direct


def test_cost_is_micro_usd_per_token() -> None:
    # gpt-4o: $2.50 / $10.00 per 1M tokens.
    assert cost_micro_usd("openai", "gpt-4o", 1000, 500) == 7500


def test_cost_is_none_for_an_unpriced_model() -> None:
    assert cost_micro_usd("openai", "my-finetune", 1000, 500) is None
    assert cost_micro_usd("groq", "llama-3.3-70b-versatile", 1000, 500) is None


def test_cloud_config_gets_permission_only(node: _StubNode, calls: list[tuple[str, dict[str, Any]]]) -> None:
    meter = require_model_access(node, ModelConfig(provider=ModelProvider.GRIPTAPE_CLOUD, model="gpt-4o"))
    meter.run(lambda: run_agent(_agent(), "hi"))

    assert [name for name, _ in calls] == ["permission"]


def test_direct_run_is_checked_and_reported_from_its_usage(
    node: _StubNode, calls: list[tuple[str, dict[str, Any]]]
) -> None:
    meter = require_model_access(node, OPENAI_GPT_4O, purpose="prompt enhancement")

    result = meter.run(lambda: run_agent(_agent(), "hi"))

    assert calls == [
        ("access", {"model": "gpt-4o", "purpose": "prompt enhancement"}),
        (
            "report",
            {
                "declared_cost_micro_usd": _cost(result.usage),
                "provider": "openai",
                "model": "gpt-4o",
                "correlation_id": "corr-1",
            },
        ),
    ]


def test_a_nested_tool_run_is_not_counted(node: _StubNode, calls: list[tuple[str, dict[str, Any]]]) -> None:
    """A tool's own model run (e.g. a sub-agent on Griptape Cloud) is not the node's direct spend."""
    sub_agent = _agent()
    sub_runs: list[RunUsage] = []

    async def ask_sub_agent() -> str:
        sub_result = await run_agent_async(sub_agent, "sub")
        sub_runs.append(sub_result.usage)
        return sub_result.output

    async def call_tool_then_answer(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[Any]:  # noqa: ARG001
        if len(messages) == 1:
            yield {0: DeltaToolCall(name="ask_sub_agent", json_args="{}")}
        else:
            yield "done"

    agent = Agent(FunctionModel(stream_function=call_tool_then_answer), tools=[ask_sub_agent])
    meter = require_model_access(node, OPENAI_GPT_4O)

    result = meter.run(lambda: run_agent(agent, "hi"))

    (sub_usage,) = sub_runs
    assert sub_usage.output_tokens > 0
    name, report = calls[-1]
    assert name == "report"
    # The outer run's usage alone; counting the sub-agent would add its tokens on top.
    assert report["declared_cost_micro_usd"] == _cost(result.usage)
    assert report["declared_cost_micro_usd"] != _cost(result.usage + sub_usage)


def test_unpriced_model_is_checked_but_not_reported(
    node: _StubNode, calls: list[tuple[str, dict[str, Any]]], caplog: pytest.LogCaptureFixture
) -> None:
    meter = require_model_access(node, ModelConfig(provider=ModelProvider.OPENAI, model="my-finetune"))

    with caplog.at_level("WARNING", logger="griptape_nodes"):
        meter.run(lambda: run_agent(_agent(), "hi"))

    assert [name for name, _ in calls] == ["access"]
    assert "no price for 'my-finetune'" in caplog.text


def test_call_without_usage_is_not_reported(
    node: _StubNode, calls: list[tuple[str, dict[str, Any]]], caplog: pytest.LogCaptureFixture
) -> None:
    """Image generation carries no token usage, so it is checked but not reported."""
    meter = require_model_access(node, ImageGenerationConfig(provider=ImageProvider.OPENAI, model="gpt-image-1"))

    with caplog.at_level("WARNING", logger="griptape_nodes"):
        meter.run(lambda: b"png bytes")

    assert [name for name, _ in calls] == ["access"]
    assert "no token usage" in caplog.text


def test_runs_outside_a_meter_are_not_counted(node: _StubNode, calls: list[tuple[str, dict[str, Any]]]) -> None:
    run_agent(_agent(), "before")
    meter = require_model_access(node, OPENAI_GPT_4O)
    result = meter.run(lambda: run_agent(_agent(), "inside"))
    run_agent(_agent(), "after")

    assert calls[-1][1]["declared_cost_micro_usd"] == _cost(result.usage)


def test_a_run_that_hits_its_request_limit_still_reports(
    node: _StubNode, calls: list[tuple[str, dict[str, Any]]]
) -> None:
    """MCP Task ends on the request limit by design; the requests it already made are still spent."""

    async def keep_calling_the_tool(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[Any]:  # noqa: ARG001
        yield {0: DeltaToolCall(name="again", json_args="{}")}

    def again() -> str:
        return "again"

    agent = Agent(FunctionModel(stream_function=keep_calling_the_tool), tools=[again])
    meter = require_model_access(node, OPENAI_GPT_4O)

    with pytest.raises(UsageLimitExceeded):
        meter.run(lambda: run_agent(agent, "hi", usage_limits=UsageLimits(request_limit=3)))

    name, report = calls[-1]
    assert name == "report"
    assert report["declared_cost_micro_usd"] > 0


# Bedrock's Amazon Titan models have no sourced price yet, so they are checked but not reported.
_KNOWN_UNPRICED = {"amazon.titan-text-premier-v1:0", "amazon.titan-text-express-v1", "amazon.titan-text-lite-v1"}


@pytest.mark.parametrize(
    ("choices", "provider"),
    [
        (openai_prompt.MODEL_CHOICES, ModelProvider.OPENAI),
        (anthropic_prompt.MODEL_CHOICES, ModelProvider.ANTHROPIC),
        (amazon_bedrock_prompt.MODEL_CHOICES, ModelProvider.BEDROCK),
    ],
)
def test_every_direct_dropdown_model_is_priced(choices: list[str], provider: ModelProvider) -> None:
    """A model added to a config dropdown without a price would be checked but silently never reported."""
    assert set(choices) - set(DIRECT_PROVIDER_PRICES[provider]) - _KNOWN_UNPRICED == set()
