"""Direct-provider budget gating and token reporting.

Only a call that goes straight to a provider is budget-checked and reported; a call through the
Griptape proxy is budgeted server-side and gets the permission check only.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import MagicMock

import pytest
from griptape_nodes.exe_types.node_types import BaseNode
from griptape_nodes.utils.budget_refusal import BudgetExceededError
from pydantic_ai import Agent
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel
from pydantic_ai.usage import UsageLimits

import griptape_nodes_library.llm.budget_gate as gate_module
import griptape_nodes_library.llm.tools as tools_module
from griptape_nodes_library.llm import models
from griptape_nodes_library.llm.agent_state import AgentState
from griptape_nodes_library.llm.budget_gate import check_model_permission, is_direct_provider, run_gated
from griptape_nodes_library.llm.image_generation import ImageGenerationConfig, ImageProvider
from griptape_nodes_library.llm.model_config import ModelConfig, ModelProvider
from griptape_nodes_library.llm.models import override_model
from griptape_nodes_library.llm.runner import run_agent
from griptape_nodes_library.llm.testing import text_model
from griptape_nodes_library.llm.tools import ToolType, build_toolset

CLOUD = ModelConfig(provider=ModelProvider.GRIPTAPE_CLOUD, model="gpt-4o")
OPENAI = ModelConfig(provider=ModelProvider.OPENAI, model="gpt-4o")
ANTHROPIC = ModelConfig(provider=ModelProvider.ANTHROPIC, model="claude-sonnet-4-6")
OPENAI_IMAGE = ImageGenerationConfig(provider=ImageProvider.OPENAI, model="gpt-image-1")


class _StubNode(BaseNode):
    def process(self) -> None:
        return


@pytest.fixture
def node() -> _StubNode:
    return _StubNode(name="StubNode")


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, Any, Any]]:
    """Record each gate and report, with the node it was charged to, without touching the engine."""
    seen: list[tuple[str, Any, Any]] = []
    monkeypatch.setattr(
        gate_module,
        "require_model_invocation_sync",
        lambda gated_node, model, **_kw: seen.append(("permission", gated_node, model)),
    )

    def _check(gated_node: Any, model: str, **_kw: Any) -> str:
        seen.append(("budget", gated_node, model))
        return f"corr-{model}"

    monkeypatch.setattr(gate_module, "require_budget_access_sync", _check)
    monkeypatch.setattr(gate_module, "require_model_access_sync", _check)
    monkeypatch.setattr(
        gate_module, "report_model_usage_sync", lambda gated_node, **kw: seen.append(("report", gated_node, kw))
    )
    return seen


def _names(calls: list[tuple[str, Any, Any]]) -> list[str]:
    return [name for name, _, _ in calls]


def _reports(calls: list[tuple[str, Any, Any]]) -> list[dict[str, Any]]:
    return [report for name, _, report in calls if name == "report"]


async def _say(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:  # noqa: ARG001
    yield "hello there"


def _agent() -> Agent[None, str]:
    return Agent(FunctionModel(stream_function=_say))


@pytest.mark.parametrize(
    ("config", "direct"),
    [
        (CLOUD, False),
        (ModelConfig(provider=ModelProvider.OLLAMA, model="llama3"), False),
        (ModelConfig(provider=ModelProvider.LMSTUDIO, model="qwen"), False),
        (ImageGenerationConfig(provider=ImageProvider.GRIPTAPE_CLOUD, model="gpt-image-1"), False),
        (OPENAI, True),
        (ANTHROPIC, True),
        (ModelConfig(provider=ModelProvider.GROQ, model="llama-3.3-70b-versatile"), True),
        (ModelConfig(provider=ModelProvider.OPENAI_COMPATIBLE, model="my-model"), True),
        (OPENAI_IMAGE, True),
    ],
)
def test_is_direct_provider(config: Any, *, direct: bool) -> None:
    assert is_direct_provider(config) is direct


def test_a_cloud_call_is_neither_budget_checked_nor_reported(
    node: _StubNode, calls: list[tuple[str, Any, Any]]
) -> None:
    """Cloud enforces its own budgets. The permission check is the node's, in `process()`."""
    run_gated(node, CLOUD, lambda: run_agent(_agent(), "hi"))

    assert calls == []


def test_the_permission_check_is_separate_from_the_budget_check(
    node: _StubNode, calls: list[tuple[str, Any, Any]]
) -> None:
    """Permission is local and fast, so a node runs it in `process()`; the budget check runs in `run_gated`."""
    check_model_permission(node, OPENAI)

    assert calls == [("permission", node, "gpt-4o")]


def test_a_direct_call_is_checked_then_reports_its_tokens(node: _StubNode, calls: list[tuple[str, Any, Any]]) -> None:
    with override_model(text_model("hi")):
        result = run_gated(node, OPENAI, lambda: run_agent(_agent(), "hi"))

    assert _names(calls) == ["budget", "report"]
    _, reported_node, report = calls[1]
    assert reported_node is node
    usage = result.usage
    assert usage.input_tokens > 0
    assert report == {
        "provider": "openai",
        "model": "gpt-4o",
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_read_tokens": usage.cache_read_tokens,
        "cache_write_tokens": usage.cache_write_tokens,
        "input_audio_tokens": usage.input_audio_tokens,
        "cache_audio_read_tokens": usage.cache_audio_read_tokens,
        "correlation_id": "corr-gpt-4o",
    }


def test_a_run_that_raises_still_reports_the_tokens_it_used(node: _StubNode, calls: list[tuple[str, Any, Any]]) -> None:
    """MCP Task ends on its request limit by design; the requests already made were spent."""

    async def keep_calling_the_tool(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[Any]:  # noqa: ARG001
        yield {0: DeltaToolCall(name="again", json_args="{}")}

    def again() -> str:
        return "again"

    agent = Agent(FunctionModel(stream_function=keep_calling_the_tool), tools=[again])

    with pytest.raises(UsageLimitExceeded):
        run_gated(
            node,
            OPENAI,
            lambda: run_agent(agent, "hi", usage_limits=UsageLimits(request_limit=3)),
        )

    (report,) = _reports(calls)
    assert report["input_tokens"] > 0


def test_a_run_that_fails_before_its_first_request_is_not_reported(
    node: _StubNode, calls: list[tuple[str, Any, Any]]
) -> None:
    """A bad key or an early cancel spends nothing, so it must not leave an empty spend record."""

    async def refuse(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:  # noqa: ARG001
        msg = "401 invalid api key"
        raise RuntimeError(msg)
        yield ""  # pragma: no cover - makes this an async generator

    with pytest.raises(RuntimeError, match="401"):
        run_gated(node, OPENAI, lambda: run_agent(Agent(FunctionModel(stream_function=refuse)), "hi"))

    assert _names(calls) == ["budget"]


def test_a_call_with_no_token_usage_is_not_reported(node: _StubNode, calls: list[tuple[str, Any, Any]]) -> None:
    """Direct image generation is budget-checked, but carries no tokens to report."""
    run_gated(node, OPENAI_IMAGE, lambda: b"png")

    assert _names(calls) == ["budget"]


def test_runs_outside_a_gated_call_are_not_reported(node: _StubNode, calls: list[tuple[str, Any, Any]]) -> None:
    with override_model(text_model("hi")):
        run_agent(_agent(), "before")
        inside = run_gated(node, OPENAI, lambda: run_agent(_agent(), "inside"))
        run_agent(_agent(), "after")

    (report,) = _reports(calls)
    assert report["input_tokens"] == inside.usage.input_tokens


def _agent_with_tool_on(tool_config: ModelConfig, monkeypatch: pytest.MonkeyPatch) -> Agent[None, str]:
    """An agent whose one tool is a real agent tool running on `tool_config`."""
    monkeypatch.setattr(models, "_require_key", lambda config: "key")
    toolset = build_toolset(
        {"tool_type": ToolType.AGENT_TOOL, "name": "helper", "agent_dict": AgentState(model=tool_config).to_wire()}
    )

    async def call_tool_then_answer(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[Any]:  # noqa: ARG001
        if len(messages) == 1:
            yield {0: DeltaToolCall(name="Helper", json_args='{"input": "hi"}')}
        else:
            yield "done"

    return Agent(FunctionModel(stream_function=call_tool_then_answer), toolsets=[toolset])


def test_a_direct_agent_tool_is_checked_and_reported_on_its_own_model(
    node: _StubNode, calls: list[tuple[str, Any, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Under a Griptape Cloud agent, a tool on Anthropic is checked and reported, charged to the calling node."""
    agent = _agent_with_tool_on(ANTHROPIC, monkeypatch)

    with override_model(text_model("from the tool")):
        run_gated(node, CLOUD, lambda: run_agent(agent, "hi"))

    assert _names(calls) == ["permission", "budget", "report"]
    _, reported_node, report = calls[2]
    assert reported_node is node
    assert (report["provider"], report["model"], report["correlation_id"]) == (
        "anthropic",
        "claude-sonnet-4-6",
        "corr-claude-sonnet-4-6",
    )
    assert report["input_tokens"] > 0


def test_a_cloud_agent_tool_under_a_direct_agent_is_not_billed_twice(
    node: _StubNode, calls: list[tuple[str, Any, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Cloud tool gets the permission check only, and its tokens are not added to the agent's report."""
    agent = _agent_with_tool_on(CLOUD, monkeypatch)

    with override_model(text_model("from the tool")):
        result = run_gated(node, OPENAI, lambda: run_agent(agent, "hi"))

    assert _names(calls) == ["budget", "permission", "report"]
    (report,) = _reports(calls)
    assert (report["model"], report["input_tokens"]) == ("gpt-4o", result.usage.input_tokens)


def test_an_agent_tool_outside_a_gated_call_is_not_gated(
    calls: list[tuple[str, Any, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = _agent_with_tool_on(ANTHROPIC, monkeypatch)

    with override_model(text_model("from the tool")):
        run_agent(agent, "hi")

    assert calls == []


def _agent_with_transcription_tool(tmp_path: Any, monkeypatch: pytest.MonkeyPatch, openai_calls: list[str]) -> Any:
    """An agent whose one tool is the real audio transcription tool, with OpenAI stubbed out."""
    audio = tmp_path / "clip.mp3"
    audio.write_bytes(b"fake audio")

    class _Transcriptions:
        def create(self, *, model: str, file: Any) -> Any:  # noqa: ARG002
            openai_calls.append(model)
            return type("Transcription", (), {"text": "hello"})()

    class _FakeOpenAI:
        def __init__(self, **_kwargs: Any) -> None:
            self.audio = type("Audio", (), {"transcriptions": _Transcriptions()})()

    monkeypatch.setattr(tools_module, "OpenAI", _FakeOpenAI)
    toolset = build_toolset({"tool_type": ToolType.AUDIO_TRANSCRIPTION, "model": "whisper-1"})

    async def call_tool_then_answer(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[Any]:  # noqa: ARG001
        if len(messages) == 1:
            yield {0: DeltaToolCall(name="transcribe_audio_from_disk", json_args=f'{{"path": "{audio}"}}')}
        else:
            yield "done"

    return Agent(FunctionModel(stream_function=call_tool_then_answer), toolsets=[toolset])


def test_the_transcription_tool_is_budget_checked_before_calling_openai(
    node: _StubNode, calls: list[tuple[str, Any, Any]], monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """It runs on the studio's own OpenAI key, so it is a direct call, charged to the calling node."""
    openai_calls: list[str] = []
    agent = _agent_with_transcription_tool(tmp_path, monkeypatch, openai_calls)

    run_gated(node, CLOUD, lambda: run_agent(agent, "transcribe it"))

    assert calls == [("budget", node, "whisper-1")]
    assert openai_calls == ["whisper-1"]


def test_a_refused_transcription_stops_the_run_without_calling_openai(
    node: _StubNode, calls: list[tuple[str, Any, Any]], monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """A refusal must stop the run, not go back to the model as the tool's text."""
    refusal = BudgetExceededError("Budget stopped this run. Studio budget is spent.", MagicMock())

    def _refuse(_node: Any, _model: str, **_kw: Any) -> None:
        raise refusal

    monkeypatch.setattr(gate_module, "require_model_access_sync", _refuse)
    openai_calls: list[str] = []
    agent = _agent_with_transcription_tool(tmp_path, monkeypatch, openai_calls)

    with pytest.raises(BudgetExceededError):
        run_gated(node, CLOUD, lambda: run_agent(agent, "transcribe it"))

    assert openai_calls == []


def test_the_transcription_tool_outside_a_gated_call_is_not_gated(
    calls: list[tuple[str, Any, Any]], monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    openai_calls: list[str] = []
    agent = _agent_with_transcription_tool(tmp_path, monkeypatch, openai_calls)

    run_agent(agent, "transcribe it")

    assert calls == []
    assert openai_calls == ["whisper-1"]
