import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any

import pytest
from cohere import (
    AssistantMessageResponse,
    TextAssistantMessageResponseContentItem,
    ToolCallV2,
    ToolCallV2Function,
    V2ChatResponse,
)
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCalls, FunctionModel

from griptape_nodes_library.llm import models
from griptape_nodes_library.llm.agent_state import AgentState, messages_from_runs, runs_from_messages
from griptape_nodes_library.llm.history import history_token_budget, prune_history
from griptape_nodes_library.llm.model_config import (
    USE_NATIVE_TOOLS_OPTION,
    ModelConfig,
    ModelProvider,
    model_config_from_legacy_driver,
)
from griptape_nodes_library.llm.models import override_model
from griptape_nodes_library.llm.runner import AgentRunCancelledError, RunCallbacks, build_agent, run_agent
from griptape_nodes_library.llm.task_support import run_task_agent
from griptape_nodes_library.llm.testing import fake_model
from griptape_nodes_library.llm.tools import build_toolsets

CLOUD = ModelConfig(provider=ModelProvider.GRIPTAPE_CLOUD, model="gpt-4.1")


class TestAnthropicSampling:
    def test_top_p_wins_over_temperature(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(models, "_require_key", lambda config: "key")
        config = ModelConfig(
            provider=ModelProvider.ANTHROPIC, model="claude-sonnet-4-6", settings={"temperature": 0.1, "top_p": 0.9}
        )
        assert models.build_model(config).settings == {"top_p": 0.9}

    def test_temperature_alone_is_kept(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(models, "_require_key", lambda config: "key")
        config = ModelConfig(provider=ModelProvider.ANTHROPIC, model="claude-sonnet-4-6", settings={"temperature": 0.1})
        assert models.build_model(config).settings == {"temperature": 0.1}


class TestRetries:
    def test_cohere_chat_gets_max_retries(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(models, "_require_key", lambda config: "key")
        seen: dict[str, Any] = {}

        async def chat(*args: Any, **kwargs: Any) -> None:
            seen.update(kwargs)

        monkeypatch.setattr("cohere.AsyncClientV2.chat", chat, raising=False)
        client = models._cohere_client(ModelConfig(provider=ModelProvider.COHERE, model="c", max_retries=4))
        asyncio.run(client.chat(model="c", messages=[]))
        assert seen["request_options"] == {"max_retries": 4}

    def test_bedrock_client_counts_total_attempts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(models, "_secret", lambda name: "us-east-1" if name == "AWS_DEFAULT_REGION" else "x")
        client = models._bedrock_client(ModelConfig(provider=ModelProvider.BEDROCK, model="m", max_retries=4))
        assert client.meta.config.retries == {"mode": "standard", "total_max_attempts": 5}


class TestSchemaMemory:
    def test_structured_result_wins_over_preamble(self) -> None:
        messages = [
            ModelRequest(parts=[UserPromptPart("hi")]),
            ModelResponse(parts=[TextPart("I'll format that."), ToolCallPart("final_result", {"name": "x"})]),
        ]
        assert runs_from_messages(messages) == [{"input": "hi", "output": '{"name":"x"}'}]


class TestFromWire:
    def test_bad_messages_keep_the_rest_of_the_state(self) -> None:
        wire = AgentState(model=CLOUD, tools=[{"tool_type": "Calculator"}]).to_wire()
        wire["messages"] = [{"kind": "request", "parts": [{"part_kind": "unknown"}]}]
        state = AgentState.from_wire(wire)
        assert state.messages == []
        assert state.model == CLOUD
        assert state.tools == [{"tool_type": "Calculator"}]


class TestHistoryPruning:
    def _runs(self, count: int, size: int) -> list[ModelMessage]:
        return messages_from_runs([{"input": f"q{i}", "output": "x" * size} for i in range(count)])

    def test_under_budget_keeps_everything(self) -> None:
        messages = self._runs(3, 10)
        assert prune_history(messages, budget=10_000) == messages

    def test_drops_oldest_runs_first(self) -> None:
        messages = self._runs(4, 400)
        pruned = prune_history(messages, budget=700)
        assert [run["input"] for run in runs_from_messages(pruned)] == ["q2", "q3"]

    def test_keeps_latest_run_over_budget(self) -> None:
        pruned = prune_history(self._runs(3, 4000), budget=1)
        assert [run["input"] for run in runs_from_messages(pruned)] == ["q2"]

    def test_openai_compatible_gets_hosted_budget(self) -> None:
        compatible = ModelConfig(provider=ModelProvider.OPENAI_COMPATIBLE, model="m")
        openai = ModelConfig(provider=ModelProvider.OPENAI, model="m")
        assert history_token_budget(compatible) == history_token_budget(openai)

    def test_agent_sends_pruned_history(self) -> None:
        seen: list[list[ModelMessage]] = []

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            seen.append(messages)
            return ModelResponse(parts=[TextPart("ok")])

        local = ModelConfig(provider=ModelProvider.OLLAMA, model="llama")
        history = self._runs(10, 6000)
        with override_model(fake_model(respond)):
            result = run_task_agent(local, "now", message_history=history)

        assert [run["input"] for run in runs_from_messages(seen[0])] == ["q9", "now"]
        assert len(runs_from_messages(result.messages)) == 11


class TestNativeTools:
    def test_opt_out_with_tools_fails_clearly(self) -> None:
        config = CLOUD.model_copy(update={"options": {USE_NATIVE_TOOLS_OPTION: False}})
        with (
            override_model(FunctionModel(lambda m, i: ModelResponse(parts=[]))),
            pytest.raises(ValueError, match="use_native_tools"),
        ):
            build_agent(config, toolsets=build_toolsets([{"tool_type": "Calculator"}]))

    def test_opt_out_without_tools_runs(self) -> None:
        config = CLOUD.model_copy(update={"options": {USE_NATIVE_TOOLS_OPTION: False}})
        with override_model(FunctionModel(lambda m, i: ModelResponse(parts=[]))):
            build_agent(config)

    def test_legacy_driver_opt_out_is_kept(self) -> None:
        config = model_config_from_legacy_driver(
            {"type": "OllamaPromptDriver", "model": "m", "use_native_tools": False}
        )
        assert config.options == {USE_NATIVE_TOOLS_OPTION: False}


def test_cohere_runs_without_streaming(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(models, "_require_key", lambda config: "key")
    replies = iter(
        [
            V2ChatResponse(
                id="1",
                finish_reason="TOOL_CALL",
                message=AssistantMessageResponse(
                    tool_calls=[
                        ToolCallV2(
                            id="c1",
                            type="function",
                            function=ToolCallV2Function(name="calculate", arguments='{"expression": "2+2"}'),
                        )
                    ]
                ),
            ),
            V2ChatResponse(
                id="2",
                finish_reason="COMPLETE",
                message=AssistantMessageResponse(content=[TextAssistantMessageResponseContentItem(text="4")]),
            ),
        ]
    )

    async def chat(*args: Any, **kwargs: Any) -> V2ChatResponse:
        return next(replies)

    monkeypatch.setattr("cohere.AsyncClientV2.chat", chat, raising=False)
    agent = build_agent(
        ModelConfig(provider=ModelProvider.COHERE, model="command-a"),
        toolsets=build_toolsets([{"tool_type": "Calculator"}]),
    )
    texts: list[str] = []
    calls: list[str] = []
    callbacks = RunCallbacks(on_text=texts.append, on_tool_call=lambda name, args: calls.append(name))

    result = run_agent(agent, "2+2?", callbacks=callbacks)

    assert result.output == "4"
    assert texts == ["4"]
    assert calls == ["calculate"]


def test_cancel_interrupts_a_model_call_with_no_events() -> None:
    async def stall(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        await asyncio.sleep(30)
        yield "late"

    model = FunctionModel(stream_function=stall)
    start = time.monotonic()
    with override_model(model), pytest.raises(AgentRunCancelledError):
        run_agent(build_agent(CLOUD), "x", callbacks=RunCallbacks(is_cancelled=lambda: time.monotonic() - start > 0.3))
    assert time.monotonic() - start < 5
