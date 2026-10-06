import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo
from pydantic_ai.models.test import TestModel

from griptape_nodes_library.llm.agent_state import AgentState, messages_from_runs
from griptape_nodes_library.llm.model_config import ModelConfig, ModelProvider, model_config_from_legacy_driver
from griptape_nodes_library.llm.models import override_model
from griptape_nodes_library.llm.rulesets import render_rulesets, rulesets_from_inputs
from griptape_nodes_library.llm.runner import (
    AgentRunCancelledError,
    RunCallbacks,
    build_agent,
    output_type_from_schema,
    prompt_model,
    run_agent,
)
from griptape_nodes_library.llm.testing import fake_model
from griptape_nodes_library.llm.tools import build_agent_from_state, build_toolsets, tool_configs_from_inputs

CLOUD = ModelConfig(provider=ModelProvider.GRIPTAPE_CLOUD, model="gpt-4.1")


class TestModelConfig:
    def test_wire_round_trip_drops_api_key(self) -> None:
        config = ModelConfig(provider=ModelProvider.OPENAI, model="gpt-5", api_key="sk-secret", settings={"seed": 1})

        wire = config.to_wire()

        assert "api_key" not in wire
        assert ModelConfig.from_wire(wire) == config.model_copy(update={"api_key": None})

    @pytest.mark.parametrize(
        ("driver", "expected_provider", "expected_base_url"),
        [
            (
                {"type": "GriptapeCloudPromptDriver", "model": "gpt-4.1", "base_url": "x"},
                ModelProvider.GRIPTAPE_CLOUD,
                None,
            ),
            ({"type": "OpenAiChatPromptDriver", "model": "m"}, ModelProvider.OPENAI, None),
            (
                {"type": "OpenAiChatPromptDriver", "model": "m", "base_url": "https://api.groq.com/openai/v1"},
                ModelProvider.GROQ,
                "https://api.groq.com/openai/v1",
            ),
            ({"type": "AnthropicPromptDriver", "model": "m"}, ModelProvider.ANTHROPIC, None),
            ({"type": "OllamaPromptDriver", "model": "m", "host": "http://h:1"}, ModelProvider.OLLAMA, "http://h:1/v1"),
        ],
    )
    def test_legacy_driver_mapping(self, driver, expected_provider, expected_base_url) -> None:
        config = model_config_from_legacy_driver(driver)

        assert config.provider == expected_provider
        assert config.base_url == expected_base_url

    def test_legacy_provider_blob_wins(self) -> None:
        config = model_config_from_legacy_driver(
            {"type": "OpenAiChatPromptDriver", "model": "m"},
            {"type": "custom", "base_url": "http://x/v1", "api_key": "k"},
        )

        assert config.provider == ModelProvider.OPENAI_COMPATIBLE
        assert config.api_key == "k"


class TestAgentState:
    def test_wire_round_trip(self) -> None:
        state = AgentState(
            model=CLOUD,
            messages=messages_from_runs([{"input": "hi", "output": "hello"}]),
            tools=[{"tool_type": "Calculator"}],
            rulesets=[{"name": "r", "rules": ["be nice"]}],
        )

        restored = AgentState.from_wire(state.to_wire())

        assert restored.model == CLOUD
        assert restored.runs() == [{"input": "hi", "output": "hello"}]
        assert restored.tools == state.tools
        assert restored.rulesets == state.rulesets

    def test_reads_legacy_wrapper(self) -> None:
        legacy = {
            "agent": {
                "type": "GriptapeNodesAgent",
                "tasks": [
                    {"type": "PromptTask", "prompt_driver": {"type": "GriptapeCloudPromptDriver", "model": "gpt-4.1"}}
                ],
                "conversation_memory": {
                    "type": "ConversationMemory",
                    "runs": [
                        {
                            "type": "Run",
                            "input": {"type": "TextArtifact", "value": "q"},
                            "output": {"type": "TextArtifact", "value": "a"},
                        }
                    ],
                },
            },
            "tools": [{"tool_type": "DateTime"}],
            "rulesets": [{"name": "r", "rules": ["x"]}],
        }

        state = AgentState.from_wire(legacy)

        assert state.model is not None
        assert state.model.provider == ModelProvider.GRIPTAPE_CLOUD
        assert state.runs() == [{"input": "q", "output": "a"}]
        assert state.tools == [{"tool_type": "DateTime"}]

    def test_reads_bare_legacy_agent_dict(self) -> None:
        state = AgentState.from_wire(
            {"conversation_memory": {"runs": [{"input": {"value": "q"}, "output": {"value": "a"}}]}}
        )

        assert state.model is None
        assert state.runs() == [{"input": "q", "output": "a"}]

    def test_bare_legacy_agent_keeps_inline_rulesets(self) -> None:
        state = AgentState.from_wire(
            {"rulesets": [{"type": "Ruleset", "name": "r", "rules": [{"type": "Rule", "value": "be nice"}]}]}
        )

        assert state.rulesets == [{"name": "r", "rules": ["be nice"]}]

    def test_garbage_yields_empty_state(self) -> None:
        assert AgentState.from_wire(None).messages == []
        assert AgentState.from_wire("not json").messages == []


class TestRulesets:
    def test_strings_promoted_and_lists_flattened(self) -> None:
        configs = rulesets_from_inputs(["  be brief ", "", [{"name": "a", "rules": ["x"]}]])

        assert configs == [{"name": "behavior_1", "rules": ["be brief"]}, {"name": "a", "rules": ["x"]}]

    def test_render_matches_griptape_layout(self) -> None:
        text = render_rulesets([{"name": "a", "rules": ["x", "y"]}])

        assert text.startswith("When responding, always use rules from the following rulesets.")
        assert 'Ruleset name: a\n"a" rules:\nRule #1\nx\nRule #2\ny' in text


class TestRunner:
    def test_streams_text_and_returns_history(self) -> None:
        chunks: list[str] = []
        with override_model(TestModel(custom_output_text="hello world")):
            agent = build_agent(CLOUD, rulesets=[{"name": "r", "rules": ["x"]}])
            result = run_agent(agent, "hi", callbacks=RunCallbacks(on_text=chunks.append))

        assert result.output == "hello world"
        assert "".join(chunks) == "hello world"
        assert AgentState(messages=result.all_messages()).runs() == [{"input": "hi", "output": "hello world"}]

    def test_message_history_reaches_model(self) -> None:
        seen: list[list[ModelMessage]] = []

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            seen.append(messages)
            return ModelResponse(parts=[TextPart("ok")])

        history = messages_from_runs([{"input": "earlier", "output": "reply"}])
        with override_model(fake_model(respond)):
            run_agent(build_agent(CLOUD), "now", message_history=history)

        assert len(seen[0]) == 3

    def test_structured_output(self) -> None:
        schema = {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"], "title": "Out"}
        with override_model(TestModel()):
            output = prompt_model(CLOUD, "x", output_type=output_type_from_schema(schema))

        assert isinstance(output, dict)
        assert "n" in output

    def test_cancellation(self) -> None:
        with override_model(TestModel()), pytest.raises(AgentRunCancelledError):
            run_agent(build_agent(CLOUD), "x", callbacks=RunCallbacks(is_cancelled=lambda: True))


class TestTools:
    def test_tool_call_events_and_calculator(self) -> None:
        calls: list[str] = []
        results: list[str] = []

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if len(messages) == 1:
                return ModelResponse(parts=[ToolCallPart("calculate", {"expression": "2 + 3"})])
            return ModelResponse(parts=[TextPart("done")])

        state = AgentState(model=CLOUD, tools=[{"tool_type": "Calculator"}, {"tool_type": "Calculator"}])
        with override_model(fake_model(respond)):
            run_agent(
                build_agent_from_state(state),
                "add",
                callbacks=RunCallbacks(
                    on_tool_call=lambda n, a: calls.append(n), on_tool_result=lambda n, r: results.append(r)
                ),
            )

        assert calls == ["calculate"]
        assert results == ["5"]

    def test_date_time_tools(self) -> None:
        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if len(messages) == 1:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            "get_datetime_diff",
                            {"start_datetime": "2021-01-01T00:00:00", "end_datetime": "2021-01-02T00:00:00"},
                        )
                    ]
                )
            return ModelResponse(parts=[TextPart("done")])

        results: list[str] = []
        with override_model(fake_model(respond)):
            run_agent(
                build_agent_from_state(AgentState(model=CLOUD, tools=[{"tool_type": "DateTime"}])),
                "diff",
                callbacks=RunCallbacks(on_tool_result=lambda n, r: results.append(r)),
            )

        assert results == ["1 day, 0:00:00"]

    def test_agent_tool_runs_sub_agent(self) -> None:
        sub = AgentState(model=CLOUD).to_wire()

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            tool_names = [t.name for t in info.function_tools]
            if "Researcher" in tool_names and len(messages) == 1:
                return ModelResponse(parts=[ToolCallPart("Researcher", {"input": "look"})])
            return ModelResponse(parts=[TextPart("sub-or-final")])

        results: list[str] = []
        state = AgentState(model=CLOUD, tools=[{"tool_type": "AgentTool", "agent_dict": sub, "name": "researcher"}])
        with override_model(fake_model(respond)):
            run_agent(
                build_agent_from_state(state),
                "go",
                callbacks=RunCallbacks(on_tool_result=lambda n, r: results.append(r)),
            )

        assert results == ["sub-or-final"]

    def test_unknown_tool_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown tool_type"):
            build_toolsets([{"tool_type": "Nope"}])

    def test_non_config_tool_input_raises(self) -> None:
        with pytest.raises(TypeError):
            tool_configs_from_inputs([object()])
