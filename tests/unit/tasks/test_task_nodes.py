from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING, Any, cast

import pytest
from griptape_nodes.node_library.library_registry import LibraryRegistry
from pydantic_ai import FunctionToolset
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from griptape_nodes_library.llm.agent_state import AgentState
from griptape_nodes_library.llm.models import override_model
from griptape_nodes_library.llm.testing import fake_model, text_model
from griptape_nodes_library.tasks.mcp_task import MCPTaskNode

if TYPE_CHECKING:
    from griptape_nodes.exe_types.node_types import BaseNode
    from pydantic_ai.models.function import AgentInfo

LIBRARY_NAME = "Griptape Nodes Library"


def _create_node(node_type: str) -> Any:
    library = LibraryRegistry.get_library(name=LIBRARY_NAME)
    node: BaseNode = library.create_node(node_type=node_type, name=node_type)
    return node


def _run(node: BaseNode) -> Any:
    """Drive a generator `process()` to completion, calling each yielded thunk."""
    generator = node.process()
    result = None
    if generator is None:
        return result
    try:
        thunk = next(generator)
        while True:
            thunk = generator.send(thunk())
    except StopIteration:
        return result


def _tool_then_text(tool_name: str, args: dict[str, Any], final_text: str) -> Any:
    """A model that calls `tool_name` once, then answers `final_text`. Records every request's messages."""
    requests: list[list[ModelMessage]] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:  # noqa: ARG001
        requests.append(messages)
        returned = any(isinstance(p, ToolReturnPart) for m in messages if isinstance(m, ModelRequest) for p in m.parts)
        if returned:
            return ModelResponse(parts=[TextPart(final_text)])
        return ModelResponse(parts=[ToolCallPart(tool_name, args, tool_call_id="call_1")])

    model = fake_model(respond)
    model.requests = requests  # type: ignore[attr-defined]
    return model


def _last_user_prompt(messages: list[ModelMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, UserPromptPart):
                    return str(part.content)
    return ""


def _instructions(messages: list[ModelMessage]) -> str:
    return next((m.instructions or "" for m in reversed(messages) if isinstance(m, ModelRequest)), "")


class TestSummarizeText:
    def test_streams_summary_with_griptape_summary_prompts(self) -> None:
        seen: list[list[ModelMessage]] = []

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:  # noqa: ARG001
            seen.append(messages)
            return ModelResponse(parts=[TextPart("a short summary")])

        node = _create_node("SummarizeText")
        node.set_parameter_value("prompt", "Long text here.")

        with override_model(fake_model(respond)):
            _run(node)

        assert node.parameter_output_values["output"] == "a short summary"
        assert _instructions(seen[0]) == "You are an expert in text summarization."
        assert _last_user_prompt(seen[0]) == 'Summarize the following text: """\nLong text here.\n"""\n\nSummary:'

    def test_blank_prompt_never_calls_the_model(self) -> None:
        node = _create_node("SummarizeText")
        node.set_parameter_value("prompt", "   ")
        calls: list[Any] = []

        with override_model(fake_model(lambda m, i: calls.append(m) or ModelResponse(parts=[TextPart("x")]))):
            _run(node)

        assert calls == []


class TestDateAndTime:
    def test_model_can_call_the_datetime_tool(self) -> None:
        model = _tool_then_text("get_current_datetime", {}, "Jun 15, 2024")
        node = _create_node("DateAndTime")
        node.set_parameter_value("prompt", "right now")

        with override_model(model):
            _run(node)

        assert node.parameter_output_values["output"] == "Jun 15, 2024"
        tool_returns = [
            p
            for m in model.requests[1]
            if isinstance(m, ModelRequest)
            for p in m.parts
            if isinstance(p, ToolReturnPart)
        ]
        assert tool_returns
        assert "Get date and time information for: right now" in _last_user_prompt(model.requests[0])


class TestScrapeWeb:
    def test_output_is_the_scraped_text_not_the_models_reflection(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("griptape_nodes_library.llm.tools.scrape_url", lambda url: f"page for {url}")
        model = _tool_then_text("get_content", {"url": "https://example.com"}, "model commentary")
        node = _create_node("ScrapeWeb")
        node.set_parameter_value("prompt", "https://example.com")

        with override_model(model):
            _run(node)

        assert node.parameter_output_values["output"] == "page for https://example.com"
        assert len(model.requests) == 1


class TestSearchWeb:
    @pytest.fixture(autouse=True)
    def _stub_search(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "griptape_nodes_library.llm.tools.search_web",
            lambda query, engine: [{"title": "T", "url": "https://u", "description": query}],
        )

    def test_without_summarize_output_is_the_raw_results(self) -> None:
        node = _create_node("SearchWeb")
        node.set_parameter_value("prompt", "cats")
        node.set_parameter_value("summarize", False)

        model = _tool_then_text("search", {"query": "cats"}, "model commentary")
        with override_model(model):
            _run(node)

        assert len(model.requests) == 1

        assert json.loads(node.parameter_output_values["output"]) == {
            "title": "T",
            "url": "https://u",
            "description": "cats",
        }

    def test_summarize_streams_the_models_answer(self) -> None:
        node = _create_node("SearchWeb")
        node.set_parameter_value("prompt", "cats")
        node.set_parameter_value("summarize", True)

        with override_model(_tool_then_text("search", {"query": "cats"}, "cats are great")):
            _run(node)

        assert node.parameter_output_values["output"] == "cats are great"


class TestAskulator:
    def test_reasoning_and_answer_stream_to_their_parameters(self) -> None:
        answer = json.dumps({"reasoning": "Two plus two.\nIt is four.", "final_answer": "4"})
        node = _create_node("Askulator")
        node.set_parameter_value("instruction", "2 + 2")

        with override_model(text_model(answer)):
            _run(node)

        assert node.parameter_output_values["output"] == "Two plus two.\nIt is four."
        assert node.parameter_output_values["result"] == "4"

    def test_calculator_tool_use_is_announced(self) -> None:
        answer = json.dumps({"reasoning": "r", "final_answer": "6"})
        node = _create_node("Askulator")
        node.set_parameter_value("instruction", "3 * 2")

        with override_model(_tool_then_text("calculate", {"expression": "3 * 2"}, answer)):
            _run(node)

        assert node.parameter_output_values["output"].startswith("Using a calculate\n")
        assert node.parameter_output_values["result"] == "6"


class TestEvaluateTextResult:
    def test_generates_steps_then_scores_on_a_ten_point_scale(self) -> None:
        seen: list[list[ModelMessage]] = []

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            seen.append(messages)
            tool = info.output_tools[0]
            if "steps" in tool.parameters_json_schema["properties"]:
                args: dict[str, Any] = {"steps": ["Compare meaning", "Check facts"]}
            else:
                args = {"score": 8, "reason": "Close paraphrase."}
            return ModelResponse(parts=[ToolCallPart(tool.name, args)])

        node = _create_node("EvaluateTextResult")
        node.set_parameter_value("criteria", "Is it a paraphrase?")
        node.set_parameter_value("input", "the input")
        node.set_parameter_value("expected_output", "the expected")
        node.set_parameter_value("actual_output", "the actual")

        with override_model(fake_model(respond)):
            _run(node)

        assert node.parameter_output_values["score"] == pytest.approx(0.8)
        assert node.parameter_output_values["reason"] == "Close paraphrase."
        steps_instructions = _instructions(seen[0])
        assert "Input, Actual Output, Expected Output" in steps_instructions
        assert "Is it a paraphrase?" in steps_instructions
        results_instructions = _instructions(seen[1])
        assert "['Compare meaning', 'Check facts']" in results_instructions
        assert "Input: the input\n\nActual Output: the actual\n\nExpected Output: the expected" in results_instructions

    def test_empty_criteria_is_rejected(self) -> None:
        node = _create_node("EvaluateTextResult")
        node.set_parameter_value("criteria", "")

        with pytest.raises(ValueError, match="criteria must not be empty"):
            next(node.process())


class TestRandomText:
    def test_empty_input_generates_a_sentence_with_the_model(self) -> None:
        node = _create_node("RandomText")

        with override_model(text_model("A random sentence.")):
            assert node._generate_with_agent("sentence", seed=7) == "A random sentence."

    def test_node_builds_without_a_cloud_credential(self) -> None:
        assert _create_node("RandomText") is not None


class TestCreateAgentSchema:
    def test_ruleset_example_emits_a_ruleset_dict(self) -> None:
        node = _create_node("CreateAgentSchema")

        node.set_parameter_value("ruleset_example", "Rule one.\n\nRule two.")

        assert node.get_parameter_value("agent_ruleset") == {
            "name": "schema_ruleset",
            "rules": ["Rule one.", "Rule two."],
        }


class TestMCPTaskNode:
    def _node(self, monkeypatch: pytest.MonkeyPatch) -> MCPTaskNode:
        node = cast("MCPTaskNode", _create_node("MCPTaskNode"))
        module = sys.modules[type(node).__module__]
        monkeypatch.setattr(module, "get_server_config", lambda _name: {"transport": "stdio", "rules": "Be brief."})
        toolset = FunctionToolset()

        @toolset.tool_plain
        def ping() -> str:
            return "pong"

        monkeypatch.setattr(type(node), "_build_toolsets", lambda self, state, config, name: [toolset])  # noqa: ARG005
        node.set_parameter_value("prompt", "ping the server")
        return node

    def test_runs_with_the_mcp_toolset_and_emits_an_agent_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        node = self._node(monkeypatch)
        model = _tool_then_text("ping", {}, "The server said pong.")

        with override_model(model):
            _run(node)

        assert node.parameter_output_values["output"] == "The server said pong."
        assert node._execution_succeeded is True
        state = AgentState.from_wire(node.parameter_output_values["agent"])
        assert state.model is not None
        assert state.runs() == [
            {
                "input": "ping the server",
                "output": "[Verified tool use:\n  Tool: ping\n  Result: pong\n]\n\nThe server said pong.",
            }
        ]
        assert "Be brief." in _instructions(model.requests[0])

    def test_continues_a_connected_agents_conversation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        node = self._node(monkeypatch)
        previous = AgentState(rulesets=[{"name": "mine", "rules": ["Rule A"]}]).with_runs(
            [{"input": "earlier question", "output": "earlier answer"}]
        )
        previous.model = AgentState.from_wire(
            {"agent": {"prompt_driver": {"type": "GriptapeCloudPromptDriver", "model": "gpt-4.1"}}, "tools": []}
        ).model
        node.set_parameter_value("agent", previous.to_wire())
        model = _tool_then_text("ping", {}, "pong received")

        with override_model(model):
            _run(node)

        state = AgentState.from_wire(node.parameter_output_values["agent"])
        assert [r["input"] for r in state.runs()] == ["earlier question", "ping the server"]
        assert state.rulesets == [{"name": "mine", "rules": ["Rule A"]}]
        assert "Rule A" in _instructions(model.requests[0])

    def test_failure_is_reported_through_status_outputs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        node = self._node(monkeypatch)

        def boom(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:  # noqa: ARG001
            msg = "provider exploded"
            raise RuntimeError(msg)

        with override_model(fake_model(boom)), pytest.raises(RuntimeError, match="provider exploded"):
            _run(node)

        assert node._execution_succeeded is False
        assert "provider exploded" in node.get_parameter_value("result_details")

        assert node._execution_succeeded is False
        assert "provider exploded" in node.get_parameter_value("result_details")

    def test_max_subtasks_caps_tool_rounds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        node = self._node(monkeypatch)
        node.set_parameter_value("max_subtasks", 2)
        requests: list[list[ModelMessage]] = []

        def always_ping(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:  # noqa: ARG001
            requests.append(messages)
            return ModelResponse(parts=[ToolCallPart("ping", {})])

        with override_model(fake_model(always_ping)), pytest.raises(UsageLimitExceeded):
            _run(node)

        assert len(requests) == 3


class TestMCPToolsetConstruction:
    def test_builds_the_incoming_agents_tools_and_the_mcp_server(self) -> None:
        node = _create_node("MCPTaskNode")
        state = AgentState(tools=[{"tool_type": "Calculator"}])
        config = {
            "tool_type": "MCPTool",
            "mcp_server_name": "demo",
            "server_config": {"transport": "stdio", "command": "echo", "args": ["hi"]},
        }

        toolsets = node._build_toolsets(state, config, "demo")

        assert len(toolsets) == 2

    def test_unbuildable_mcp_server_is_an_error(self) -> None:
        node = _create_node("MCPTaskNode")
        config = {"tool_type": "MCPTool", "mcp_server_name": "demo", "server_config": {"transport": "stdio"}}

        with pytest.raises(RuntimeError, match="Failed to create MCP tool for server 'demo'"):
            node._build_toolsets(AgentState(), config, "demo")
