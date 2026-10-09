from __future__ import annotations

from typing import Any

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo

from griptape_nodes_library.agents.memory.clear_agent_memory import ClearAgentMemory
from griptape_nodes_library.agents.memory.display_agent_memory import DisplayAgentMemory
from griptape_nodes_library.agents.memory.replace_item_in_agent_memory import ReplaceItemInAgentMemory
from griptape_nodes_library.agents.memory.summarize_agent_memory import SummarizeAgentMemory
from griptape_nodes_library.convert.agent_to_tool import AgentToTool
from griptape_nodes_library.llm.agent_state import AgentState, messages_from_runs
from griptape_nodes_library.llm.model_config import ModelConfig, ModelProvider
from griptape_nodes_library.llm.models import override_model
from griptape_nodes_library.llm.testing import fake_model, text_model
from griptape_nodes_library.llm.tools import build_toolsets

CLOUD = ModelConfig(provider=ModelProvider.GRIPTAPE_CLOUD, model="gpt-4.1")
RUNS = [{"input": "first q", "output": "first a"}, {"input": "second q", "output": "second a"}]


def _build_agent_wire() -> dict:
    return AgentState(
        model=CLOUD,
        messages=messages_from_runs(RUNS),
        tools=[{"tool_type": "Calculator"}],
        rulesets=[{"name": "r", "rules": ["be nice"]}],
    ).to_wire()


AGENT_WIRE = _build_agent_wire()
"""Built once: message timestamps differ between builds."""


def _drive(node: Any) -> None:
    """Run `process()`, calling each thunk a generator `process()` yields."""
    gen = node.process()
    if gen is None:
        return
    try:
        func = next(gen)
        while True:
            func = gen.send(func())
    except StopIteration:
        return


def test_display_agent_memory_lists_runs_and_passes_agent_through() -> None:
    node = DisplayAgentMemory(name="Display")
    node.set_parameter_value("agent", AGENT_WIRE)

    _drive(node)

    assert node.parameter_output_values["memory"] == {"runs": RUNS}
    assert node.parameter_output_values["agent"] == AGENT_WIRE


def test_display_agent_memory_reads_legacy_agent_wrappers() -> None:
    legacy = {
        "agent": {"conversation_memory": {"runs": [{"input": {"value": "q"}, "output": {"value": "a"}}]}},
        "tools": [],
        "rulesets": [],
    }
    node = DisplayAgentMemory(name="Display")
    node.set_parameter_value("agent", legacy)

    _drive(node)

    assert node.parameter_output_values["memory"] == {"runs": [{"input": "q", "output": "a"}]}


def test_display_agent_memory_without_agent_is_empty() -> None:
    node = DisplayAgentMemory(name="Display")

    _drive(node)

    assert node.parameter_output_values["memory"] == {"runs": []}


def test_clear_agent_memory_drops_history_and_keeps_the_rest() -> None:
    node = ClearAgentMemory(name="Clear")
    node.set_parameter_value("agent", AGENT_WIRE)

    _drive(node)

    state = AgentState.from_wire(node.parameter_output_values["agent"])
    assert state.messages == []
    assert state.model == CLOUD
    assert state.tools == [{"tool_type": "Calculator"}]
    assert state.rulesets == [{"name": "r", "rules": ["be nice"]}]


def test_replace_item_in_agent_memory_replaces_selected_run() -> None:
    node = ReplaceItemInAgentMemory(name="Replace")
    node.set_parameter_value("agent", AGENT_WIRE)
    node.set_parameter_value("memory_to_replace", "1: second q")
    node.set_parameter_value("new_output", "edited answer")

    _drive(node)

    state = AgentState.from_wire(node.parameter_output_values["agent"])
    assert state.runs() == [RUNS[0], {"input": "second q", "output": "edited answer"}]
    assert state.model == CLOUD


def test_replace_item_choices_come_from_agent_runs() -> None:
    node = ReplaceItemInAgentMemory(name="Replace")

    node.set_parameter_value("agent", AGENT_WIRE)

    choices = node.get_parameter_by_name("memory_to_replace")
    assert choices is not None
    assert node.get_parameter_value("memory_to_replace") == "0: first q"
    assert node.parameter_output_values["orig_output"] == "first a"


def test_summarize_agent_memory_replaces_runs_with_the_summary() -> None:
    node = SummarizeAgentMemory(name="Summarize")
    node.set_parameter_value("agent", AGENT_WIRE)

    with override_model(text_model("they talked")):
        _drive(node)

    assert node.parameter_output_values["summary"] == "they talked"
    state = AgentState.from_wire(node.parameter_output_values["agent"])
    assert state.runs() == [{"input": "conversation summary", "output": "they talked"}]
    assert state.model == CLOUD
    assert state.tools == [{"tool_type": "Calculator"}]


def test_summarize_agent_memory_sends_the_history_and_prompt() -> None:
    node = SummarizeAgentMemory(name="Summarize")
    node.set_parameter_value("agent", AGENT_WIRE)
    node.set_parameter_value("prompt", "Summarize please")
    seen: list[list[str]] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append([str(p.content) for m in messages for p in m.parts if isinstance(p, (UserPromptPart, TextPart))])
        return ModelResponse(parts=[TextPart("sum")])

    with override_model(fake_model(respond)):
        _drive(node)

    assert seen == [["first q", "first a", "second q", "second a", "Summarize please"]]


def test_summarize_agent_memory_with_no_runs_emits_agent_unchanged() -> None:
    empty = AgentState(model=CLOUD).to_wire()
    node = SummarizeAgentMemory(name="Summarize")
    node.set_parameter_value("agent", empty)

    _drive(node)

    assert node.parameter_output_values["agent"] == empty
    assert "summary" not in node.parameter_output_values


def test_agent_to_tool_emits_a_buildable_agent_tool_config() -> None:
    node = AgentToTool(name="ToTool")
    node.set_parameter_value("agent", AGENT_WIRE)
    node.set_parameter_value("name", "Helper")
    node.set_parameter_value("description", "Helps")

    _drive(node)

    tool = node.parameter_output_values["tool"]
    assert tool["tool_type"] == "AgentTool"
    assert tool["agent_dict"] == AGENT_WIRE
    assert (tool["name"], tool["description"]) == ("Helper", "Helps")
    assert len(build_toolsets([tool])) == 1


def test_agent_to_tool_without_agent_emits_nothing() -> None:
    node = AgentToTool(name="ToTool")

    _drive(node)

    assert node.parameter_output_values["tool"] is None


@pytest.mark.parametrize("node_class", [ClearAgentMemory, DisplayAgentMemory, SummarizeAgentMemory])
def test_memory_nodes_keep_the_agent_parameter(node_class: type) -> None:
    node = node_class(name="n")

    param = node.get_parameter_by_name("agent")

    assert param is not None
    assert param.input_types == ["Agent"]
