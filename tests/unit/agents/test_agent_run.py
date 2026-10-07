"""End-to-end ``Agent.process`` runs against a pydantic-ai test model."""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

import pytest
from griptape.artifacts import ImageUrlArtifact
from griptape_nodes.exe_types.core_types import ParameterList
from pydantic_ai.messages import ImageUrl
from pydantic_ai.models.test import TestModel

import griptape_nodes_library.agents.agent as agent_module
import griptape_nodes_library.utils.model_invocation as model_invocation_module
from griptape_nodes_library.agents.agent import Agent
from griptape_nodes_library.agents.memory.clear_agent_memory import ClearAgentMemory
from griptape_nodes_library.agents.memory.display_agent_memory import DisplayAgentMemory
from griptape_nodes_library.agents.memory.replace_item_in_agent_memory import ReplaceItemInAgentMemory
from griptape_nodes_library.agents.memory.summarize_agent_memory import SummarizeAgentMemory
from griptape_nodes_library.utils.agent_state import AgentState
from griptape_nodes_library.utils.local_agent_runner import LocalAgentRunner


class _Allowed:
    result_details = ""

    def failed(self) -> bool:
        return False


class _TestRunner(LocalAgentRunner):
    def __init__(self, extra_tools: list | None = None) -> None:
        super().__init__(extra_tools=extra_tools or [], model_override=TestModel(custom_output_text="reply"))


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(model_invocation_module, "declare_model_invocation_sync", lambda _n, _m: _Allowed())
    monkeypatch.setattr(agent_module, "LocalAgentRunner", _TestRunner)


def _set_list(node: Agent, name: str, values: list[object]) -> None:
    parameter_list = node.get_parameter_by_name(name)
    assert isinstance(parameter_list, ParameterList)
    for value in values:
        node.set_parameter_value(parameter_list.add_child_parameter().name, value)


def _run(node: Agent) -> None:
    gen: Generator[Any, Any, None] = node.process()
    try:
        work = next(gen)
        gen.send(work())
    except StopIteration:
        pass


def _agent_out(node: Agent) -> AgentState:
    state = AgentState.from_wire(node.parameter_output_values["agent"])
    assert state is not None
    return state


def test_run_outputs_reply_and_agent_with_the_turn(agent_node: Agent) -> None:
    agent_node.set_parameter_value("prompt", "Hi")
    _set_list(agent_node, "rulesets", ["Be brief"])

    _run(agent_node)

    state = _agent_out(agent_node)
    assert agent_node.get_parameter_value("output") == "reply"
    assert [(t.prompt, t.response) for t in state.turns()] == [("Hi", "reply")]
    assert state.rulesets == [{"name": "behavior_1", "rules": ["Be brief"]}]
    assert "api_key" not in agent_node.parameter_output_values["agent"]["provider"]


def test_images_go_with_the_prompt(agent_node: Agent) -> None:
    agent_node.set_parameter_value("prompt", "What is this?")
    _set_list(agent_node, "images", [ImageUrlArtifact("http://localhost/a.png")])

    _run(agent_node)

    turn = _agent_out(agent_node).turns()[0]
    assert list(turn.user_content) == ["What is this?", ImageUrl(url="http://localhost/a.png")]


def test_connected_agent_continues_its_conversation(agent_node: Agent) -> None:
    agent_node.set_parameter_value("prompt", "First")
    _run(agent_node)
    upstream = agent_node.parameter_output_values["agent"]

    second = Agent(name="Agent2")
    second.set_parameter_value("agent", upstream)
    second.set_parameter_value("prompt", "Second")
    _run(second)

    assert [t.prompt for t in _agent_out(second).turns()] == ["First", "Second"]


def test_no_prompt_creates_the_agent_without_running(agent_node: Agent) -> None:
    agent_node.set_parameter_value("prompt", "")

    with pytest.raises(StopIteration):
        next(agent_node.process())

    assert agent_node.parameter_output_values["output"] == "Agent created."
    assert _agent_out(agent_node).messages == []


def _two_turn_agent(agent_node: Agent) -> dict:
    agent_node.set_parameter_value("prompt", "First")
    _run(agent_node)
    second = Agent(name="Agent2")
    second.set_parameter_value("agent", agent_node.parameter_output_values["agent"])
    second.set_parameter_value("prompt", "Second")
    _run(second)
    return second.parameter_output_values["agent"]


def test_display_memory(agent_node: Agent) -> None:
    node = DisplayAgentMemory(name="Display")
    node.set_parameter_value("agent", _two_turn_agent(agent_node))

    node.process()

    assert node.parameter_output_values["memory"] == {
        "runs": [{"input": "First", "output": "reply"}, {"input": "Second", "output": "reply"}]
    }


def test_clear_memory(agent_node: Agent) -> None:
    node = ClearAgentMemory(name="Clear")
    node.set_parameter_value("agent", _two_turn_agent(agent_node))

    node.process()

    state = AgentState.from_wire(node.parameter_output_values["agent"])
    assert state is not None
    assert state.messages == []


def test_replace_item_in_memory(agent_node: Agent) -> None:
    node = ReplaceItemInAgentMemory(name="Replace")
    node.set_parameter_value("agent", _two_turn_agent(agent_node))
    node.set_parameter_value("memory_to_replace", "1: Second")
    assert node.parameter_output_values["orig_output"] == "reply"
    node.set_parameter_value("new_output", "edited")

    node.process()

    state = AgentState.from_wire(node.parameter_output_values["agent"])
    assert state is not None
    assert [(t.prompt, t.response) for t in state.turns()] == [("First", "reply"), ("Second", "edited")]


def test_summarize_memory(agent_node: Agent, monkeypatch: pytest.MonkeyPatch) -> None:
    import griptape_nodes_library.agents.memory.summarize_agent_memory as summarize_module

    monkeypatch.setattr(summarize_module, "LocalAgentRunner", _TestRunner)
    node = SummarizeAgentMemory(name="Summarize")
    node.set_parameter_value("agent", _two_turn_agent(agent_node))

    node.process()

    state = AgentState.from_wire(node.parameter_output_values["agent"])
    assert state is not None
    assert node.parameter_output_values["summary"] == "reply"
    assert [(t.prompt, t.response) for t in state.turns()] == [("conversation summary", "reply")]
