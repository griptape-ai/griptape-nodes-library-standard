"""Griptape-era `Agent` values main saved, fed straight into every node that takes an agent."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo

import griptape_nodes_library.utils.model_invocation as model_invocation_module
from griptape_nodes_library.agents.agent import Agent
from griptape_nodes_library.agents.memory.clear_agent_memory import ClearAgentMemory
from griptape_nodes_library.agents.memory.display_agent_memory import DisplayAgentMemory
from griptape_nodes_library.agents.memory.replace_item_in_agent_memory import ReplaceItemInAgentMemory
from griptape_nodes_library.agents.memory.summarize_agent_memory import SummarizeAgentMemory
from griptape_nodes_library.convert.agent_to_tool import AgentToTool
from griptape_nodes_library.llm.agent_state import AgentState
from griptape_nodes_library.llm.models import override_model
from griptape_nodes_library.llm.testing import fake_model
from griptape_nodes_library.llm.tools import build_toolset

from .harness import saved_values

TEMPLATES = Path(__file__).parents[3] / "workflows" / "templates"


def _legacy_values() -> dict[str, dict]:
    return {
        "wrapper": saved_values("agent_memory")[("a2", "agent", True)],
        "wrapper_with_tools": saved_values("agent_tools_rules")[("a1", "agent", True)],
        "wrapper_with_provider": saved_values("third_party_provider")[("a2", "agent", True)],
        "bare_to_dict": saved_values(TEMPLATES / "fill_in_the_story.py")[("Agent", "agent", True)],
    }


LEGACY = _legacy_values()


def _legacy_runs(value: dict) -> list[dict[str, str]]:
    runs = value.get("agent", value)["conversation_memory"]["runs"]
    return [{"input": r["input"]["value"], "output": r["output"]["value"]} for r in runs]


class _Allowed:
    result_details = ""

    def failed(self) -> bool:
        return False


@pytest.fixture(autouse=True)
def _allow_model_invocation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(model_invocation_module, "declare_model_invocation_sync", lambda _node, _model: _Allowed())
    monkeypatch.setenv("GT_CLOUD_API_KEY", "gt-compat-fake")


@pytest.fixture
def prompts() -> Any:
    seen: list[list[str]] = []

    def respond(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        seen.append(
            [
                str(p.content)
                for m in messages
                if isinstance(m, ModelRequest)
                for p in m.parts
                if isinstance(p, UserPromptPart)
            ]
        )
        return ModelResponse(parts=[TextPart(f"reply {len(seen)}")])

    with override_model(fake_model(respond)):
        yield seen


def _drive(node: Any) -> None:
    gen = node.process()
    if gen is None:
        return
    try:
        func = next(gen)
        while True:
            func = gen.send(func())
    except StopIteration:
        return


@pytest.fixture(params=sorted(LEGACY))
def legacy(request: pytest.FixtureRequest) -> dict:
    return LEGACY[request.param]


def test_agent_continues_legacy_agent(legacy: dict, prompts: list[list[str]]) -> None:
    node = Agent(name="Agent")
    node.set_parameter_value("agent", legacy)
    node.set_parameter_value("prompt", "next")
    _drive(node)

    history = [r["input"] for r in _legacy_runs(legacy)]
    assert prompts[-1] == [*history, "next"]
    state = AgentState.from_wire(node.parameter_output_values["agent"])
    assert state.runs()[:-1] == AgentState.from_wire(legacy).runs()
    assert state.tools == AgentState.from_wire(legacy).tools
    assert state.rulesets == AgentState.from_wire(legacy).rulesets
    assert "sk-compat-fake" not in repr(node.parameter_output_values["agent"])


def test_display_memory(legacy: dict) -> None:
    node = DisplayAgentMemory(name="Display")
    node.set_parameter_value("agent", legacy)
    node.process()
    assert node.parameter_output_values["memory"] == {"runs": AgentState.from_wire(legacy).runs()}


def test_replace_memory(legacy: dict) -> None:
    node = ReplaceItemInAgentMemory(name="Replace")
    node.set_parameter_value("agent", legacy)
    choice = node.get_parameter_value("memory_to_replace")
    assert choice.startswith("0: ")
    node.set_parameter_value("new_input", "swapped in")
    node.set_parameter_value("new_output", "swapped out")
    node.process()
    runs = AgentState.from_wire(node.parameter_output_values["agent"]).runs()
    assert runs[0] == {"input": "swapped in", "output": "swapped out"}
    assert runs[1:] == AgentState.from_wire(legacy).runs()[1:]


def test_summarize_memory(legacy: dict, prompts: list[list[str]]) -> None:
    node = SummarizeAgentMemory(name="Summarize")
    node.set_parameter_value("agent", legacy)
    node.process()
    assert prompts[-1][:-1] == [r["input"] for r in _legacy_runs(legacy)]
    assert AgentState.from_wire(node.parameter_output_values["agent"]).runs() == [
        {"input": "conversation summary", "output": "reply 1"}
    ]


def test_clear_memory(legacy: dict) -> None:
    node = ClearAgentMemory(name="Clear")
    node.set_parameter_value("agent", legacy)
    node.process()
    state = AgentState.from_wire(node.parameter_output_values["agent"])
    assert state.runs() == []
    legacy_model = AgentState.from_wire(legacy).model
    assert state.model is not None and legacy_model is not None
    assert state.model.to_wire() == legacy_model.to_wire()


@pytest.mark.asyncio
async def test_agent_to_tool(legacy: dict, prompts: list[list[str]]) -> None:
    node = AgentToTool(name="AgentToTool")
    node.set_parameter_value("agent", legacy)
    node.set_parameter_value("name", "Helper")
    node.set_parameter_value("description", "Helps.")
    node.process()
    toolset = build_toolset(node.parameter_output_values["tool"])
    assert toolset is not None
    result = await toolset.tools["Helper"].function("hello")  # type: ignore[attr-defined]
    assert result == "reply 1"
    assert prompts[-1] == [*(r["input"] for r in _legacy_runs(legacy)), "hello"]
