from __future__ import annotations

import json
from typing import Any

import pytest
from griptape_nodes.retained_mode.events.agent_events import ProviderConfig
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo

import griptape_nodes_library.utils.model_invocation as model_invocation_module
from griptape_nodes_library.agents.agent import Agent
from griptape_nodes_library.llm.agent_state import AgentState, messages_from_runs
from griptape_nodes_library.llm.model_config import ModelConfig, ModelProvider
from griptape_nodes_library.llm.models import override_model
from griptape_nodes_library.llm.testing import fake_model, text_model

CLOUD = ModelConfig(provider=ModelProvider.GRIPTAPE_CLOUD, model="gpt-4.1")


class _AllowedDeclaration:
    result_details = ""

    def failed(self) -> bool:
        return False


@pytest.fixture(autouse=True)
def _allow_model_invocation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        model_invocation_module, "declare_model_invocation_sync", lambda _node, _model: _AllowedDeclaration()
    )


def _run(node: Agent) -> None:
    """Drive `process()` the way the engine does: run each yielded callable and send its result back."""
    gen = node.process()
    try:
        func = next(gen)
        while True:
            func = gen.send(func())
    except StopIteration:
        return


def _prepare(node: Agent, **values: Any) -> None:
    for name, value in values.items():
        node.set_parameter_value(name, value)


def _stub_list_values(node: Agent, monkeypatch: pytest.MonkeyPatch, **lists: list[Any]) -> None:
    original = node.get_parameter_list_value
    monkeypatch.setattr(node, "get_parameter_list_value", lambda name: lists.get(name, original(name)))


def _user_text(messages: list[ModelMessage]) -> list[str]:
    return [
        str(part.content)
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if part.part_kind == "user-prompt"
    ]


def _recording_model(reply: str, seen: list[tuple[list[ModelMessage], AgentInfo]]):
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append((list(messages), info))
        return ModelResponse(parts=[TextPart(reply)])

    return fake_model(respond)


def test_prompt_streams_into_output_and_updates_agent_wire(agent_node: Agent) -> None:
    _prepare(agent_node, prompt="Hello there", include_details=True)

    with override_model(text_model("General Kenobi")):
        _run(agent_node)

    assert agent_node.get_parameter_value("output") == "General Kenobi"
    state = AgentState.from_wire(agent_node.parameter_output_values["agent"])
    assert state.runs() == [{"input": "Hello there", "output": "General Kenobi"}]
    assert state.model is not None
    assert state.model.provider == ModelProvider.GRIPTAPE_CLOUD
    assert "General Kenobi" in agent_node.parameter_output_values["logs"]


def test_no_prompt_creates_agent_without_running_the_model(agent_node: Agent) -> None:
    def must_not_run(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise AssertionError

    with override_model(fake_model(must_not_run)):
        _run(agent_node)

    assert agent_node.parameter_output_values["output"] == "Agent created."
    assert AgentState.from_wire(agent_node.parameter_output_values["agent"]).messages == []


def test_incoming_agent_supplies_model_history_tools_and_merges_rulesets(
    agent_node: Agent, monkeypatch: pytest.MonkeyPatch
) -> None:
    upstream = AgentState(
        model=CLOUD,
        messages=messages_from_runs([{"input": "earlier question", "output": "earlier answer"}]),
        tools=[{"tool_type": "Calculator"}],
        rulesets=[{"name": "upstream", "rules": ["be brief"]}],
    )
    _prepare(agent_node, agent=upstream.to_wire(), prompt="next question")
    _stub_list_values(agent_node, monkeypatch, rulesets=["speak French"])
    seen: list[tuple[list[ModelMessage], AgentInfo]] = []

    with override_model(_recording_model("ok", seen)):
        _run(agent_node)

    messages, info = seen[0]
    assert _user_text(messages) == ["earlier question", "next question"]
    assert info.instructions is not None
    assert "be brief" in info.instructions
    assert "speak French" in info.instructions
    assert any(tool.name == "calculate" for tool in info.function_tools)
    state = AgentState.from_wire(agent_node.parameter_output_values["agent"])
    assert state.model == CLOUD
    assert state.tools == [{"tool_type": "Calculator"}]
    assert [r["name"] for r in state.rulesets] == ["upstream", "behavior_1"]
    assert [r["input"] for r in state.runs()] == ["earlier question", "next question"]


def test_legacy_agent_wrapper_is_still_accepted(agent_node: Agent) -> None:
    legacy = {
        "agent": {
            "tasks": [{"prompt_driver": {"type": "GriptapeCloudPromptDriver", "model": "gpt-4.1"}}],
            "conversation_memory": {"runs": [{"input": {"value": "old q"}, "output": {"value": "old a"}}]},
        },
        "tools": [],
        "rulesets": [],
    }
    _prepare(agent_node, agent=legacy, prompt="new q")
    seen: list[tuple[list[ModelMessage], AgentInfo]] = []

    with override_model(_recording_model("ok", seen)):
        _run(agent_node)

    assert _user_text(seen[0][0]) == ["old q", "new q"]
    assert AgentState.from_wire(agent_node.parameter_output_values["agent"]).model == CLOUD


@pytest.mark.parametrize(
    "agent_memory",
    [
        {"runs": [{"input": "simple q", "output": "simple a"}]},
        {"conversation_memory": {"runs": [{"input": {"value": "simple q"}, "output": {"value": "simple a"}}]}},
        json.dumps({"runs": [{"input": "simple q", "output": "simple a"}]}),
    ],
)
def test_agent_memory_parameter_replaces_history(agent_node: Agent, agent_memory: Any) -> None:
    _prepare(agent_node, prompt="follow up", agent_memory=agent_memory)
    seen: list[tuple[list[ModelMessage], AgentInfo]] = []

    with override_model(_recording_model("ok", seen)):
        _run(agent_node)

    assert _user_text(seen[0][0]) == ["simple q", "follow up"]


def test_connected_model_config_is_used_and_its_provider_kept(agent_node: Agent) -> None:
    config = ModelConfig(provider=ModelProvider.ANTHROPIC, model="claude-fake", settings={"temperature": 0.3})
    agent_node.parameter_values["model"] = config
    _prepare(agent_node, prompt="hi")

    with override_model(text_model("ok")):
        _run(agent_node)

    assert AgentState.from_wire(agent_node.parameter_output_values["agent"]).model == config


def test_third_party_provider_wire_holds_the_secret_name_not_the_key(
    agent_node: Agent, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = ProviderConfig(
        name="my-llm",
        type="openai",
        model="",
        base_url="http://llm.local/v1",
        api_key_secret_name="MY_LLM_KEY",  # noqa: S106
    )
    monkeypatch.setattr(agent_node._provider, "_fetch_providers", lambda: [provider])
    monkeypatch.setattr(agent_node._provider, "resolve_provider_api_key", lambda _p: "sk-super-secret")
    agent_node.parameter_values["model_provider"] = "my-llm"
    agent_node.parameter_values["model"] = "llama-3"
    _prepare(agent_node, prompt="hi")

    with override_model(text_model("ok")):
        _run(agent_node)

    wire = agent_node.parameter_output_values["agent"]
    assert "sk-super-secret" not in json.dumps(wire)
    assert wire["model"] == {
        "provider": "openai_compatible",
        "model": "llama-3",
        "base_url": "http://llm.local/v1",
        "api_key_secret": "MY_LLM_KEY",
        "settings": {},
        "options": {},
    }


@pytest.mark.parametrize(
    ("provider_type", "expected"),
    [
        ("ollama", ModelProvider.OLLAMA),
        ("lmstudio", ModelProvider.LMSTUDIO),
        ("anything-else", ModelProvider.OPENAI_COMPATIBLE),
    ],
)
def test_third_party_provider_type_maps_to_model_provider(
    agent_node: Agent, monkeypatch: pytest.MonkeyPatch, provider_type: str, expected: ModelProvider
) -> None:
    provider = ProviderConfig(name="p", type=provider_type, model="")
    monkeypatch.setattr(agent_node._provider, "_fetch_providers", lambda: [provider])

    config = agent_node._resolve_model_config("m", "p")

    assert config.provider == expected
    assert config.base_url is None
    assert config.api_key_secret is None


def test_tool_calls_are_logged_when_details_are_on(agent_node: Agent, monkeypatch: pytest.MonkeyPatch) -> None:
    _prepare(agent_node, prompt="what is 2+2", include_details=True)
    _stub_list_values(agent_node, monkeypatch, tools=[{"tool_type": "Calculator"}])

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if any(part.part_kind == "tool-return" for m in messages if isinstance(m, ModelRequest) for part in m.parts):
            return ModelResponse(parts=[TextPart("The answer is 4")])
        return ModelResponse(parts=[ToolCallPart("calculate", {"expression": "2+2"})])

    with override_model(fake_model(respond)):
        _run(agent_node)

    logs = agent_node.parameter_output_values["logs"]
    assert "[Tools]: Calculator" in logs
    assert "[Using tool calculate:" in logs
    assert agent_node.get_parameter_value("output") == "The answer is 4"
    state = AgentState.from_wire(agent_node.parameter_output_values["agent"])
    assert state.runs() == [
        {
            "input": "what is 2+2",
            "output": '[Verified tool use:\n  Tool: calculate\n  Input: {"expression":"2+2"}\n  Result: 4\n]\n\nThe answer is 4',
        }
    ]


def test_tool_calls_are_not_logged_without_details(agent_node: Agent, monkeypatch: pytest.MonkeyPatch) -> None:
    _prepare(agent_node, prompt="what is 2+2")
    _stub_list_values(agent_node, monkeypatch, tools=[{"tool_type": "Calculator"}])
    calls = {"n": 0}

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls["n"] += 1
        if calls["n"] == 1:
            return ModelResponse(parts=[ToolCallPart("calculate", {"expression": "2+2"})])
        return ModelResponse(parts=[TextPart("4")])

    with override_model(fake_model(respond)):
        _run(agent_node)

    assert "Using tool" not in agent_node.parameter_output_values["logs"]


def test_live_tool_objects_are_rejected(agent_node: Agent, monkeypatch: pytest.MonkeyPatch) -> None:
    _prepare(agent_node, prompt="hi")
    _stub_list_values(agent_node, monkeypatch, tools=[object()])

    with pytest.raises(TypeError, match="Unsupported tool value"), override_model(text_model("ok")):
        _run(agent_node)


def test_output_schema_produces_json_text(agent_node: Agent) -> None:
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}
    _prepare(agent_node, prompt="hi", output_schema=schema)

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        tool = info.output_tools[0]
        return ModelResponse(parts=[ToolCallPart(tool.name, {"answer": "42"})])

    with override_model(fake_model(respond)):
        _run(agent_node)

    assert json.loads(agent_node.get_parameter_value("output")) == {"answer": "42"}


def test_cancellation_logs_and_still_emits_the_agent(agent_node: Agent, monkeypatch: pytest.MonkeyPatch) -> None:
    upstream = AgentState(model=CLOUD, messages=messages_from_runs([{"input": "q", "output": "a"}]))
    _prepare(agent_node, agent=upstream.to_wire(), prompt="more")
    monkeypatch.setattr(Agent, "is_cancellation_requested", property(lambda _self: True))

    with override_model(text_model("never seen")):
        _run(agent_node)

    assert "[Agent execution cancelled by user.]" in agent_node.parameter_output_values["logs"]
    assert not agent_node.parameter_output_values.get("output")
    state = AgentState.from_wire(agent_node.parameter_output_values["agent"])
    assert state.runs() == [{"input": "q", "output": "a"}]


def test_additional_context_dict_renders_the_prompt_template(agent_node: Agent) -> None:
    _prepare(agent_node, prompt="Hello {{ name }}")
    agent_node.parameter_values["additional_context"] = {"name": "Ada"}
    seen: list[tuple[list[ModelMessage], AgentInfo]] = []

    with override_model(_recording_model("ok", seen)):
        _run(agent_node)

    assert _user_text(seen[0][0]) == ["Hello Ada"]
