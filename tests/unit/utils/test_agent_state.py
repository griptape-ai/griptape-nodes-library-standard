"""Tests for ``AgentState``: the wire format, reading legacy griptape wrappers, and editing history."""

from __future__ import annotations

from griptape.drivers.prompt.griptape_cloud import GriptapeCloudPromptDriver
from griptape.drivers.prompt.openai import OpenAiChatPromptDriver
from griptape.structures import Agent as GtAgent
from griptape_nodes.drivers.cloud_models import ProviderID
from pydantic_ai.messages import (
    BinaryContent,
    ImageUrl,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from griptape_nodes_library.utils.agent_state import AGENT_STATE_FORMAT, AgentState, ProviderKind, ProviderRef
from griptape_nodes_library.utils.agent_utils import wrap_agent

PNG = b"\x89PNG\r\n\x1a\nfake"


def _two_turn_state() -> AgentState:
    return AgentState(
        model="gpt-4.1",
        messages=[
            ModelRequest(parts=[UserPromptPart(content=["What is this?", ImageUrl(url="http://localhost/a.png")])]),
            ModelResponse(parts=[ToolCallPart(tool_name="lookup", args={"q": "a"}, tool_call_id="c1")]),
            ModelRequest(parts=[ToolReturnPart(tool_name="lookup", content="a cat", tool_call_id="c1")]),
            ModelResponse(parts=[TextPart(content="A cat.")]),
            ModelRequest(parts=[UserPromptPart(content="Thanks")]),
            ModelResponse(parts=[TextPart(content="You're welcome.")]),
        ],
    )


def test_wire_round_trip_keeps_media_and_tool_calls() -> None:
    state = _two_turn_state()
    state.messages.append(
        ModelRequest(parts=[UserPromptPart(content=[BinaryContent(data=PNG, media_type="image/png")])])
    )

    wire = state.to_wire()
    restored = AgentState.from_wire(wire)

    assert wire["format"] == AGENT_STATE_FORMAT
    assert restored is not None
    assert restored.to_wire() == wire
    last_prompt = restored.turns()[-1].user_content
    assert isinstance(last_prompt[0], BinaryContent)
    assert last_prompt[0].data == PNG


def test_wire_carries_no_credential() -> None:
    state = AgentState(
        model="llama3",
        provider=ProviderRef(kind=ProviderKind.OPENAI_COMPATIBLE, name="local", api_key_secret="LOCAL_KEY"),
    )

    assert "api_key" not in state.to_wire()["provider"]


def test_from_wire_ignores_non_agents() -> None:
    assert AgentState.from_wire(None) is None
    assert AgentState.from_wire({}) is None
    assert AgentState.from_wire("agent") is None


def test_turns_group_tool_calls_with_their_prompt() -> None:
    turns = _two_turn_state().turns()

    assert [(t.start, t.end) for t in turns] == [(0, 4), (4, 6)]
    assert turns[0].prompt == "What is this?\n[image]"
    assert turns[0].response == "A cat."
    assert turns[1].prompt == "Thanks"


def test_replace_turn_keeps_unchanged_side_and_drops_tool_calls() -> None:
    state = _two_turn_state()

    state.replace_turn(0, response="A dog.")

    turns = state.turns()
    assert len(state.messages) == 4
    assert turns[0].response == "A dog."
    assert turns[0].user_content == ["What is this?", ImageUrl(url="http://localhost/a.png")]
    assert turns[1].response == "You're welcome."


def test_replace_history_leaves_one_turn() -> None:
    state = _two_turn_state()

    state.replace_history("conversation summary", "We talked about a cat.")

    assert [(t.prompt, t.response) for t in state.turns()] == [("conversation summary", "We talked about a cat.")]


def test_instructions_render_every_ruleset() -> None:
    state = AgentState(
        model="m",
        rulesets=[{"name": "tone", "rules": ["Be brief", ""]}, {"name": "empty", "rules": []}],
    )

    assert state.instructions() == 'Follow every rule below.\n\nRuleset "tone":\n- Be brief'
    assert AgentState(model="m").instructions() is None


def test_reads_legacy_cloud_wrapper_with_memory() -> None:
    agent = GtAgent(prompt_driver=GriptapeCloudPromptDriver(model="gpt-4.1", api_key="k", temperature=0.3))
    agent.conversation_memory.runs = []  # pyright: ignore[reportOptionalMemberAccess]
    agent_dict = agent.to_dict()
    agent_dict["conversation_memory"]["runs"] = [
        {"input": {"type": "TextArtifact", "value": "hi"}, "output": {"type": "TextArtifact", "value": "hello"}}
    ]
    tools = [{"tool_type": "DateTime"}]
    rulesets = [{"name": "behavior_1", "rules": ["Be brief"]}]

    state = AgentState.from_wire(wrap_agent(agent_dict, tools, rulesets))

    assert state is not None
    assert state.provider == ProviderRef()
    assert state.model == "gpt-4.1"
    assert state.model_settings["temperature"] == 0.3
    assert state.tools == tools
    assert state.rulesets == rulesets
    assert [(t.prompt, t.response) for t in state.turns()] == [("hi", "hello")]


def test_legacy_provider_blob_drops_plaintext_key() -> None:
    agent = GtAgent(prompt_driver=OpenAiChatPromptDriver(model="llama3", api_key="k", base_url="http://h/v1"))
    provider = {"name": "local", "type": "openai", "base_url": "http://h/v1", "api_key": "secret-value"}

    state = AgentState.from_wire(wrap_agent(agent.to_dict(), [], [], provider=provider))

    assert state is not None
    assert state.provider == ProviderRef(kind=ProviderKind.OPENAI_COMPATIBLE, name="local", base_url="http://h/v1")
    assert "secret-value" not in str(state.to_wire())


def test_legacy_ollama_blob_maps_to_ollama() -> None:
    agent = GtAgent(prompt_driver=OpenAiChatPromptDriver(model="llama3", api_key="k"))
    provider = {"name": "ollama", "type": ProviderID.OLLAMA, "base_url": "http://localhost:11434/v1"}

    state = AgentState.from_wire(wrap_agent(agent.to_dict(), [], [], provider=provider))

    assert state is not None
    assert state.provider.kind == ProviderKind.OLLAMA


def test_legacy_direct_driver_maps_to_its_openai_compatible_endpoint() -> None:
    agent = GtAgent(prompt_driver=OpenAiChatPromptDriver(model="gpt-4.1", api_key="k"))

    state = AgentState.from_wire(agent.to_dict())

    assert state is not None
    assert state.provider.kind == ProviderKind.OPENAI_COMPATIBLE
    assert state.provider.base_url == "https://api.openai.com/v1"
    assert state.provider.api_key_secret == "OPENAI_API_KEY"
