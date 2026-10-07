"""Tests that griptape-based nodes still read an agent passed as ``AgentState``."""

from __future__ import annotations

from typing import Any

import pytest
from griptape.drivers.prompt.griptape_cloud import GriptapeCloudPromptDriver
from griptape.drivers.prompt.ollama import OllamaPromptDriver
from griptape.drivers.prompt.openai import OpenAiChatPromptDriver
from griptape.tasks import PromptTask
from pydantic_ai.messages import ImageUrl, ModelRequest, ModelResponse, TextPart, UserPromptPart

import griptape_nodes_library.utils.agent_state as agent_state_module
import griptape_nodes_library.utils.cloud_credential_utils as cloud_credential_utils
from griptape_nodes_library.agents.griptape_nodes_agent import GriptapeNodesAgent
from griptape_nodes_library.utils.agent_state import AgentState, ProviderKind, ProviderRef, direct_provider_ref
from griptape_nodes_library.utils.agent_utils import legacy_wrapper_for, restore_provider_driver, unwrap_agent


class _FakeSecrets:
    def __init__(self, values: dict[str, str]) -> None:
        self._values = values

    def get_secret(self, name: str, **_kwargs: Any) -> str | None:
        return self._values.get(name)


@pytest.fixture(autouse=True)
def _secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    secrets = _FakeSecrets({"GT_CLOUD_API_KEY": "gt-key", "LOCAL_KEY": "local-key"})
    monkeypatch.setattr(agent_state_module.GriptapeNodes, "SecretsManager", lambda: secrets)
    monkeypatch.setattr(cloud_credential_utils.GriptapeNodes, "SecretsManager", lambda: secrets)


def _state(provider: ProviderRef) -> AgentState:
    return AgentState(
        provider=provider,
        model="gpt-4.1",
        tools=[{"tool_type": "Calculator"}],
        rulesets=[{"name": "behavior_1", "rules": ["Be brief"]}],
        messages=[
            ModelRequest(parts=[UserPromptPart(content=["What is this?", ImageUrl(url="http://h/a.png")])]),
            ModelResponse(parts=[TextPart(content="A cat.")]),
        ],
    )


def _restore(wire: dict) -> tuple[GriptapeNodesAgent, list, list]:
    agent_dict, tools, rulesets = unwrap_agent(wire)
    agent = GriptapeNodesAgent().from_dict(agent_dict)
    restore_provider_driver(agent, wire)
    return agent, tools, rulesets


def test_cloud_state_unwraps_to_a_cloud_agent_with_text_memory() -> None:
    agent, tools, rulesets = _restore(_state(ProviderRef()).to_wire())

    task = agent.tasks[0]
    assert isinstance(task, PromptTask)
    assert isinstance(task.prompt_driver, GriptapeCloudPromptDriver)
    assert task.prompt_driver.model == "gpt-4.1"
    assert task.prompt_driver.api_key == "gt-key"
    assert tools == [{"tool_type": "Calculator"}]
    assert rulesets == [{"name": "behavior_1", "rules": ["Be brief"]}]
    assert agent.conversation_memory is not None
    run = agent.conversation_memory.runs[0]
    assert (run.input.to_text(), run.output.to_text()) == ("What is this?\n[image]", "A cat.")


def test_openai_compatible_state_restores_its_driver_and_key() -> None:
    provider = ProviderRef(
        kind=ProviderKind.OPENAI_COMPATIBLE, name="local", base_url="http://h/v1", api_key_secret="LOCAL_KEY"
    )

    agent, _, _ = _restore(_state(provider).to_wire())

    driver = agent.tasks[0].prompt_driver  # pyright: ignore[reportAttributeAccessIssue]
    assert isinstance(driver, OpenAiChatPromptDriver)
    assert driver.base_url == "http://h/v1"
    assert driver.api_key == "local-key"


def test_ollama_state_restores_the_native_driver() -> None:
    provider = ProviderRef(kind=ProviderKind.OLLAMA, name="ollama", base_url="http://localhost:11434/v1")

    agent, _, _ = _restore(_state(provider).to_wire())

    driver = agent.tasks[0].prompt_driver  # pyright: ignore[reportAttributeAccessIssue]
    assert isinstance(driver, OllamaPromptDriver)
    assert driver.host == "http://localhost:11434"


@pytest.mark.parametrize(
    "provider",
    [
        ProviderRef(),
        ProviderRef(kind=ProviderKind.OLLAMA, name="ollama", base_url="http://localhost:11434/v1"),
        direct_provider_ref("AnthropicPromptDriver", None),
    ],
)
def test_provider_survives_a_trip_through_a_griptape_node(provider: ProviderRef) -> None:
    """A griptape node re-wraps the legacy wrapper it read; the next Agent node must see the same provider."""
    wrapper = legacy_wrapper_for(_state(provider).to_wire())
    wrapper.get("provider", {}).pop("api_key", None)

    state = AgentState.from_wire(wrapper)

    assert state is not None
    assert state.provider == provider
