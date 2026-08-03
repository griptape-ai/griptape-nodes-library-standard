"""Tests that ``Agent.process`` declares the model invocation before running the
model (issue #431).

``Agent`` runs models through griptape framework prompt drivers without ever
declaring the call to the engine's permission layer. ``process`` now declares
the invocation right before the network call, once the driver's model is
settled, and fails closed (raises) when the declaration is denied.
"""

from __future__ import annotations

from typing import Any

import pytest

import griptape_nodes_library.agents.agent as agent_module
from griptape_nodes_library.agents.agent import Agent


class _FakeDeclaration:
    """Stand-in for the `ResultPayload` returned by `declare_model_invocation_sync`."""

    def __init__(self, *, ok: bool, details: str = "") -> None:
        self._ok = ok
        self.result_details = details

    def failed(self) -> bool:
        return not self._ok


def _stub_secret(monkeypatch: pytest.MonkeyPatch, value: str | None) -> None:
    class _FakeSecrets:
        def get_secret(self, _name: str) -> str | None:
            return value

    monkeypatch.setattr(agent_module.GriptapeNodes, "SecretsManager", lambda: _FakeSecrets())


@pytest.fixture
def agent_node(monkeypatch: pytest.MonkeyPatch) -> Agent:
    _stub_secret(monkeypatch, "gt-cloud-key")
    node = Agent(name="Agent")
    node.set_parameter_value("model", "claude-sonnet-4-6")
    node.set_parameter_value("prompt", "Hello there")
    return node


def test_declares_invocation_with_selected_model_before_running(
    agent_node: Agent, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    def _fake_declare(node: Agent, api_model_id: str) -> _FakeDeclaration:
        captured["node"] = node
        captured["api_model_id"] = api_model_id
        return _FakeDeclaration(ok=True)

    monkeypatch.setattr(agent_module, "declare_model_invocation_sync", _fake_declare)

    gen = agent_node.process()
    runner = next(gen)

    assert captured["api_model_id"] == "claude-sonnet-4-6"
    assert captured["node"] is agent_node
    assert callable(runner)


def test_raises_before_running_when_declaration_is_denied(agent_node: Agent, monkeypatch: pytest.MonkeyPatch) -> None:
    ran = {"called": False}

    def _fake_declare(_node: Agent, _api_model_id: str) -> _FakeDeclaration:
        return _FakeDeclaration(ok=False, details="denied by policy")

    def _fake_process(self: Agent, agent: Any, prompt: Any) -> None:  # pragma: no cover - must not run
        ran["called"] = True

    monkeypatch.setattr(agent_module, "declare_model_invocation_sync", _fake_declare)
    monkeypatch.setattr(Agent, "_process", _fake_process)

    gen = agent_node.process()
    with pytest.raises(RuntimeError, match="denied by policy"):
        next(gen)

    assert ran["called"] is False


def test_falls_back_to_default_message_when_result_details_missing(
    agent_node: Agent, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _fake_declare(_node: Agent, _api_model_id: str) -> _FakeDeclaration:
        return _FakeDeclaration(ok=False, details="")

    monkeypatch.setattr(agent_module, "declare_model_invocation_sync", _fake_declare)

    gen = agent_node.process()
    with pytest.raises(RuntimeError, match="was not permitted"):
        next(gen)


def test_declares_connected_agents_model_over_stale_dropdown_value(
    agent_node: Agent, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A connected Agent supplies the model that actually runs.

    The node's own `model` dropdown keeps its last value when an upstream Agent
    is connected -- the parameter is hidden, not cleared -- so the declaration
    must read the model from the restored agent's task driver, not from the
    stale parameter value.
    """
    from griptape.drivers.prompt.griptape_cloud import GriptapeCloudPromptDriver
    from griptape.structures import Agent as GtStructureAgent

    from griptape_nodes_library.utils.agent_utils import wrap_agent

    # Restoring the wrapper rebuilds a GriptapeCloudPromptDriver, whose api_key
    # default reads this env var.
    monkeypatch.setenv("GT_CLOUD_API_KEY", "fake-key")
    upstream = GtStructureAgent(prompt_driver=GriptapeCloudPromptDriver(model="gpt-4.1", api_key="fake-key"))
    agent_node.set_parameter_value("agent", wrap_agent(upstream.to_dict(), [], []))
    # The dropdown still holds its previous selection (set in the fixture).
    assert agent_node.get_parameter_value("model") == "claude-sonnet-4-6"

    captured: dict[str, Any] = {}

    def _fake_declare(node: Agent, api_model_id: str) -> _FakeDeclaration:
        captured["api_model_id"] = api_model_id
        return _FakeDeclaration(ok=True)

    monkeypatch.setattr(agent_module, "declare_model_invocation_sync", _fake_declare)

    gen = agent_node.process()
    next(gen)

    assert captured["api_model_id"] == "gpt-4.1"
