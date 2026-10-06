from __future__ import annotations

import pytest

from griptape_nodes_library.agents.agent import Agent
from griptape_nodes_library.llm.model_config import ModelConfig, ModelProvider

_LICENSE = "header.payload.signature"
"""A Griptape Nodes License is a JWT: three dot-separated segments."""


def _stub_secrets(monkeypatch: pytest.MonkeyPatch, secrets: dict[str, str]) -> None:
    """Make the engine's SecretsManager resolve only the names in *secrets*.

    Stubbed at the SecretsManager rather than at ``resolve_cloud_api_key`` so the
    License-before-API-key precedence is exercised for real, and so a test can model
    a license-only user by simply omitting ``GT_CLOUD_API_KEY``.
    """
    import griptape_nodes_library.utils.cloud_credential_utils as cloud_credential_utils

    class _FakeSecrets:
        def get_secret(self, name: str, **_kwargs: object) -> str | None:
            return secrets.get(name)

    monkeypatch.setattr(cloud_credential_utils.GriptapeNodes, "SecretsManager", lambda: _FakeSecrets())


def _stub_params(agent_node: Agent, monkeypatch: pytest.MonkeyPatch, *, model: object, agent: object) -> None:
    """Override get_parameter_value for the model/agent params, passing others through."""
    original = agent_node.get_parameter_value

    def _get(name: str) -> object:
        if name == "model":
            return model
        if name == "agent":
            return agent
        return original(name)

    monkeypatch.setattr(agent_node, "get_parameter_value", _get)


def _connected_model_config() -> ModelConfig:
    """A ModelConfig standing in for a connected Prompt Model Config node."""
    return ModelConfig(provider=ModelProvider.ANTHROPIC, model="claude-fake")


def test_validation_fails_when_no_credential_and_default_model_used(
    agent_node: Agent, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_secrets(monkeypatch, {})
    _stub_params(agent_node, monkeypatch, model="claude-sonnet-4-6", agent=None)

    exceptions = agent_node.validate_before_workflow_run()

    assert exceptions is not None
    assert len(exceptions) == 1
    # Names both credentials, so a license-only user is not sent after an API key.
    message = str(exceptions[0])
    assert "license" in message.lower()
    assert "GT_CLOUD_API_KEY" in message


def test_validation_passes_when_cloud_key_present_and_default_model_used(
    agent_node: Agent, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_secrets(monkeypatch, {"GT_CLOUD_API_KEY": "gt-cloud-key"})
    _stub_params(agent_node, monkeypatch, model="claude-sonnet-4-6", agent=None)

    assert agent_node.validate_before_workflow_run() is None


def test_validation_passes_with_license_and_no_api_key(agent_node: Agent, monkeypatch: pytest.MonkeyPatch) -> None:
    # A license-only user has no GT_CLOUD_API_KEY at all. Griptape Cloud's chat
    # endpoints authenticate a License, so the Agent must run. Regression test for
    # the "'GT_CLOUD_API_KEY is not defined'" ResolveNode failure.
    _stub_secrets(monkeypatch, {"GRIPTAPE_NODES_LICENSE": _LICENSE})
    _stub_params(agent_node, monkeypatch, model="claude-sonnet-4-6", agent=None)

    assert agent_node.validate_before_workflow_run() is None


def test_validation_skips_cloud_key_when_model_config_connected(
    agent_node: Agent, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A connected Prompt Model Config (e.g. Anthropic) carries its own credentials,
    # so the Griptape Cloud key must not be required. Regression test for issue #71.
    _stub_secrets(monkeypatch, {})
    _stub_params(agent_node, monkeypatch, model=_connected_model_config(), agent=None)

    assert agent_node.validate_before_workflow_run() is None


def test_validation_skips_cloud_key_when_agent_connected(agent_node: Agent, monkeypatch: pytest.MonkeyPatch) -> None:
    # A connected agent carries its own model, so the cloud key is not required.
    _stub_secrets(monkeypatch, {})
    _stub_params(agent_node, monkeypatch, model="claude-sonnet-4-6", agent={"type": "Agent"})

    assert agent_node.validate_before_workflow_run() is None
