from __future__ import annotations

import pytest
from griptape.drivers.prompt.base_prompt_driver import BasePromptDriver

from griptape_nodes_library.agents.agent import API_KEY_ENV_VAR, Agent


def _stub_secret(monkeypatch: pytest.MonkeyPatch, value: str | None) -> None:
    """Make the engine's SecretsManager return *value* for any secret lookup."""
    import griptape_nodes_library.agents.agent as agent_module

    class _FakeSecrets:
        def get_secret(self, _name: str) -> str | None:
            return value

    monkeypatch.setattr(agent_module.GriptapeNodes, "SecretsManager", lambda: _FakeSecrets())


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


def _fake_prompt_driver() -> BasePromptDriver:
    """A BasePromptDriver instance standing in for a connected Prompt Model Config."""

    class _FakeDriver(BasePromptDriver):
        def try_run(self, *_args: object, **_kwargs: object) -> object:  # pragma: no cover - never invoked
            raise NotImplementedError

        def try_stream(self, *_args: object, **_kwargs: object) -> object:  # pragma: no cover - never invoked
            raise NotImplementedError

    return _FakeDriver(model="fake-model", tokenizer=None)  # type: ignore[arg-type]


def test_validation_fails_when_cloud_key_missing_and_default_driver_used(
    agent_node: Agent, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_secret(monkeypatch, None)
    _stub_params(agent_node, monkeypatch, model="claude-sonnet-4-6", agent=None)

    exceptions = agent_node.validate_before_workflow_run()

    assert exceptions is not None
    assert len(exceptions) == 1
    assert API_KEY_ENV_VAR in str(exceptions[0])


def test_validation_passes_when_cloud_key_present_and_default_driver_used(
    agent_node: Agent, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_secret(monkeypatch, "gt-cloud-key")
    _stub_params(agent_node, monkeypatch, model="claude-sonnet-4-6", agent=None)

    assert agent_node.validate_before_workflow_run() is None


def test_validation_skips_cloud_key_when_prompt_driver_connected(
    agent_node: Agent, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A connected Prompt Model Config (e.g. Anthropic) carries its own credentials,
    # so the Griptape Cloud key must not be required. Regression test for issue #71.
    _stub_secret(monkeypatch, None)
    _stub_params(agent_node, monkeypatch, model=_fake_prompt_driver(), agent=None)

    assert agent_node.validate_before_workflow_run() is None


def test_validation_skips_cloud_key_when_agent_connected(agent_node: Agent, monkeypatch: pytest.MonkeyPatch) -> None:
    # A connected agent carries its own driver, so the cloud key is not required.
    _stub_secret(monkeypatch, None)
    _stub_params(agent_node, monkeypatch, model="claude-sonnet-4-6", agent={"type": "Agent"})

    assert agent_node.validate_before_workflow_run() is None
