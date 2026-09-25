"""Tests that swapping the model dropdown between providers cannot wedge it.

`ProviderSelectionComponent` repoints the shared `model` dropdown at whichever
provider is selected. A provider that reports no models (unreachable, or its
api_key secret unset) must leave the dropdown on the previous provider's models
and report the failure: emptying it is unrecoverable, because
`_update_option_choices` assigns `choices=[]` before rejecting the empty default
it was handed, and the empty list survives save/reload through the serialized
`simple_dropdown`, after which every assignment indexes `choices[0]` on it.
"""

from __future__ import annotations

import pytest
from griptape_nodes.traits.button import Button, ButtonDetailsMessagePayload
from griptape_nodes.traits.options import Options

from griptape_nodes_library.agents.agent import Agent


@pytest.fixture
def agent() -> Agent:
    return Agent(name="Agent")


def _model_choices(node: Agent) -> list[str]:
    model_param = node.get_parameter_by_name("model")
    assert model_param is not None
    return model_param.find_elements_by_type(Options)[0].choices


def _offer_models(node: Agent, models: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the engine round-trip that lists a third-party provider's own models."""
    monkeypatch.setattr(node._provider, "fetch_models_for_provider", lambda _name: models)


def test_provider_models_replace_the_dropdown_choices(agent: Agent, monkeypatch: pytest.MonkeyPatch) -> None:
    _offer_models(agent, ["llama3.2", "qwen3"], monkeypatch)

    assert agent._provider.update_model_choices_for_provider("ollama") is None
    assert _model_choices(agent) == ["llama3.2", "qwen3"]
    assert agent.get_parameter_value("model") == "llama3.2"


def test_provider_reporting_no_models_leaves_the_dropdown_alone(agent: Agent, monkeypatch: pytest.MonkeyPatch) -> None:
    _offer_models(agent, ["llama3.2"], monkeypatch)
    agent._provider.update_model_choices_for_provider("ollama")

    _offer_models(agent, [], monkeypatch)
    failure = agent._provider.update_model_choices_for_provider("lm_studio")

    assert failure is not None
    assert "lm_studio" in failure
    assert _model_choices(agent) == ["llama3.2"]
    assert agent.get_parameter_value("model") == "llama3.2"


def test_refresh_models_button_reports_the_failure(agent: Agent, monkeypatch: pytest.MonkeyPatch) -> None:
    """The button the artist reaches for after fixing the provider says what went wrong."""
    monkeypatch.setattr(agent._provider, "_fetch_provider_names", lambda: ["griptape_cloud", "lm_studio"])
    button = Button(on_click=lambda _button, _details: None)
    details = ButtonDetailsMessagePayload(label="Refresh", variant="secondary", size="icon", state="enabled")
    agent._provider._refresh_providers_button(button, details)
    agent.set_parameter_value("model_provider", "lm_studio")

    _offer_models(agent, [], monkeypatch)
    result = agent._provider._refresh_models_button(button, details)

    assert result is not None
    assert result.success is False
    assert "lm_studio" in str(result.details)


def test_returning_to_griptape_cloud_restores_the_cloud_models(agent: Agent, monkeypatch: pytest.MonkeyPatch) -> None:
    """The cloud branch offers the Griptape Cloud models again, with license decoration."""
    _offer_models(agent, ["llama3.2"], monkeypatch)
    agent._provider.update_model_choices_for_provider("ollama")

    assert agent._provider.update_model_choices_for_provider("griptape_cloud") is None

    model_param = agent.get_parameter_by_name("model")
    assert model_param is not None
    assert model_param.ui_options.get("dropdown_row_icons") is True
    assert [row["name"] for row in model_param.ui_options["data"]] == agent._model_access.model_choices
    assert agent.get_parameter_value("model") in agent._model_access.model_choices


@pytest.mark.xfail(
    strict=True,
    reason="ModelAccessComponent.reinstall_options() adds a second Options trait rather than replacing "
    "the first, so the stale trait -- whose choices _update_option_choices overwrote with the provider's "
    "models -- keeps governing conversion, and the component's legacy keys never return. A saved legacy "
    "value then snaps to choices[0] instead of migrating. Fixed by the engine's dropdown-vocabulary API; "
    "drop this marker when the engine pin bumps.",
)
def test_returning_to_griptape_cloud_restores_the_legacy_vocabulary(
    agent: Agent, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A legacy value still migrates after the dropdown has visited another provider.

    The component accepts its `deprecated_values` keys by carrying them in the
    dropdown's choices, so restoring the cloud vocabulary has to restore them too:
    without them the `Options` trait rewrites a saved legacy value to the first
    choice before the migration converter ever sees it.
    """
    _offer_models(agent, ["llama3.2"], monkeypatch)
    agent._provider.update_model_choices_for_provider("ollama")
    agent._provider.update_model_choices_for_provider("griptape_cloud")

    agent.set_parameter_value("model", "GPT-4o")

    assert agent.get_parameter_value("model") == "gpt-4o"
