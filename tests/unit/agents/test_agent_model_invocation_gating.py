"""Tests that ``Agent.process`` declares the model invocation before running the
model (issue #431).

``Agent`` runs models through griptape framework prompt drivers without ever
declaring the call to the engine's permission layer. ``process`` now declares
the invocation right before the network call, once the driver's model is
settled, and fails closed (raises) when the declaration is denied.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
from griptape_nodes.exe_types.node_types import BaseNode
from griptape_nodes.node_library.library_registry import LibraryRegistry

import griptape_nodes_library.agents.agent as agent_module
import griptape_nodes_library.utils.model_invocation as model_invocation_module
from griptape_nodes_library.agents.agent import Agent

LIBRARY_NAME = "Griptape Nodes Library"


def _create_node(node_type: str) -> BaseNode:
    """Create a node through the library so its metadata carries `library` / `node_type`.

    `_get_selected_model_id` / `resolve_catalog_model_id` read those two metadata
    keys to resolve a node's declared models; a bare `NodeClass(name=...)`
    construction (as some other Agent unit tests use) does not set them and would
    leave every model-id resolution in `process()` returning `None`.
    """
    library = LibraryRegistry.get_library(name=LIBRARY_NAME)
    return library.create_node(node_type=node_type, name=node_type)


class _FakeDeclaration:
    """Stand-in for the `ResultPayload` returned by `declare_model_invocation_sync`."""

    def __init__(self, *, ok: bool, details: str = "") -> None:
        self._ok = ok
        self.result_details = details

    def failed(self) -> bool:
        return not self._ok


def _stub_secret(monkeypatch: pytest.MonkeyPatch, value: str | None) -> None:
    class _FakeSecrets:
        # Accepts **kwargs because the Cloud credential resolver passes
        # should_error_on_not_found= when probing for the License.
        def get_secret(self, _name: str, **_kwargs: object) -> str | None:
            return value

    monkeypatch.setattr(agent_module.GriptapeNodes, "SecretsManager", lambda: _FakeSecrets())


@pytest.fixture
def agent_node(monkeypatch: pytest.MonkeyPatch) -> Agent:
    _stub_secret(monkeypatch, "gt-cloud-key")
    node = cast(Agent, _create_node("Agent"))
    # A current, non-deprecated model that is not the node's default, so the
    # assertions below distinguish a selected value from the fallback default.
    node.set_parameter_value("model", "claude-opus-5")
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

    monkeypatch.setattr(model_invocation_module, "declare_model_invocation_sync", _fake_declare)

    gen = agent_node.process()
    runner = next(gen)

    assert captured["api_model_id"] == "claude-opus-5"
    assert captured["node"] is agent_node
    assert callable(runner)


def test_raises_before_running_when_declaration_is_denied(agent_node: Agent, monkeypatch: pytest.MonkeyPatch) -> None:
    ran = {"called": False}

    def _fake_declare(_node: Agent, _api_model_id: str) -> _FakeDeclaration:
        return _FakeDeclaration(ok=False, details="denied by policy")

    def _fake_process(self: Agent, agent: Any, prompt: Any) -> None:  # pragma: no cover - must not run
        ran["called"] = True

    monkeypatch.setattr(model_invocation_module, "declare_model_invocation_sync", _fake_declare)
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

    monkeypatch.setattr(model_invocation_module, "declare_model_invocation_sync", _fake_declare)

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
    assert agent_node.get_parameter_value("model") == "claude-opus-5"

    captured: dict[str, Any] = {}

    def _fake_declare(node: Agent, api_model_id: str) -> _FakeDeclaration:
        captured["api_model_id"] = api_model_id
        return _FakeDeclaration(ok=True)

    monkeypatch.setattr(model_invocation_module, "declare_model_invocation_sync", _fake_declare)

    gen = agent_node.process()
    next(gen)

    assert captured["api_model_id"] == "gpt-4.1"


def _record_budget_requests(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Clear permission, answer the budget check, and record every budget request."""
    from griptape_nodes.retained_mode.events.budget_events import (
        BudgetAccessRequest,
        BudgetAccessResultSuccess,
        ReportUsageRequest,
        ReportUsageResultSuccess,
    )

    seen: list[Any] = []
    real_handle_request = model_invocation_module.GriptapeNodes.handle_request

    def _handle(request: Any) -> Any:
        if isinstance(request, BudgetAccessRequest):
            seen.append(request)
            return BudgetAccessResultSuccess(correlation_id="corr-1", checked=True, result_details="ok")
        if isinstance(request, ReportUsageRequest):
            seen.append(request)
            return ReportUsageResultSuccess(idempotency_key="k", result_details="queued")
        return real_handle_request(request)

    monkeypatch.setattr(model_invocation_module, "declare_model_invocation_sync", lambda *_: _FakeDeclaration(ok=True))
    monkeypatch.setattr(model_invocation_module.GriptapeNodes, "handle_request", _handle)
    return seen


def test_griptape_cloud_agent_is_not_budget_checked(agent_node: Agent, monkeypatch: pytest.MonkeyPatch) -> None:
    """The proxy enforces budgets server-side; a client check would count the call twice."""
    seen = _record_budget_requests(monkeypatch)

    next(agent_node.process())

    assert seen == []


def test_connected_direct_driver_is_budget_checked_and_reported(
    agent_node: Agent, monkeypatch: pytest.MonkeyPatch
) -> None:
    from griptape.drivers.prompt.openai import OpenAiChatPromptDriver
    from griptape.events import EventBus, FinishPromptEvent
    from griptape_nodes.retained_mode.events.budget_events import BudgetAccessRequest, ReportUsageRequest

    seen = _record_budget_requests(monkeypatch)
    # A connected Prompt Model Config supplies the driver; the dropdown's Options trait would
    # reject a driver set directly, so stand in for the connection.
    driver = OpenAiChatPromptDriver(model="gpt-4o", api_key="k")
    node_cls = type(agent_node)  # the registry loads its own copy of the class
    real_get = node_cls.get_parameter_value
    monkeypatch.setattr(
        node_cls, "get_parameter_value", lambda self, name: driver if name == "model" else real_get(self, name)
    )

    def _fake_process(self: Agent, agent: Any, prompt: Any) -> Any:
        EventBus.publish_event(
            FinishPromptEvent(model="gpt-4o", result="hi", input_token_count=1000, output_token_count=500)
        )
        return agent

    monkeypatch.setattr(node_cls, "_process", _fake_process)

    gen = agent_node.process()
    runner = next(gen)
    runner()

    check, report = seen
    assert isinstance(check, BudgetAccessRequest)
    # The check is keyed on the catalog id; the node label is never sent.
    assert (check.model_id, check.node_type, check.node_id) == ("gtc_gpt_4o", "Agent", None)
    assert isinstance(report, ReportUsageRequest)
    assert (report.declared_cost_micro_usd, report.correlation_id) == (7500, "corr-1")
