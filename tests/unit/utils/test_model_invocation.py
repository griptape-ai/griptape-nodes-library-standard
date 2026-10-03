"""Contracts of the vendored model-invocation gate.

`require_model_invocation_sync` fails closed. `require_model_access_sync` adds the
budget check for direct provider calls, which fails open unless Cloud denies.

The helper is the raising half of the declaration path: every node that must
abort when the permission layer denies a model goes through it, so the two ways
it can refuse (an unidentified model, and an outright denial) are worth pinning
down independently of any one node.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from griptape_nodes.exe_types.node_types import BaseNode
from griptape_nodes.retained_mode.events.budget_events import (
    BudgetAccessRequest,
    BudgetAccessResultFailure,
    BudgetAccessResultSuccess,
    ReportUsageRequest,
    ReportUsageResultSuccess,
)
from griptape_nodes.retained_mode.events.model_events import (
    DeclareModelInvocationRequest,
    DeclareModelInvocationResultFailure,
    DeclareModelInvocationResultSuccess,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.utils.budget_refusal import BudgetExceededError

import griptape_nodes_library.utils.model_invocation as model_invocation_module
from griptape_nodes_library.utils.model_invocation import (
    report_model_usage_sync,
    require_model_access_sync,
    require_model_invocation_sync,
)


class _StubNode(BaseNode):
    """Concrete `BaseNode` so the helper's `type(node).__name__` / `node.name` are real."""

    def process(self) -> None:
        return


@pytest.fixture
def node() -> _StubNode:
    return _StubNode(name="StubNode")


@pytest.fixture
def declared(monkeypatch: pytest.MonkeyPatch) -> list[DeclareModelInvocationRequest]:
    """Record every declaration that reaches the engine, passing it through."""
    seen: list[DeclareModelInvocationRequest] = []
    real_handle_request = GriptapeNodes.handle_request

    def _recording_handle_request(request: Any) -> Any:
        if isinstance(request, DeclareModelInvocationRequest):
            seen.append(request)
        return real_handle_request(request)

    monkeypatch.setattr(GriptapeNodes, "handle_request", _recording_handle_request)
    return seen


def test_permitted_model_declares_and_returns(node: _StubNode, declared: list[DeclareModelInvocationRequest]) -> None:
    """No policy denies the call in this environment, so the engine clears it."""
    require_model_invocation_sync(node, "gpt-4o")

    assert [(d.model_id, d.node_name) for d in declared] == [("gpt-4o", "StubNode")]


def test_denied_model_raises_with_the_engines_reason(node: _StubNode, monkeypatch: pytest.MonkeyPatch) -> None:
    """`result_details` explains *why* the policy refused, so it must survive."""
    monkeypatch.setattr(
        GriptapeNodes,
        "handle_request",
        lambda _request: DeclareModelInvocationResultFailure(result_details="seat limit reached"),
    )

    with pytest.raises(RuntimeError, match=r"Cannot run _StubNode 'StubNode': seat limit reached"):
        require_model_invocation_sync(node, "gpt-4o")


def test_denied_model_falls_back_to_naming_the_model(node: _StubNode, monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty explanation must not produce a RuntimeError that names nothing."""
    monkeypatch.setattr(
        GriptapeNodes,
        "handle_request",
        lambda _request: DeclareModelInvocationResultFailure(result_details="   "),
    )

    with pytest.raises(RuntimeError, match=r"invocation of model 'gpt-4o' was not permitted"):
        require_model_invocation_sync(node, "gpt-4o")


@pytest.mark.parametrize("api_model_id", [None, "", "   "])
def test_unidentified_model_is_refused_without_declaring(
    node: _StubNode,
    declared: list[DeclareModelInvocationRequest],
    api_model_id: str | None,
) -> None:
    """A driver may leave `model` unset and let the provider choose --
    `GriptapeCloudPromptDriver.model` defaults to None. There is nothing to gate
    in that case, so the helper refuses rather than asking the permission layer
    to rule on a model nobody has named.
    """
    with pytest.raises(RuntimeError, match="no model was identified"):
        require_model_invocation_sync(node, api_model_id)

    assert declared == []


def test_purpose_distinguishes_two_gates_in_one_node(node: _StubNode, monkeypatch: pytest.MonkeyPatch) -> None:
    """A node that gates more than one invocation (e.g. GenerateImage gates prompt
    enhancement separately from image generation) needs the denial to say which.
    """
    monkeypatch.setattr(
        GriptapeNodes,
        "handle_request",
        lambda _request: DeclareModelInvocationResultFailure(result_details="seat limit reached"),
    )

    with pytest.raises(RuntimeError, match=r"_StubNode 'StubNode' \(prompt enhancement\): seat limit reached"):
        require_model_invocation_sync(node, "gpt-4o", purpose="prompt enhancement")


def test_purpose_is_omitted_when_not_given(node: _StubNode, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        GriptapeNodes,
        "handle_request",
        lambda _request: DeclareModelInvocationResultFailure(result_details="seat limit reached"),
    )

    with pytest.raises(RuntimeError, match=r"Cannot run _StubNode 'StubNode': seat limit reached"):
        require_model_invocation_sync(node, "gpt-4o")


def test_unresolvable_model_id_still_declares(
    node: _StubNode, declared: list[DeclareModelInvocationRequest], caplog: pytest.LogCaptureFixture
) -> None:
    """A model that is not one of the node's declared catalog models is declared
    under its raw provider id rather than being dropped, so an unregistered node
    still fails closed against policy instead of going ungated.
    """
    with caplog.at_level("WARNING", logger="griptape_nodes"):
        require_model_invocation_sync(node, "some-unregistered-model")

    assert [d.model_id for d in declared] == ["some-unregistered-model"]
    assert "is not a declared catalog model" in caplog.text


def test_resolve_catalog_model_id_returns_none_for_undeclared_node(node: _StubNode) -> None:
    """A node constructed outside the library path declares no models, so there
    is no provider-id -> catalog-key mapping to resolve against.
    """
    assert model_invocation_module.resolve_catalog_model_id(node, "gpt-4o") is None


# --- require_model_access_sync / report_model_usage_sync -------------------------------------


def _routing(monkeypatch: pytest.MonkeyPatch, budget_result: Any) -> list[Any]:
    """Clear every declaration, answer budget checks with `budget_result`, and record all requests."""
    seen: list[Any] = []

    def _handle(request: Any) -> Any:
        seen.append(request)
        if isinstance(request, DeclareModelInvocationRequest):
            return DeclareModelInvocationResultSuccess(model_id=request.model_id, result_details="ok")
        if isinstance(request, BudgetAccessRequest):
            return budget_result
        return ReportUsageResultSuccess(idempotency_key="k", result_details="queued")

    monkeypatch.setattr(GriptapeNodes, "handle_request", _handle)
    return seen


def _cleared(*, checked: bool = True) -> BudgetAccessResultSuccess:
    return BudgetAccessResultSuccess(correlation_id="corr-1", checked=checked, result_details="ok")


def test_access_declares_then_checks_budget(node: _StubNode, monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _routing(monkeypatch, _cleared())

    correlation_id = require_model_access_sync(node, "gpt-4o", estimated_cost_micro_usd=40_000)

    assert correlation_id == "corr-1"
    assert [type(r) for r in seen] == [DeclareModelInvocationRequest, BudgetAccessRequest]
    check = seen[1]
    assert (check.model_id, check.estimated_cost_micro_usd, check.node_type) == ("gpt-4o", 40_000, "_StubNode")
    # The node label is user-authored and must never be sent.
    assert check.node_id is None


def test_access_denied_by_permission_never_reaches_budget(node: _StubNode, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[Any] = []

    def _handle(request: Any) -> Any:
        seen.append(request)
        return DeclareModelInvocationResultFailure(result_details="seat limit reached")

    monkeypatch.setattr(GriptapeNodes, "handle_request", _handle)

    with pytest.raises(RuntimeError, match="seat limit reached"):
        require_model_access_sync(node, "gpt-4o")

    assert not any(isinstance(r, BudgetAccessRequest) for r in seen)


def test_access_raises_the_engines_budget_exception_unwrapped(node: _StubNode, monkeypatch: pytest.MonkeyPatch) -> None:
    exc = BudgetExceededError("Budget stopped this run. Studio budget is spent.", MagicMock())
    _routing(
        monkeypatch,
        BudgetAccessResultFailure(blocked_by=[], correlation_id="c", result_details=str(exc), exception=exc),
    )

    with pytest.raises(BudgetExceededError) as raised:
        require_model_access_sync(node, "gpt-4o")

    assert raised.value is exc


def test_access_proceeds_when_budget_was_not_checked(node: _StubNode, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail open: Cloud could not be asked, so the call goes ahead."""
    _routing(monkeypatch, _cleared(checked=False))

    assert require_model_access_sync(node, "gpt-4o") == "corr-1"


def test_report_carries_correlation_and_node_type(node: _StubNode, monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _routing(monkeypatch, _cleared())

    report_model_usage_sync(
        node, declared_cost_micro_usd=7500, provider="openai", model="gpt-4o", correlation_id="corr-1"
    )

    (report,) = seen
    assert isinstance(report, ReportUsageRequest)
    assert (report.declared_cost_micro_usd, report.provider, report.model) == (7500, "openai", "gpt-4o")
    assert (report.correlation_id, report.node_type, report.node_id) == ("corr-1", "_StubNode", None)
    assert report.activity_type == "chat_completion"


def test_report_never_raises(node: _StubNode, monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(_request: Any) -> Any:
        raise RuntimeError("engine down")

    monkeypatch.setattr(GriptapeNodes, "handle_request", _boom)

    report_model_usage_sync(node, declared_cost_micro_usd=1, provider=None, model=None, correlation_id=None)
