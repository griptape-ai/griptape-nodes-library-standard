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
from griptape_nodes.retained_mode.events import converters
from griptape_nodes.retained_mode.events.budget_events import (
    BudgetAccessRequest,
    BudgetAccessResultFailure,
    BudgetAccessResultSuccess,
)
from griptape_nodes.retained_mode.events.generic_events import GenericResultFailure
from griptape_nodes.retained_mode.events.model_events import (
    DeclareModelInvocationRequest,
    DeclareModelInvocationResultFailure,
    DeclareModelInvocationResultSuccess,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.retained_mode.managers import budget_manager as budget_manager_module
from griptape_nodes.utils.budget_refusal import BudgetExceededError, refusal_from_check
from griptape_nodes.utils.budget_refusal import describe as describe_budget_refusal

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

    with pytest.raises(RuntimeError, match=r"^Cannot run the model: seat limit reached$"):
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

    with pytest.raises(RuntimeError, match=r"^Cannot run the model \(prompt enhancement\): seat limit reached$"):
        require_model_invocation_sync(node, "gpt-4o", purpose="prompt enhancement")


def test_purpose_is_omitted_when_not_given(node: _StubNode, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        GriptapeNodes,
        "handle_request",
        lambda _request: DeclareModelInvocationResultFailure(result_details="seat limit reached"),
    )

    with pytest.raises(RuntimeError, match=r"^Cannot run the model: seat limit reached$"):
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


# --- require_model_access_sync ------------------------------------------------------------


def _routing(monkeypatch: pytest.MonkeyPatch, budget_result: Any) -> list[Any]:
    """Clear every declaration, answer budget checks with `budget_result`, and record all requests."""
    seen: list[Any] = []

    def _handle(request: Any) -> Any:
        seen.append(request)
        if isinstance(request, DeclareModelInvocationRequest):
            return DeclareModelInvocationResultSuccess(model_id=request.model_id, result_details="ok")
        return budget_result

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
    assert (check.model, check.estimated_cost_micro_usd, check.node_type) == ("gpt-4o", 40_000, "_StubNode")
    # The node label is user-authored and must never be sent.
    assert check.node_id is None


def test_access_declares_the_catalog_key_but_checks_the_provider_name(
    node: _StubNode, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Permission gates on the catalog key; the budget check uses the provider's name, as the report does."""
    monkeypatch.setattr(model_invocation_module, "resolve_catalog_model_id", lambda _node, _api_model_id: "gtc_gpt_4o")
    seen = _routing(monkeypatch, _cleared())

    require_model_access_sync(node, "gpt-4o")

    declaration, check = seen
    assert declaration.model_id == "gtc_gpt_4o"
    assert check.model == "gpt-4o"


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


def test_access_stops_on_a_flattened_refusal_carrying_the_exception(
    node: _StubNode, monkeypatch: pytest.MonkeyPatch
) -> None:
    """From a worker process a refusal crosses back as a generic failure; it must still stop the call."""
    exc = BudgetExceededError("Budget stopped this run. Studio budget is spent.", MagicMock())
    _routing(monkeypatch, GenericResultFailure(result_details=str(exc), exception=exc))

    with pytest.raises(BudgetExceededError) as raised:
        require_model_access_sync(node, "gpt-4o")

    assert raised.value is exc


def _forwarded(failure: BudgetAccessResultFailure) -> Any:
    """The failure as a node in a worker process receives it: through the engine's converter."""
    return converters.engine.structure(converters.engine.unstructure(failure), BudgetAccessResultFailure)


# One `blocked_by` entry as Griptape Cloud's POST /api/budget-checks returns it.
STUDIO_MONTHLY = {
    "budget_id": "b1",
    "budget_name": "Studio monthly",
    "scope_type": "TAG",
    "reset_period": "MONTHLY",
    "enforcement": "HARD",
    "limit_credits": 1000,
    "spent_credits": 1200,
    "remaining_credits": -200,
}


def test_a_refusal_forwarded_from_a_worker_still_names_its_budget(
    node: _StubNode, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Across the worker boundary the engine's exception becomes a ForwardedException naming only its
    type. It must come back as a BudgetExceededError that the engine's halt path words with the budget's
    name, as it would for a refusal in the same process.
    """
    refusal = refusal_from_check({"allowed": False, "blocked_by": [STUDIO_MONTHLY]})
    exc = BudgetExceededError(describe_budget_refusal(refusal), refusal)
    forwarded = _forwarded(
        BudgetAccessResultFailure(
            blocked_by=[STUDIO_MONTHLY], correlation_id="c", result_details=str(exc), exception=exc
        )
    )
    assert type(forwarded.exception).__name__ == "ForwardedException"
    _routing(monkeypatch, forwarded)

    with pytest.raises(BudgetExceededError) as raised:
        require_model_access_sync(node, "gpt-4o")

    halt = GriptapeNodes.NodeManager()._budget_halt_for(raised.value, "Agent_1")
    assert halt is not None
    assert 'blocked by the budget "Studio monthly"' in str(halt)


def test_a_forwarded_refusal_without_budgets_keeps_its_message(
    node: _StubNode, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no budgets to rebuild from, the already-worded message is kept rather than reworded from nothing."""
    exc = BudgetExceededError("Budget stopped this run. Studio budget is spent.", MagicMock())
    forwarded = _forwarded(
        BudgetAccessResultFailure(blocked_by=[], correlation_id="c", result_details=str(exc), exception=exc)
    )
    forwarded.blocked_by = None
    _routing(monkeypatch, forwarded)

    with pytest.raises(BudgetExceededError, match=r"Budget stopped this run\. Studio budget is spent\."):
        require_model_access_sync(node, "gpt-4o")


def test_access_stops_on_a_failure_without_an_exception(node: _StubNode, monkeypatch: pytest.MonkeyPatch) -> None:
    _routing(monkeypatch, GenericResultFailure(result_details="Budget stopped this run. Studio budget is spent."))

    with pytest.raises(RuntimeError, match=r"Cannot run _StubNode: Budget stopped this run\. Studio budget"):
        require_model_access_sync(node, "gpt-4o")


def test_access_proceeds_when_budget_was_not_checked(node: _StubNode, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail open: Cloud could not be asked, so the call goes ahead."""
    _routing(monkeypatch, _cleared(checked=False))

    assert require_model_access_sync(node, "gpt-4o") == "corr-1"


class _RecordedReport:
    """Stands in for `ReportUsageRequest`, so the test pins what the helper sends."""

    def __init__(self, **fields: Any) -> None:
        self.fields = fields


def test_report_sends_token_counts_for_cloud_to_price(node: _StubNode, monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list[_RecordedReport] = []
    monkeypatch.setattr(model_invocation_module, "ReportUsageRequest", _RecordedReport)
    monkeypatch.setattr(GriptapeNodes, "handle_request", sent.append)

    report_model_usage_sync(
        node,
        provider="openai",
        model="gpt-4o",
        input_tokens=1000,
        output_tokens=500,
        cache_read_tokens=900,
        correlation_id="corr-1",
    )

    (report,) = sent
    # No dollar figure: Cloud prices the tokens. The node label is never sent.
    assert report.fields == {
        "provider": "openai",
        "model": "gpt-4o",
        "activity_type": "chat_completion",
        "input_tokens": 1000,
        "output_tokens": 500,
        "cache_read_tokens": 900,
        "cache_write_tokens": 0,
        "node_type": "_StubNode",
        "correlation_id": "corr-1",
    }


def test_report_never_raises(node: _StubNode, monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(_request: Any) -> Any:
        raise RuntimeError("engine down")

    monkeypatch.setattr(model_invocation_module, "ReportUsageRequest", _RecordedReport)
    monkeypatch.setattr(GriptapeNodes, "handle_request", _boom)

    report_model_usage_sync(node, provider="openai", model="gpt-4o", input_tokens=1, output_tokens=1)


def test_report_reaches_cloud_as_a_usage_body(node: _StubNode, monkeypatch: pytest.MonkeyPatch) -> None:
    """Through the engine's real handler: the body bound for POST /api/spend carries usage, not dollars."""
    posted: list[dict[str, Any]] = []
    monkeypatch.setattr(budget_manager_module, "resolve_cloud_credential", lambda _secrets: "gt-key")
    monkeypatch.setattr(budget_manager_module.BudgetManager, "_send_report", lambda _self, body: posted.append(body))

    class _NowExecutor:
        def submit(self, fn: Any, *args: Any) -> None:
            fn(*args)

    monkeypatch.setattr(GriptapeNodes.BudgetManager(), "_report_executor", _NowExecutor())

    report_model_usage_sync(
        node,
        provider="openai",
        model="gpt-4o",
        input_tokens=1000,
        output_tokens=500,
        cache_read_tokens=900,
        correlation_id="corr-1",
    )

    (body,) = posted
    assert body["usage"] == {
        "input_tokens": 1000,
        "output_tokens": 500,
        "cache_read_tokens": 900,
        "cache_write_tokens": 0,
    }
    assert "declared_cost_micro_usd" not in body
    assert (body["provider"], body["model"], body["correlation_id"]) == ("openai", "gpt-4o", "corr-1")
    assert body["tags"]["node_type"] == "_StubNode"
