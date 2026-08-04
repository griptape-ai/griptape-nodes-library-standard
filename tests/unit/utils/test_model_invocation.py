"""Fail-closed contract of `require_model_invocation_sync`.

The helper is the raising half of the declaration path: every node that must
abort when the permission layer denies a model goes through it, so the two ways
it can refuse (an unidentified model, and an outright denial) are worth pinning
down independently of any one node.
"""

from __future__ import annotations

from typing import Any

import pytest
from griptape_nodes.exe_types.node_types import BaseNode
from griptape_nodes.retained_mode.events.model_events import (
    DeclareModelInvocationRequest,
    DeclareModelInvocationResultFailure,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

import griptape_nodes_library.utils.model_invocation as model_invocation_module
from griptape_nodes_library.utils.model_invocation import require_model_invocation_sync


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
