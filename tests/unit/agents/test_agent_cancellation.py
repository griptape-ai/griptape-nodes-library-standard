"""Cancelling an Agent node must surface as a cancellation, end to end through `process()`.

`process()` yields `_process`, which returns the agent early when cancellation is
requested — before the agent has produced any output. Control then returns to `process()`,
which used to read `agent.output` unconditionally and so raised `ValueError: Structure's
output Task has no output`. That is the error users saw after cancelling a run, and it
points at the structure rather than at the cancel.

These drive the real generator, so they pin the behaviour at the call site rather than
just the helper it delegates to.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from griptape.structures import Agent as GtAgent

if TYPE_CHECKING:
    from griptape_nodes_library.agents.agent import Agent

_LICENSE = "header.payload.signature"
"""A Griptape Nodes License is a JWT: three dot-separated segments."""


@pytest.fixture(autouse=True)
def _cloud_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    """Satisfy the credential lookup the default prompt driver performs on construction."""
    import griptape_nodes_library.utils.cloud_credential_utils as cloud_credential_utils

    class _FakeSecrets:
        def get_secret(self, name: str, **_kwargs: object) -> str | None:
            return {"GRIPTAPE_NODES_LICENSE": _LICENSE}.get(name)

    monkeypatch.setattr(cloud_credential_utils.GriptapeNodes, "SecretsManager", lambda: _FakeSecrets())


def _run_process(node: Agent) -> Any:
    """Drive `process()` the way the engine does, resolving each yielded thunk in turn."""
    generator = node.process()
    try:
        thunk = next(generator)
        while True:
            thunk = generator.send(thunk())
    except StopIteration as stop:
        return stop.value


def _stub_agent_run(node: Agent, monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the network call with the shape a cancelled stream leaves behind.

    `_process` returns the agent without its output task having run, which is precisely
    what makes the unguarded `agent.output` read raise.
    """

    def _cancelled_process(agent: GtAgent, _prompt: object) -> GtAgent:
        return agent

    monkeypatch.setattr(node, "_process", _cancelled_process)


def _stub_model_gating(node: Agent, monkeypatch: pytest.MonkeyPatch) -> None:
    """Neither permission gate is under test here; both would otherwise reach the engine."""
    monkeypatch.setattr(node._model_access, "raise_if_selection_denied", lambda: None)
    monkeypatch.setattr(
        "griptape_nodes_library.agents.agent.require_model_invocation_sync",
        lambda *_args, **_kwargs: None,
    )


class TestCancelledAgentRunReportsCancellation:
    def test_cancelled_run_raises_a_cancellation_not_a_missing_output_error(
        self, agent_node: Agent, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_model_gating(agent_node, monkeypatch)
        _stub_agent_run(agent_node, monkeypatch)
        agent_node.set_parameter_value("prompt", "tell a long story")
        agent_node.request_cancellation()

        with pytest.raises(RuntimeError, match="was cancelled before it produced a result"):
            _run_process(agent_node)

    def test_cancelled_run_does_not_report_the_structure_output_error(
        self, agent_node: Agent, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The specific misleading message must be gone, not merely replaced by another error."""
        _stub_model_gating(agent_node, monkeypatch)
        _stub_agent_run(agent_node, monkeypatch)
        agent_node.set_parameter_value("prompt", "tell a long story")
        agent_node.request_cancellation()

        with pytest.raises(RuntimeError) as excinfo:
            _run_process(agent_node)

        assert "has no output" not in str(excinfo.value)

    def test_a_run_whose_stale_flag_was_cleared_is_not_treated_as_cancelled(
        self, agent_node: Agent, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The engine clears the flag at dispatch, so a re-run after a cancel must proceed.

        It still fails on the unproduced output here, because `_process` is stubbed to
        return an agent that never ran — the point is that it gets *past* the cancel guard
        rather than short-circuiting on a flag left over from the previous run.
        """
        _stub_model_gating(agent_node, monkeypatch)
        _stub_agent_run(agent_node, monkeypatch)
        agent_node.set_parameter_value("prompt", "tell a long story")
        agent_node.request_cancellation()
        agent_node.clear_cancellation()

        with pytest.raises(ValueError, match="has no output"):
            _run_process(agent_node)
