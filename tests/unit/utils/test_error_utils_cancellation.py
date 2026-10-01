"""A cancelled node must report cancellation, not a missing structure output.

`raise_if_cancelled` guards the `try_throw_error(agent.output)` that follows a cancelled
run. Without it, a node that cooperatively cancelled returns with its structure's output
task never having run, and reading `agent.output` raises `ValueError: Structure's output
Task has no output` — a message about the symptom that says nothing about the cancel, and
the one users actually saw after cancelling a run.
"""

from __future__ import annotations

import pytest
from griptape.structures import Agent as GtAgent
from griptape_nodes.exe_types.node_types import BaseNode

from griptape_nodes_library.utils.error_utils import raise_if_cancelled


class _StubNode(BaseNode):
    """Concrete `BaseNode` so the helper reads a real cancellation flag and node name."""

    def process(self) -> None:
        return


@pytest.fixture
def node() -> _StubNode:
    return _StubNode(name="StubNode")


class TestRaiseIfCancelled:
    def test_raises_naming_cancellation_when_the_flag_is_set(self, node: _StubNode) -> None:
        node.request_cancellation()

        with pytest.raises(RuntimeError, match="was cancelled before it produced a result") as excinfo:
            raise_if_cancelled(node)

        assert node.name in str(excinfo.value), "the error should say which node was cancelled"

    def test_passes_through_when_no_cancellation_was_requested(self, node: _StubNode) -> None:
        assert not node.is_cancellation_requested

        raise_if_cancelled(node)

    def test_passes_through_once_a_stale_flag_is_cleared(self, node: _StubNode) -> None:
        """The engine clears the flag at dispatch; a node re-run after a cancel must not raise."""
        node.request_cancellation()
        node.clear_cancellation()

        raise_if_cancelled(node)

    def test_guards_the_error_a_cancelled_run_would_otherwise_surface(self, node: _StubNode) -> None:
        """Pin the error being replaced: reading output off a never-run agent is the wrong message."""
        never_ran = GtAgent()

        # This is what agent.py hit at the try_throw_error(agent.output) call site.
        with pytest.raises(ValueError, match="has no output"):
            _ = never_ran.output

        # With the flag set, the guard raises first, so that read is never reached.
        node.request_cancellation()
        with pytest.raises(RuntimeError, match="cancelled"):
            raise_if_cancelled(node)
