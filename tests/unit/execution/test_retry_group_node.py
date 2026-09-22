"""Unit tests for RetryGroupNode."""

from __future__ import annotations

import copy

import pytest
from griptape_nodes.exe_types.node_groups.subflow_node_group import LEFT_PARAMETERS_KEY, RIGHT_PARAMETERS_KEY

from griptape_nodes_library.execution.retry_group import RetryGroupNode


@pytest.fixture()
def default_node() -> RetryGroupNode:
    """Return a RetryGroupNode with default parameter values."""
    return RetryGroupNode(name="test_retry")


class TestSideParametersMetadata:
    def test_right_rail_membership(self, default_node: RetryGroupNode) -> None:
        expected_right = {"group_exec_out", "done", "continue_loop", "total_iterations", "was_successful"}

        assert set(default_node.metadata[RIGHT_PARAMETERS_KEY]) == expected_right
        assert len(default_node.metadata[RIGHT_PARAMETERS_KEY]) == len(expected_right)

    def test_left_rail_membership(self, default_node: RetryGroupNode) -> None:
        expected_left = {"group_exec_in", "iteration"}

        assert set(default_node.metadata[LEFT_PARAMETERS_KEY]) == expected_left
        assert len(default_node.metadata[LEFT_PARAMETERS_KEY]) == len(expected_left)


class TestRestoredRailsAreStable:
    """Rebuilding a node from a saved node's metadata must not grow the rails."""

    def test_right_rail_survives_three_generations(self, default_node: RetryGroupNode) -> None:
        generation_2 = RetryGroupNode(name="gen_2", metadata=copy.deepcopy(default_node.metadata))
        generation_3 = RetryGroupNode(name="gen_3", metadata=copy.deepcopy(generation_2.metadata))

        expected_right = set(default_node.metadata[RIGHT_PARAMETERS_KEY])

        assert set(generation_3.metadata[RIGHT_PARAMETERS_KEY]) == expected_right
        assert len(generation_3.metadata[RIGHT_PARAMETERS_KEY]) == len(expected_right)

    def test_left_rail_survives_three_generations(self, default_node: RetryGroupNode) -> None:
        generation_2 = RetryGroupNode(name="gen_2", metadata=copy.deepcopy(default_node.metadata))
        generation_3 = RetryGroupNode(name="gen_3", metadata=copy.deepcopy(generation_2.metadata))

        expected_left = set(default_node.metadata[LEFT_PARAMETERS_KEY])

        assert set(generation_3.metadata[LEFT_PARAMETERS_KEY]) == expected_left
        assert len(generation_3.metadata[LEFT_PARAMETERS_KEY]) == len(expected_left)
