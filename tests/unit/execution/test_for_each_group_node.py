"""Unit tests for ForEachGroupNode."""

from __future__ import annotations

import copy

import pytest
from griptape_nodes.exe_types.node_groups.subflow_node_group import LEFT_PARAMETERS_KEY, RIGHT_PARAMETERS_KEY

from griptape_nodes_library.execution.for_each_group import ForEachGroupNode


@pytest.fixture()
def default_node() -> ForEachGroupNode:
    """Return a ForEachGroupNode with default parameter values."""
    return ForEachGroupNode(name="test_for_each")


class TestSideParametersMetadata:
    def test_left_rail_membership(self, default_node: ForEachGroupNode) -> None:
        expected_left = {"items", "exec_in", "group_exec_in", "on_each", "current_item", "index"}

        assert set(default_node.metadata[LEFT_PARAMETERS_KEY]) == expected_left
        assert len(default_node.metadata[LEFT_PARAMETERS_KEY]) == len(expected_left)

    def test_right_rail_membership(self, default_node: ForEachGroupNode) -> None:
        expected_right = {
            "exec_out",
            "group_exec_out",
            "loop_complete",
            "new_item_to_add",
            "skip_iteration",
            "break_loop",
            "results",
        }

        assert set(default_node.metadata[RIGHT_PARAMETERS_KEY]) == expected_right
        assert len(default_node.metadata[RIGHT_PARAMETERS_KEY]) == len(expected_right)


class TestRestoredRailsAreStable:
    """Rebuilding a node from a saved node's metadata must not grow the rails."""

    def test_left_rail_survives_three_generations(self, default_node: ForEachGroupNode) -> None:
        generation_2 = ForEachGroupNode(name="gen_2", metadata=copy.deepcopy(default_node.metadata))
        generation_3 = ForEachGroupNode(name="gen_3", metadata=copy.deepcopy(generation_2.metadata))

        expected_left = set(default_node.metadata[LEFT_PARAMETERS_KEY])

        assert set(generation_3.metadata[LEFT_PARAMETERS_KEY]) == expected_left
        assert len(generation_3.metadata[LEFT_PARAMETERS_KEY]) == len(expected_left)

    def test_right_rail_survives_three_generations(self, default_node: ForEachGroupNode) -> None:
        generation_2 = ForEachGroupNode(name="gen_2", metadata=copy.deepcopy(default_node.metadata))
        generation_3 = ForEachGroupNode(name="gen_3", metadata=copy.deepcopy(generation_2.metadata))

        expected_right = set(default_node.metadata[RIGHT_PARAMETERS_KEY])

        assert set(generation_3.metadata[RIGHT_PARAMETERS_KEY]) == expected_right
        assert len(generation_3.metadata[RIGHT_PARAMETERS_KEY]) == len(expected_right)

    def test_left_rail_backfills_a_rail_saved_without_items_or_current_item(
        self, default_node: ForEachGroupNode
    ) -> None:
        """A workflow saved before items/current_item existed gains them back."""
        metadata = copy.deepcopy(default_node.metadata)
        metadata[LEFT_PARAMETERS_KEY] = ["exec_in", "group_exec_in", "on_each", "index"]

        restored = ForEachGroupNode(name="gen_2", metadata=metadata)

        expected_left = set(default_node.metadata[LEFT_PARAMETERS_KEY])

        assert set(restored.metadata[LEFT_PARAMETERS_KEY]) == expected_left
        assert len(restored.metadata[LEFT_PARAMETERS_KEY]) == len(expected_left)
