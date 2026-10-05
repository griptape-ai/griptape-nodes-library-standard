"""Unit tests for ForLoopGroupNode."""

from __future__ import annotations

import copy

import pytest
from griptape_nodes.exe_types.node_groups.subflow_node_group import LEFT_PARAMETERS_KEY, RIGHT_PARAMETERS_KEY

from griptape_nodes_library.execution.for_loop_group import ForLoopGroupNode


@pytest.fixture()
def default_node() -> ForLoopGroupNode:
    """Return a ForLoopGroupNode with default parameter values."""
    return ForLoopGroupNode(name="test_for_loop")


def _make_node(*, start: int, end: int, step: int = 1, end_inclusive: bool = True) -> ForLoopGroupNode:
    """Create a ForLoopGroupNode and override its parameter values."""
    node = ForLoopGroupNode(name="test_for_loop")
    node.set_parameter_value("start", start)
    node.set_parameter_value("end", end)
    node.set_parameter_value("step", step)
    node.set_parameter_value("end_inclusive", end_inclusive)
    return node


class TestDefaultParameterValues:
    def test_default_start(self, default_node: ForLoopGroupNode) -> None:
        assert default_node.get_parameter_value("start") == 1

    def test_default_end(self, default_node: ForLoopGroupNode) -> None:
        assert default_node.get_parameter_value("end") == 10

    def test_default_step(self, default_node: ForLoopGroupNode) -> None:
        assert default_node.get_parameter_value("step") == 1

    def test_default_end_inclusive(self, default_node: ForLoopGroupNode) -> None:
        assert default_node.get_parameter_value("end_inclusive") is True


class TestGetAllIterationValues:
    def test_ascending_inclusive(self) -> None:
        node = _make_node(start=1, end=5, step=1, end_inclusive=True)
        assert node.get_all_iteration_values() == [1, 2, 3, 4, 5]

    def test_ascending_exclusive(self) -> None:
        node = _make_node(start=1, end=5, step=1, end_inclusive=False)
        assert node.get_all_iteration_values() == [1, 2, 3, 4]

    def test_ascending_with_step(self) -> None:
        node = _make_node(start=1, end=10, step=2, end_inclusive=True)
        assert node.get_all_iteration_values() == [1, 3, 5, 7, 9]

    def test_descending_inclusive(self) -> None:
        node = _make_node(start=10, end=1, step=3, end_inclusive=True)
        assert node.get_all_iteration_values() == [10, 7, 4, 1]

    def test_empty_range_exclusive(self) -> None:
        node = _make_node(start=5, end=5, step=1, end_inclusive=False)
        assert node.get_all_iteration_values() == []

    def test_single_iteration_inclusive(self) -> None:
        node = _make_node(start=5, end=5, step=1, end_inclusive=True)
        assert node.get_all_iteration_values() == [5]


class TestLeftParametersMetadata:
    def test_start_in_left_parameters(self, default_node: ForLoopGroupNode) -> None:
        left = default_node.metadata.get(LEFT_PARAMETERS_KEY, [])
        assert "start" in left

    def test_end_in_left_parameters(self, default_node: ForLoopGroupNode) -> None:
        left = default_node.metadata.get(LEFT_PARAMETERS_KEY, [])
        assert "end" in left

    def test_step_in_left_parameters(self, default_node: ForLoopGroupNode) -> None:
        left = default_node.metadata.get(LEFT_PARAMETERS_KEY, [])
        assert "step" in left

    def test_left_rail_membership(self, default_node: ForLoopGroupNode) -> None:
        expected_left = {"exec_in", "start", "end", "step", "group_exec_in", "on_each", "index"}

        assert set(default_node.metadata[LEFT_PARAMETERS_KEY]) == expected_left
        assert len(default_node.metadata[LEFT_PARAMETERS_KEY]) == len(expected_left)

    def test_right_rail_membership(self, default_node: ForLoopGroupNode) -> None:
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

    def test_left_rail_survives_three_generations(self, default_node: ForLoopGroupNode) -> None:
        generation_2 = ForLoopGroupNode(name="gen_2", metadata=copy.deepcopy(default_node.metadata))
        generation_3 = ForLoopGroupNode(name="gen_3", metadata=copy.deepcopy(generation_2.metadata))

        expected_left = set(default_node.metadata[LEFT_PARAMETERS_KEY])

        assert set(generation_3.metadata[LEFT_PARAMETERS_KEY]) == expected_left
        assert len(generation_3.metadata[LEFT_PARAMETERS_KEY]) == len(expected_left)

    def test_right_rail_survives_three_generations(self, default_node: ForLoopGroupNode) -> None:
        generation_2 = ForLoopGroupNode(name="gen_2", metadata=copy.deepcopy(default_node.metadata))
        generation_3 = ForLoopGroupNode(name="gen_3", metadata=copy.deepcopy(generation_2.metadata))

        expected_right = set(default_node.metadata[RIGHT_PARAMETERS_KEY])

        assert set(generation_3.metadata[RIGHT_PARAMETERS_KEY]) == expected_right
        assert len(generation_3.metadata[RIGHT_PARAMETERS_KEY]) == len(expected_right)

    def test_left_rail_backfills_a_rail_saved_without_start_end_or_step(self, default_node: ForLoopGroupNode) -> None:
        """A workflow saved before start/end/step existed gains them back."""
        metadata = copy.deepcopy(default_node.metadata)
        metadata[LEFT_PARAMETERS_KEY] = ["exec_in", "group_exec_in", "on_each", "index"]

        restored = ForLoopGroupNode(name="gen_2", metadata=metadata)

        expected_left = set(default_node.metadata[LEFT_PARAMETERS_KEY])

        assert set(restored.metadata[LEFT_PARAMETERS_KEY]) == expected_left
        assert len(restored.metadata[LEFT_PARAMETERS_KEY]) == len(expected_left)
