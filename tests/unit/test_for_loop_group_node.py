"""Unit tests for ForLoopGroupNode."""

from __future__ import annotations

import pytest

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
        from griptape_nodes.exe_types.node_groups.subflow_node_group import LEFT_PARAMETERS_KEY

        left = default_node.metadata.get(LEFT_PARAMETERS_KEY, [])
        assert "start" in left

    def test_end_in_left_parameters(self, default_node: ForLoopGroupNode) -> None:
        from griptape_nodes.exe_types.node_groups.subflow_node_group import LEFT_PARAMETERS_KEY

        left = default_node.metadata.get(LEFT_PARAMETERS_KEY, [])
        assert "end" in left

    def test_step_in_left_parameters(self, default_node: ForLoopGroupNode) -> None:
        from griptape_nodes.exe_types.node_groups.subflow_node_group import LEFT_PARAMETERS_KEY

        left = default_node.metadata.get(LEFT_PARAMETERS_KEY, [])
        assert "step" in left
