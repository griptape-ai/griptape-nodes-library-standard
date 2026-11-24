"""Tests for the SortList node."""

import pytest
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

from griptape_nodes_library.lists.sort_list import SortList


class TestSortList:
    @pytest.fixture
    def node(self, griptape_nodes: GriptapeNodes) -> SortList:  # noqa: ARG002
        return SortList(name="test_sort_list")

    def test_sorts_list(self, node: SortList) -> None:
        node.set_parameter_value("items", [3, 1, 2])
        node.process()

        assert node.parameter_output_values["output"] == [1, 2, 3]

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("[3,1,2,", [1, 2, 3]),
            ("[3, 1, 2", [1, 2, 3]),
            ("['b','a'", ["a", "b"]),
        ],
    )
    def test_sorts_malformed_json_text(self, node: SortList, raw: str, expected: list) -> None:
        node.set_parameter_value("items", raw)
        node.process()

        assert node.parameter_output_values["output"] == expected

    def test_text_that_is_not_a_list_sorts_to_empty(self, node: SortList) -> None:
        node.set_parameter_value("items", "hello world")
        node.process()

        assert node.parameter_output_values["output"] == []
