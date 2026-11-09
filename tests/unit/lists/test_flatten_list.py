"""Tests for the FlattenList node."""

import pytest
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

from griptape_nodes_library.lists.flatten_list import FlattenList


class TestFlattenList:
    @pytest.fixture
    def node(self, griptape_nodes: GriptapeNodes) -> FlattenList:  # noqa: ARG002
        return FlattenList(name="test_flatten_list")

    def test_flattens_every_level_by_default(self, node: FlattenList) -> None:
        node.set_parameter_value("items", [1, [2, [3, [4]]], [], [[5]]])
        node.process()

        assert node.parameter_output_values["output"] == [1, 2, 3, 4, 5]

    def test_depth_limits_levels_flattened(self, node: FlattenList) -> None:
        node.set_parameter_value("items", [1, [2, [3, [4]]]])

        node.set_parameter_value("depth", 1)
        node.process()
        assert node.parameter_output_values["output"] == [1, 2, [3, [4]]]

        node.set_parameter_value("depth", 2)
        node.process()
        assert node.parameter_output_values["output"] == [1, 2, 3, [4]]

    def test_flat_list_is_unchanged(self, node: FlattenList) -> None:
        node.set_parameter_value("items", ["a", "b", "c"])
        node.process()

        assert node.parameter_output_values["output"] == ["a", "b", "c"]

    def test_strings_and_dicts_stay_whole(self, node: FlattenList) -> None:
        node.set_parameter_value("items", [["abc", {"k": [1, 2]}], [(1, 2)]])
        node.process()

        assert node.parameter_output_values["output"] == ["abc", {"k": [1, 2]}, (1, 2)]

    def test_missing_input_gives_empty_list(self, node: FlattenList) -> None:
        node.process()

        assert node.parameter_output_values["output"] == []

    def test_json_text_is_parsed(self, node: FlattenList) -> None:
        node.set_parameter_value("items", "[[1, 2], [3, [4]]]")
        node.process()

        assert node.parameter_output_values["output"] == [1, 2, 3, 4]

    def test_single_json_value_becomes_one_item_list(self, node: FlattenList) -> None:
        node.set_parameter_value("items", '{"k": 1}')
        node.process()

        assert node.parameter_output_values["output"] == [{"k": 1}]

    def test_invalid_json_text_raises(self, node: FlattenList) -> None:
        node.set_parameter_value("items", "[1, [2, 3]")

        with pytest.raises(ValueError, match="Invalid JSON"):
            node.process()
