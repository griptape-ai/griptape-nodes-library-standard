"""Tests for list nodes that look up, remove, or replace a single item."""

from typing import Any

import pytest
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

from griptape_nodes_library.lists.get_index_of_item import GetIndexOfItem
from griptape_nodes_library.lists.get_list_contains_item import GetListContainsItem
from griptape_nodes_library.lists.remove_from_list import RemoveFromList
from griptape_nodes_library.lists.replace_in_list import ReplaceInList

FALSY_ITEMS = [0, "", False]


class TestGetIndexOfItem:
    @pytest.fixture
    def node(self, griptape_nodes: GriptapeNodes) -> GetIndexOfItem:  # noqa: ARG002
        return GetIndexOfItem(name="test_get_index_of_item")

    @pytest.mark.parametrize("item", FALSY_ITEMS)
    def test_finds_falsy_item(self, node: GetIndexOfItem, item: Any) -> None:
        node.set_parameter_value("items", [1, item, 3])
        node.set_parameter_value("item", item)
        node.process()

        assert node.parameter_output_values["index"] == 1

    def test_missing_item_returns_minus_one(self, node: GetIndexOfItem) -> None:
        node.set_parameter_value("item", 9)
        node.set_parameter_value("items", [1, 2])
        node.process()

        assert node.parameter_output_values["index"] == -1

    def test_unset_item_returns_minus_one(self, node: GetIndexOfItem) -> None:
        node.set_parameter_value("items", [1, 2])
        node.process()

        assert node.parameter_output_values["index"] == -1


class TestGetListContainsItem:
    @pytest.fixture
    def node(self, griptape_nodes: GriptapeNodes) -> GetListContainsItem:  # noqa: ARG002
        return GetListContainsItem(name="test_get_list_contains_item")

    @pytest.mark.parametrize("item", FALSY_ITEMS)
    def test_finds_falsy_item(self, node: GetListContainsItem, item: Any) -> None:
        node.set_parameter_value("items", [1, item, 3])
        node.set_parameter_value("item", item)
        node.process()

        assert node.parameter_output_values["contains_item"] is True

    def test_unset_item_is_false(self, node: GetListContainsItem) -> None:
        node.set_parameter_value("items", [1, 2])
        node.process()

        assert node.parameter_output_values["contains_item"] is False


class TestRemoveFromList:
    @pytest.fixture
    def node(self, griptape_nodes: GriptapeNodes) -> RemoveFromList:  # noqa: ARG002
        return RemoveFromList(name="test_remove_from_list")

    def test_removes_matching_item(self, node: RemoveFromList) -> None:
        node.set_parameter_value("items", [1, 2, 3])
        node.set_parameter_value("item", 2)
        node.process()

        assert node.parameter_output_values["output"] == [1, 3]

    def test_missing_item_leaves_list_unchanged(self, node: RemoveFromList) -> None:
        node.set_parameter_value("items", [1, 2])
        node.set_parameter_value("item", 9)
        node.process()

        assert node.parameter_output_values["output"] == [1, 2]

    def test_removes_falsy_item(self, node: RemoveFromList) -> None:
        node.set_parameter_value("items", [1, 0, 3])
        node.set_parameter_value("item", 0)
        node.process()

        assert node.parameter_output_values["output"] == [1, 3]

    def test_unset_item_raises_clear_error(self, node: RemoveFromList) -> None:
        node.set_parameter_value("items", [1, 2])

        with pytest.raises(ValueError, match="Set 'item'"):
            node.process()

    def test_unset_index_raises_clear_error(self, node: RemoveFromList) -> None:
        node.set_parameter_value("items", [1, 2])
        node.set_parameter_value("remove_item_by", "index")

        with pytest.raises(ValueError, match="Set 'index'"):
            node.process()


class TestReplaceInList:
    @pytest.fixture
    def node(self, griptape_nodes: GriptapeNodes) -> ReplaceInList:  # noqa: ARG002
        return ReplaceInList(name="test_replace_in_list")

    @pytest.mark.parametrize("new_item", FALSY_ITEMS)
    def test_replaces_with_falsy_item(self, node: ReplaceInList, new_item: Any) -> None:
        node.set_parameter_value("items", [1, 2, 3])
        node.set_parameter_value("replace_by", "index")
        node.set_parameter_value("index_to_replace", 1)
        node.set_parameter_value("new_item", new_item)
        node.process()

        assert node.parameter_output_values["output"] == [1, new_item, 3]
