"""Tests for GroupByKey node."""

import pytest
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

from griptape_nodes_library.dict.group_by_key import GroupByKey


class TestGroupByKey:
    @pytest.fixture
    def node(self, griptape_nodes: GriptapeNodes) -> GroupByKey:  # noqa: ARG002
        return GroupByKey(name="test_group_by_key")

    def test_groups_key_value_pairs(self, node: GroupByKey) -> None:
        node.parameter_values["items"] = [{"Lighting": "rim"}, {"FX": "dust"}, {"Lighting": "moon"}]
        node.process()
        assert node.parameter_output_values["output"] == {"Lighting": ["rim", "moon"], "FX": ["dust"]}

    def test_keeps_first_seen_key_order(self, node: GroupByKey) -> None:
        node.parameter_values["items"] = [{"b": 1}, {"a": 2}, {"b": 3}]
        node.process()
        assert list(node.parameter_output_values["output"]) == ["b", "a"]

    def test_multi_key_dict_contributes_each_pair(self, node: GroupByKey) -> None:
        node.parameter_values["items"] = [{"a": 1, "b": 2}, {"a": 3}]
        node.process()
        assert node.parameter_output_values["output"] == {"a": [1, 3], "b": [2]}

    def test_groups_records_by_field(self, node: GroupByKey) -> None:
        rows = [
            {"shot": "sh010", "note": "rim"},
            {"shot": "sh020", "note": "dust"},
            {"shot": "sh010", "note": "grain"},
        ]
        node.parameter_values["items"] = rows
        node.parameter_values["group_by"] = "shot"
        node.process()
        assert node.parameter_output_values["output"] == {"sh010": [rows[0], rows[2]], "sh020": [rows[1]]}

    def test_non_string_group_values_become_string_keys(self, node: GroupByKey) -> None:
        node.parameter_values["items"] = [{"frame": 1001}, {"frame": 1001}]
        node.parameter_values["group_by"] = "frame"
        node.process()
        assert list(node.parameter_output_values["output"]) == ["1001"]

    def test_single_dict_is_treated_as_one_item(self, node: GroupByKey) -> None:
        node.parameter_values["items"] = {"a": 1}
        node.process()
        assert node.parameter_output_values["output"] == {"a": [1]}

    def test_empty_input(self, node: GroupByKey) -> None:
        node.parameter_values["items"] = []
        node.process()
        assert node.parameter_output_values["output"] == {}

    def test_regroups_when_items_arrive_after_resolving(self, node: GroupByKey) -> None:
        node.process()
        assert node.parameter_output_values["output"] == {}
        node.set_parameter_value("items", [{"Lighting": "rim"}, {"Lighting": "moon"}])
        assert node.parameter_output_values["output"] == {"Lighting": ["rim", "moon"]}

    def test_bad_items_while_wiring_do_not_raise(self, node: GroupByKey) -> None:
        node.set_parameter_value("items", ["not a dict"])
        assert node.parameter_output_values["output"] == {}

    def test_non_dict_item_raises(self, node: GroupByKey) -> None:
        node.parameter_values["items"] = [{"a": 1}, "oops"]
        with pytest.raises(TypeError, match="Item 1 in 'Items' is a str"):
            node.process()

    def test_missing_group_field_raises(self, node: GroupByKey) -> None:
        node.parameter_values["items"] = [{"shot": "sh010"}, {"note": "rim"}]
        node.parameter_values["group_by"] = "shot"
        with pytest.raises(KeyError, match="Item 1 in 'Items' has no 'shot' field"):
            node.process()
