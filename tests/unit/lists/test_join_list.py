"""Tests for the JoinList node."""

import pytest
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

from griptape_nodes_library.lists.join_list import JoinList


class TestJoinList:
    @pytest.fixture
    def node(self, griptape_nodes: GriptapeNodes) -> JoinList:  # noqa: ARG002
        return JoinList(name="test_join_list")

    def test_joins_with_new_lines_by_default(self, node: JoinList) -> None:
        node.set_parameter_value("items", ["a", "b", "c"])
        node.process()

        assert node.parameter_output_values["output"] == "a\nb\nc"

    def test_custom_separator(self, node: JoinList) -> None:
        node.set_parameter_value("items", ["a", "b"])
        node.set_parameter_value("separator", ", ")
        node.process()

        assert node.parameter_output_values["output"] == "a, b"

    def test_tab_escape_in_separator(self, node: JoinList) -> None:
        node.set_parameter_value("items", ["a", "b"])
        node.set_parameter_value("separator", "\\t")
        node.process()

        assert node.parameter_output_values["output"] == "a\tb"

    def test_doubled_backslash_is_a_literal_backslash(self, node: JoinList) -> None:
        node.set_parameter_value("items", ["a.png", "b.png"])
        node.set_parameter_value("item_prefix", "C:\\\\temp\\\\")
        node.set_parameter_value("separator", "\\\\n")
        node.process()

        assert node.parameter_output_values["output"] == "C:\\temp\\a.png\\nC:\\temp\\b.png"

    def test_prefix_and_suffix_wrap_each_item(self, node: JoinList) -> None:
        node.set_parameter_value("items", ["a", "b"])
        node.set_parameter_value("item_prefix", "- ")
        node.set_parameter_value("item_suffix", ";")
        node.process()

        assert node.parameter_output_values["output"] == "- a;\n- b;"

    def test_non_text_items_are_converted(self, node: JoinList) -> None:
        node.set_parameter_value("items", [1, 2.5, True, None])
        node.process()

        assert node.parameter_output_values["output"] == "1\n2.5\nTrue\n"

    def test_empty_or_missing_list_gives_empty_text(self, node: JoinList) -> None:
        node.process()
        assert node.parameter_output_values["output"] == ""

        node.set_parameter_value("items", [])
        node.process()
        assert node.parameter_output_values["output"] == ""

    def test_output_updates_when_a_value_is_set(self, node: JoinList) -> None:
        node.set_parameter_value("items", ["a", "b"])

        assert node.parameter_output_values["output"] == "a\nb"
