"""Tests for MarkdownTableToListOfDicts node."""

import pytest
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

from griptape_nodes_library.convert.markdown_table_to_list_of_dicts import (
    MarkdownTableToListOfDicts,
)


class TestMarkdownTableToListOfDicts:
    @pytest.fixture
    def node(self, griptape_nodes: GriptapeNodes) -> MarkdownTableToListOfDicts:  # noqa: ARG002
        return MarkdownTableToListOfDicts(name="test_markdown_table")

    def test_basic_table(self, node: MarkdownTableToListOfDicts) -> None:
        node.parameter_values["markdown"] = (
            "| name | age | city |\n|------|-----|------|\n| Alice | 30 | NYC |\n| Bob | 25 | LA |"
        )
        node.process()
        assert node.parameter_output_values["output"] == [
            {"name": "Alice", "age": "30", "city": "NYC"},
            {"name": "Bob", "age": "25", "city": "LA"},
        ]

    def test_separator_row_is_skipped(self, node: MarkdownTableToListOfDicts) -> None:
        node.parameter_values["markdown"] = "| a | b |\n| :--- | ---: |\n| 1 | 2 |"
        node.process()
        assert node.parameter_output_values["output"] == [{"a": "1", "b": "2"}]

    def test_table_without_outer_pipes(self, node: MarkdownTableToListOfDicts) -> None:
        node.parameter_values["markdown"] = "name | age\n---|---\nAlice | 30"
        node.process()
        assert node.parameter_output_values["output"] == [{"name": "Alice", "age": "30"}]

    def test_ragged_row_is_padded(self, node: MarkdownTableToListOfDicts) -> None:
        node.parameter_values["markdown"] = "| a | b |\n|---|---|\n| 1 |"
        node.process()
        assert node.parameter_output_values["output"] == [{"a": "1", "b": ""}]

    def test_escaped_pipe_in_cell(self, node: MarkdownTableToListOfDicts) -> None:
        node.parameter_values["markdown"] = "| expr |\n|------|\n| a \\| b |"
        node.process()
        assert node.parameter_output_values["output"] == [{"expr": "a | b"}]

    def test_empty_input(self, node: MarkdownTableToListOfDicts) -> None:
        node.parameter_values["markdown"] = ""
        node.process()
        assert node.parameter_output_values["output"] == []

    def test_output_type_is_list(self, node: MarkdownTableToListOfDicts) -> None:
        node.parameter_values["markdown"] = "| a |\n|---|\n| 1 |"
        node.process()
        assert isinstance(node.parameter_output_values["output"], list)
