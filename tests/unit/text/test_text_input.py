"""Tests for the TextInput node's optional split output."""

import pytest
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

from griptape_nodes_library.text.create_multiline_text import TextInput

SPLIT_PARAMS = ["split_mode", "delimiter_type", "include_delimiter", "trim_whitespace", "output_split"]


def _is_hidden(node: TextInput, name: str) -> bool:
    param = node.get_parameter_by_name(name)
    assert param is not None
    return bool(param.ui_options.get("hide"))


class TestTextInputSplit:
    @pytest.fixture
    def node(self, griptape_nodes: GriptapeNodes) -> TextInput:  # noqa: ARG002
        return TextInput(name="test_text_input")

    def test_split_off_by_default_hides_split_params(self, node: TextInput) -> None:
        assert node.get_parameter_value("split_text") is False
        assert all(_is_hidden(node, name) for name in SPLIT_PARAMS)

    def test_split_off_outputs_only_text(self, node: TextInput) -> None:
        node.set_parameter_value("text", "a\nb")
        node.process()

        assert node.parameter_output_values["text"] == "a\nb"
        assert node.parameter_output_values["output_split"] is None

    def test_split_on_shows_params_and_outputs_list(self, node: TextInput) -> None:
        node.set_parameter_value("text", "a\nb")
        node.set_parameter_value("split_text", True)
        node.process()

        assert not any(_is_hidden(node, name) for name in SPLIT_PARAMS)
        assert node.parameter_output_values["text"] == "a\nb"
        assert node.parameter_output_values["output_split"] == ["a", "b"]

    def test_parse_list_hides_delimiter_options(self, node: TextInput) -> None:
        node.set_parameter_value("split_text", True)
        node.set_parameter_value("split_mode", "parse_list")

        assert _is_hidden(node, "delimiter_type")
        assert _is_hidden(node, "include_delimiter")
        assert not _is_hidden(node, "trim_whitespace")
        assert not _is_hidden(node, "output_split")

    def test_turning_split_off_hides_and_clears_output(self, node: TextInput) -> None:
        node.set_parameter_value("text", "a,b")
        node.set_parameter_value("split_text", True)
        node.set_parameter_value("delimiter_type", "comma")
        assert node.parameter_output_values["output_split"] == ["a", "b"]

        node.set_parameter_value("split_text", False)

        assert all(_is_hidden(node, name) for name in SPLIT_PARAMS)
        assert node.parameter_output_values["output_split"] is None
