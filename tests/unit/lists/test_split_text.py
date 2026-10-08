"""Tests for the SplitText node."""

import pytest
from griptape_nodes.exe_types.core_types import ParameterMode
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

from griptape_nodes_library.lists.split_text import SplitText
from griptape_nodes_library.text.create_multiline_text import TextInput

TOGGLE_OPTIONS = ["include_delimiter", "trim_whitespace", "remove_empty"]


class TestSplitText:
    @pytest.fixture
    def node(self, griptape_nodes: GriptapeNodes) -> SplitText:  # noqa: ARG002
        return SplitText(name="test_split_text")

    def test_keeps_empty_items_by_default(self, node: SplitText) -> None:
        node.set_parameter_value("text", "a\n\nb")

        assert node.parameter_output_values["output"] == ["a", "", "b"]

    def test_remove_empty_drops_blank_lines(self, node: SplitText) -> None:
        node.set_parameter_value("text", "a\n\nb")
        node.set_parameter_value("remove_empty", True)

        assert node.parameter_output_values["output"] == ["a", "b"]

    def test_toggle_options_accept_input(self, node: SplitText) -> None:
        for name in TOGGLE_OPTIONS:
            param = node.get_parameter_by_name(name)
            assert param is not None
            assert ParameterMode.INPUT in param.allowed_modes

    def test_parse_list_hides_delimiter_options(self, node: SplitText) -> None:
        node.set_parameter_value("split_mode", "parse_list")

        for name in ["delimiter_type", "include_delimiter"]:
            param = node.get_parameter_by_name(name)
            assert param is not None
            assert param.ui_options.get("hide")


class TestSplitOptionsMatchTextInput:
    """Both nodes build their options from SplitOptions, so only the intended differences should show."""

    def test_same_options_and_defaults(self, griptape_nodes: GriptapeNodes) -> None:  # noqa: ARG002
        split_node = SplitText(name="test_split_text")
        text_node = TextInput(name="test_text_input")

        split_names = [param.name for param in split_node.split_options.parameters]
        text_names = [param.name for param in text_node.split_options.parameters]
        assert split_names == text_names

        for split_param, text_param in zip(
            split_node.split_options.parameters, text_node.split_options.parameters, strict=True
        ):
            assert split_param.tooltip == text_param.tooltip
            if split_param.name != "remove_empty":
                assert split_param.default_value == text_param.default_value
