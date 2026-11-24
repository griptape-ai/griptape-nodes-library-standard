"""Tests for the SearchReplaceText node."""

import pytest
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

from griptape_nodes_library.text.search_replace_text import SearchReplaceText


@pytest.fixture
def node(griptape_nodes: GriptapeNodes) -> SearchReplaceText:  # noqa: ARG001
    node = SearchReplaceText(name="test_search_replace_text")
    node.parameter_values["input_text"] = "abc"
    node.parameter_values["search_pattern"] = "b"
    return node


class TestPlainMode:
    @pytest.mark.parametrize(
        ("replacement_text", "expected"),
        [
            ("\\1", "a\\1c"),
            ("\\d", "a\\dc"),
            ("x\\", "ax\\c"),
            ("a\\nb", "aa\\nbc"),
            ("\\\\", "a\\\\c"),
            ("\\g<0>\\g<0>", "a\\g<0>\\g<0>c"),
            ("C:\\Users\\shots", "aC:\\Users\\shotsc"),
        ],
    )
    def test_replacement_text_is_inserted_literally(
        self, node: SearchReplaceText, replacement_text: str, expected: str
    ) -> None:
        node.parameter_values["replacement_text"] = replacement_text
        assert node._search_replace() == expected

    def test_search_pattern_is_matched_literally(self, node: SearchReplaceText) -> None:
        node.parameter_values["input_text"] = "a.c abc"
        node.parameter_values["search_pattern"] = "."
        node.parameter_values["replacement_text"] = "-"
        assert node._search_replace() == "a-c abc"

    def test_replace_first_only(self, node: SearchReplaceText) -> None:
        node.parameter_values["input_text"] = "bbb"
        node.parameter_values["replacement_text"] = "\\"
        node.parameter_values["replace_all"] = False
        assert node._search_replace() == "\\bb"

    def test_empty_replacement_removes_match(self, node: SearchReplaceText) -> None:
        node.parameter_values["replacement_text"] = None
        assert node._search_replace() == "ac"


class TestRegexMode:
    @pytest.fixture(autouse=True)
    def use_regex(self, node: SearchReplaceText) -> None:
        node.parameter_values["use_regex"] = True

    @pytest.mark.parametrize(
        ("search_pattern", "replacement_text", "expected"),
        [
            ("(b)", "[\\1]", "a[b]c"),
            ("(?P<letter>b)", "\\g<letter>\\g<letter>", "abbc"),
            ("b", "\\n", "a\nc"),
            ("b", "\\\\", "a\\c"),
        ],
    )
    def test_backreferences_and_escapes_are_interpreted(
        self, node: SearchReplaceText, search_pattern: str, replacement_text: str, expected: str
    ) -> None:
        node.parameter_values["search_pattern"] = search_pattern
        node.parameter_values["replacement_text"] = replacement_text
        assert node._search_replace() == expected

    def test_invalid_pattern_raises(self, node: SearchReplaceText) -> None:
        node.parameter_values["search_pattern"] = "("
        node.parameter_values["replacement_text"] = "x"
        with pytest.raises(ValueError, match="'search_pattern'"):
            node._search_replace()

    def test_invalid_group_reference_raises(self, node: SearchReplaceText) -> None:
        node.parameter_values["replacement_text"] = "\\1"
        with pytest.raises(ValueError, match="'replacement_text'"):
            node._search_replace()

    def test_process_reports_invalid_pattern(self, node: SearchReplaceText) -> None:
        node.parameter_values["search_pattern"] = "("
        with pytest.raises(ValueError, match="not a valid regular expression"):
            node.process()

    def test_live_update_keeps_input_for_invalid_pattern(self, node: SearchReplaceText) -> None:
        node.set_parameter_value("search_pattern", "(")
        assert node.parameter_output_values["output"] == "abc"
