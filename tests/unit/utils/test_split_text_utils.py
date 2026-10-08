"""Tests for the shared text splitting helpers."""

import pytest

from griptape_nodes_library.utils.split_text_utils import SplitMode, split_text


class TestSplitText:
    def test_splits_by_named_delimiter(self) -> None:
        assert split_text("a,b,c", SplitMode.SPLIT, "comma", include_delimiter=False, trim_whitespace=False) == [
            "a",
            "b",
            "c",
        ]

    def test_include_delimiter_appends_to_all_but_last(self) -> None:
        assert split_text("a\nb\nc", "split", "newlines", include_delimiter=True, trim_whitespace=False) == [
            "a\n",
            "b\n",
            "c",
        ]

    def test_trim_whitespace_strips_both_ends(self) -> None:
        assert split_text("a , b,  c  ", "split", "comma", include_delimiter=False, trim_whitespace=True) == [
            "a",
            "b",
            "c",
        ]

    def test_trim_keeps_whitespace_delimiter_when_included(self) -> None:
        assert split_text("a  \n  b", "split", "newlines", include_delimiter=True, trim_whitespace=True) == [
            "a\n",
            "b",
        ]

    def test_trim_strips_parse_list_delimiter_fallback(self) -> None:
        assert split_text(" one | two ", "parse_list", "pipe", include_delimiter=False, trim_whitespace=True) == [
            "one",
            "two",
        ]

    def test_keeps_empty_items_by_default(self) -> None:
        assert split_text("a\n\nb", "split", "newlines", include_delimiter=False, trim_whitespace=False) == [
            "a",
            "",
            "b",
        ]

    def test_remove_empty_drops_blank_and_whitespace_only_items(self) -> None:
        assert split_text(
            "a\n\n   \nb\n", "split", "newlines", include_delimiter=False, trim_whitespace=False, remove_empty=True
        ) == ["a", "b"]

    def test_remove_empty_drops_blank_items_with_delimiter_included(self) -> None:
        assert split_text(
            "a\n\nb", "split", "newlines", include_delimiter=True, trim_whitespace=False, remove_empty=True
        ) == ["a\n", "b"]

    def test_remove_empty_applies_to_parse_list(self) -> None:
        assert split_text(
            '["one", "", "two"]',
            "parse_list",
            "newlines",
            include_delimiter=False,
            trim_whitespace=False,
            remove_empty=True,
        ) == ["one", "two"]

    def test_parse_list_reads_json(self) -> None:
        assert split_text('["one", 2]', "parse_list", "newlines", include_delimiter=False, trim_whitespace=False) == [
            "one",
            "2",
        ]

    def test_parse_list_reads_python_literal(self) -> None:
        assert split_text(
            "['one', 'two']", "parse_list", "newlines", include_delimiter=False, trim_whitespace=False
        ) == [
            "one",
            "two",
        ]

    def test_parse_list_falls_back_to_delimiter(self) -> None:
        assert split_text("one two", "parse_list", "space", include_delimiter=False, trim_whitespace=False) == [
            "one",
            "two",
        ]

    def test_unknown_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown split mode"):
            split_text("a", "bogus", "newlines", include_delimiter=False, trim_whitespace=False)
