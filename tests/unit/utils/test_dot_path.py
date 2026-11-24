"""Tests for set_value_at_path."""

from __future__ import annotations

from typing import Any

import pytest

from griptape_nodes_library.utils.dot_path import set_value_at_path


@pytest.mark.parametrize(
    ("data", "path", "expected"),
    [
        ({"items": ["a", "b"]}, "items[0]", {"items": ["X", "b"]}),
        ({"items": ["a", "b"]}, "items[1]", {"items": ["a", "X"]}),
        ({"items": ["a", "b"]}, "items[2]", {"items": ["a", "b", "X"]}),
        ({"items": ["a"]}, "items[3]", {"items": ["a", None, None, "X"]}),
        ({}, "items[0]", {"items": ["X"]}),
        ({"items": [{"title": "Book"}]}, "items[0].title", {"items": [{"title": "X"}]}),
        ({"items": [{"title": "Book"}]}, "items[2].title", {"items": [{"title": "Book"}, None, {"title": "X"}]}),
        ({"items": ["a"]}, "items[3].t", {"items": ["a", None, None, {"t": "X"}]}),
        ({}, "items[0].title", {"items": [{"title": "X"}]}),
        ({}, "a.b.c", {"a": {"b": {"c": "X"}}}),
        ({"user": {"name": "A"}}, "user.name", {"user": {"name": "X"}}),
        ({"a": {"list": [1, 2]}}, "a.list[1]", {"a": {"list": [1, "X"]}}),
        ({"a": 1}, "", "X"),
        ({"a": None}, "a.b", {"a": {"b": "X"}}),
        ({"items": None}, "items[0]", {"items": ["X"]}),
        ({"items": None}, "items[0].t", {"items": [{"t": "X"}]}),
        ({"items": ["a", None]}, "items[1].t", {"items": ["a", {"t": "X"}]}),
    ],
)
def test_sets_value(data: Any, path: str, expected: Any) -> None:
    assert set_value_at_path(data, path, "X") == expected


def test_does_not_modify_input() -> None:
    data = {"items": ["a"]}

    set_value_at_path(data, "items[0]", "X")

    assert data == {"items": ["a"]}


@pytest.mark.parametrize(
    ("data", "path"),
    [
        ({"a": 1}, "a.b"),
        ({"a": 1}, "a[0]"),
        ({"a": "text"}, "a[0].b"),
        ({"items": ["a"]}, "items[0].title"),
        ({"a": [[1, 2]]}, "a[0][1]"),
        ({"items": {"k": 1}}, "items[0]"),
        (["a", "b"], "items[0]"),
        (["a", "b"], "a"),
        (5, "a"),
    ],
)
def test_path_that_cannot_be_followed_raises(data: Any, path: str) -> None:
    with pytest.raises(ValueError, match="Cannot follow path"):
        set_value_at_path(data, path, "X")


def test_padded_list_can_be_written_through_again() -> None:
    padded = set_value_at_path({"items": ["a"]}, "items[2].t", "X")

    assert set_value_at_path(padded, "items[1].t", "Y") == {"items": ["a", {"t": "Y"}, {"t": "X"}]}
