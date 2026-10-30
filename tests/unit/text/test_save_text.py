"""Tests for the SaveText node's text input."""

import pytest
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

from griptape_nodes_library.text.save_text import SaveText


class TestSaveTextInput:
    @pytest.fixture
    def node(self, griptape_nodes: GriptapeNodes) -> SaveText:  # noqa: ARG002
        return SaveText(name="test_save_text")

    @pytest.mark.parametrize("source_type", ["str", "yaml", "json", "dict", "int"])
    def test_text_accepts_any_incoming_type(self, node: SaveText, source_type: str) -> None:
        param = node.get_parameter_by_name("text")
        assert param is not None
        assert param.is_incoming_type_allowed(source_type)

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("key: value\n", "key: value\n"),
            ({"key": "value"}, '{"key": "value"}'),
            (42, "42"),
        ],
    )
    def test_text_converts_value_to_string(self, node: SaveText, value: object, expected: str) -> None:
        node.set_parameter_value("text", value)
        assert node.get_parameter_value("text") == expected
