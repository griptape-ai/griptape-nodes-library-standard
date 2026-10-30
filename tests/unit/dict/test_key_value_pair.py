"""Tests for the KeyValuePair node."""

from typing import Any

import pytest
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

from griptape_nodes_library.dict.key_value_pair import KeyValuePair


class TestKeyValuePair:
    @pytest.fixture
    def node(self, griptape_nodes: GriptapeNodes) -> KeyValuePair:  # noqa: ARG002
        return KeyValuePair(name="test_key_value_pair")

    @pytest.mark.parametrize(
        "value",
        [
            ["a", "b"],
            {"nested": [1, 2]},
            3,
            1.5,
            True,
            "text",
        ],
    )
    def test_value_keeps_its_type(self, node: KeyValuePair, value: Any) -> None:
        node.set_parameter_value("key", "lines")
        node.set_parameter_value("value", value)
        node.process()

        assert node.parameter_output_values["dictionary"] == {"lines": value}
