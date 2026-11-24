"""Tests for the AddToList node."""

import pytest
from griptape_nodes.exe_types.core_types import ParameterMode
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

from griptape_nodes_library.lists.add_to_list import AddToList


class TestAddToList:
    @pytest.fixture
    def node(self, griptape_nodes: GriptapeNodes) -> AddToList:  # noqa: ARG002
        return AddToList(name="test_add_to_list")

    def test_skip_empty_values_is_input_and_property_only(self, node: AddToList) -> None:
        assert node.skip_empty_values.allowed_modes == {ParameterMode.INPUT, ParameterMode.PROPERTY}

    def test_only_the_list_is_output(self, node: AddToList) -> None:
        node.set_parameter_value("items", [1])
        node.set_parameter_value("item", 2)
        node.process()

        assert node.parameter_output_values == {"output": [1, 2]}

    def test_skip_empty_values_skips_none(self, node: AddToList) -> None:
        node.set_parameter_value("items", [1])
        node.set_parameter_value("item", None)
        node.process()

        assert node.parameter_output_values["output"] == [1]

    def test_skip_empty_values_off_adds_none(self, node: AddToList) -> None:
        node.set_parameter_value("items", [1])
        node.set_parameter_value("item", None)
        node.set_parameter_value("skip_empty_values", False)
        node.process()

        assert node.parameter_output_values["output"] == [1, None]
