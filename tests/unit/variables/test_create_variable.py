"""Tests for CreateVariable node."""

from unittest.mock import patch

import pytest
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

from griptape_nodes_library.variables.create_variable import CreateVariable


@pytest.fixture
def node(griptape_nodes: GriptapeNodes) -> CreateVariable:  # noqa: ARG001
    return CreateVariable(name="test_create_variable")


@pytest.fixture
def flow_name():
    return "test_flow"


@pytest.fixture
def mock_flow(node, flow_name):
    """Mock _get_flow_name to return a consistent flow name."""
    with patch.object(node, "_get_flow_name", return_value=flow_name):
        yield


class TestBeforeValueSetVariableName:
    """Tests for eager variable registration when variable_name is set."""

    def test_setting_variable_name_registers_variable(self, node, mock_flow) -> None:
        """Setting variable_name should eagerly create the variable."""
        with patch.object(node, "_register_variable") as mock_register:
            node.before_value_set(node.variable_name_param, "my_var")

        mock_register.assert_called_once_with("my_var")

    def test_changing_variable_name_unregisters_old_and_registers_new(self, node, mock_flow) -> None:
        """Changing variable_name should remove the old variable and create the new one."""
        node.parameter_values["variable_name"] = "old_var"

        with (
            patch.object(node, "_unregister_variable") as mock_unregister,
            patch.object(node, "_register_variable") as mock_register,
        ):
            node.before_value_set(node.variable_name_param, "new_var")

        mock_unregister.assert_called_once_with("old_var")
        mock_register.assert_called_once_with("new_var")

    def test_setting_empty_variable_name_does_not_register(self, node, mock_flow) -> None:
        """Setting variable_name to empty string should not create a variable."""
        with patch.object(node, "_register_variable") as mock_register:
            node.before_value_set(node.variable_name_param, "")

        mock_register.assert_not_called()

    def test_clearing_variable_name_unregisters_old(self, node, mock_flow) -> None:
        """Clearing variable_name should remove the old variable."""
        node.parameter_values["variable_name"] = "old_var"

        with (
            patch.object(node, "_unregister_variable") as mock_unregister,
            patch.object(node, "_register_variable") as mock_register,
        ):
            node.before_value_set(node.variable_name_param, "")

        mock_unregister.assert_called_once_with("old_var")
        mock_register.assert_not_called()

    def test_setting_same_variable_name_does_not_unregister(self, node, mock_flow) -> None:
        """Setting the same variable_name should not unregister the variable."""
        node.parameter_values["variable_name"] = "my_var"

        with (
            patch.object(node, "_unregister_variable") as mock_unregister,
            patch.object(node, "_register_variable") as mock_register,
        ):
            node.before_value_set(node.variable_name_param, "my_var")

        mock_unregister.assert_not_called()
        mock_register.assert_called_once_with("my_var")


class TestRegisterVariable:
    """Tests for _register_variable helper."""

    def test_creates_variable_when_not_exists(self, node, flow_name, mock_flow) -> None:
        """Should create a variable if it doesn't already exist."""
        from griptape_nodes.retained_mode.events.variable_events import (
            CreateVariableResultSuccess,
            HasVariableResultSuccess,
        )

        has_result = HasVariableResultSuccess(exists=False, result_details="ok")
        create_result = CreateVariableResultSuccess(result_details="ok")

        with patch.object(GriptapeNodes, "handle_request", side_effect=[has_result, create_result]) as mock_handle:
            node._register_variable("my_var")

        assert mock_handle.call_count == 2

    def test_skips_creation_when_variable_exists(self, node, flow_name, mock_flow) -> None:
        """Should not create a variable if it already exists."""
        from griptape_nodes.retained_mode.events.variable_events import HasVariableResultSuccess

        has_result = HasVariableResultSuccess(exists=True, result_details="ok")

        with patch.object(GriptapeNodes, "handle_request", return_value=has_result) as mock_handle:
            node._register_variable("my_var")

        assert mock_handle.call_count == 1


class TestUnregisterVariable:
    """Tests for _unregister_variable helper."""

    def test_deletes_variable_when_exists(self, node, flow_name, mock_flow) -> None:
        """Should delete a variable if it exists."""
        from griptape_nodes.retained_mode.events.variable_events import (
            DeleteVariableResultSuccess,
            HasVariableResultSuccess,
        )

        has_result = HasVariableResultSuccess(exists=True, result_details="ok")
        delete_result = DeleteVariableResultSuccess(result_details="ok")

        with patch.object(GriptapeNodes, "handle_request", side_effect=[has_result, delete_result]) as mock_handle:
            node._unregister_variable("my_var")

        assert mock_handle.call_count == 2

    def test_skips_deletion_when_variable_not_exists(self, node, flow_name, mock_flow) -> None:
        """Should not attempt deletion if the variable doesn't exist."""
        from griptape_nodes.retained_mode.events.variable_events import HasVariableResultSuccess

        has_result = HasVariableResultSuccess(exists=False, result_details="ok")

        with patch.object(GriptapeNodes, "handle_request", return_value=has_result) as mock_handle:
            node._unregister_variable("my_var")

        assert mock_handle.call_count == 1
