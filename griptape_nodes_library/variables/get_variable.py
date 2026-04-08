import logging
from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode, ParameterTypeBuiltin
from griptape_nodes.exe_types.node_types import ControlNode, NodeResolutionState

from griptape_nodes_library.variables.variable_utils import (
    create_advanced_parameter_group,
    get_variable,
    list_variable_names,
    scope_string_to_variable_scope,
)

logger = logging.getLogger("griptape_nodes")


class GetVariable(ControlNode):
    def __init__(
        self,
        name: str,
        metadata: dict[Any, Any] | None = None,
    ) -> None:
        super().__init__(name, metadata)

        self.variable_name_param = Parameter(
            name="variable_name",
            type="str",
            allowed_modes={ParameterMode.INPUT, ParameterMode.OUTPUT, ParameterMode.PROPERTY},
            tooltip="Name of the variable to retrieve",
        )
        self.add_parameter(self.variable_name_param)

        self.value_param = Parameter(
            name="value",
            type=ParameterTypeBuiltin.ALL.value,
            allowed_modes={ParameterMode.OUTPUT},
            tooltip="The value of the workflow variable",
        )
        self.add_parameter(self.value_param)

        # Advanced parameters group (collapsed by default)
        advanced = create_advanced_parameter_group()
        self.scope_param = advanced.scope_param
        self.add_node_element(advanced.parameter_group)

        self._refresh_variable_name_choices()

    def _refresh_variable_name_choices(self) -> None:
        """Refresh the dropdown choices for variable_name with currently available variables."""
        try:
            scope_str = self.get_parameter_value(self.scope_param.name)
            scope = scope_string_to_variable_scope(scope_str)
            variable_names = list_variable_names(node_name=self.name, scope=scope)
        except Exception:
            logger.debug("Could not list variables for dropdown refresh", exc_info=True)
            variable_names = []

        # Include the current value so it remains valid even if not in the listed variables
        current_value = self.get_parameter_value(self.variable_name_param.name)
        if current_value and current_value not in variable_names:
            variable_names.append(current_value)

        # Set dropdown via ui_options directly (not the Options trait) to allow custom input
        self.variable_name_param.update_ui_options_key("simple_dropdown", variable_names)

    def before_value_set(self, parameter: Parameter, value: Any) -> Any:
        """Refresh the variable name dropdown when the scope or variable_name changes."""
        if parameter == self.variable_name_param or parameter == self.scope_param:
            self._refresh_variable_name_choices()
        return value

    def process(self) -> None:
        variable_name = self.get_parameter_value(self.variable_name_param.name)
        scope_str = self.get_parameter_value(self.scope_param.name)

        # Convert scope string to VariableScope enum
        scope = scope_string_to_variable_scope(scope_str)

        # This can throw if the variable doesn't exist.
        variable = get_variable(node_name=self.name, variable_name=variable_name, scope=scope)

        var_value = variable.value

        self.set_parameter_value(self.value_param.name, var_value)

        variable_name = self.get_parameter_value(self.variable_name_param.name)

        # Set the output values.
        self.parameter_output_values[self.variable_name_param.name] = variable_name
        self.parameter_output_values[self.value_param.name] = var_value

    @property
    def state(self) -> NodeResolutionState:
        """Overrides BaseNode.state @property to treat it as volatile (if the value has changed, mark as unresolved)."""
        if self._state == NodeResolutionState.RESOLVED:
            variable_name = self.get_parameter_value(self.variable_name_param.name)
            scope_str = self.get_parameter_value(self.scope_param.name)

            # Convert scope string to VariableScope enum
            scope = scope_string_to_variable_scope(scope_str)

            # This can throw if the variable doesn't exist.
            try:
                variable = get_variable(node_name=self.name, variable_name=variable_name, scope=scope)

                var_value = variable.value
                our_value = self.get_parameter_value(self.value_param.name)

                # Mark ourselves as unresolved if the value is different than what we last emitted.
                if var_value != our_value:
                    return NodeResolutionState.UNRESOLVED
            except LookupError:
                # Variable may not have been created yet; assume unresolved.
                return NodeResolutionState.UNRESOLVED
        return super().state

    @state.setter
    def state(self, new_state: NodeResolutionState) -> None:
        # Have to override the setter if we override the getter.
        self._state = new_state
