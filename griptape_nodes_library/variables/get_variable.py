from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode, ParameterTypeBuiltin
from griptape_nodes.exe_types.node_types import VariableAccess

from griptape_nodes_library.variables.base_variable_node import BaseVariableNode
from griptape_nodes_library.variables.variable_utils import get_variable


class GetVariable(BaseVariableNode):
    # process() only calls GetVariableRequest; it never writes.
    VARIABLE_ACCESS = VariableAccess.READ

    def __init__(
        self,
        name: str,
        metadata: dict[Any, Any] | None = None,
    ) -> None:
        super().__init__(name, metadata)

        self._add_variable_name_parameter(
            Parameter(
                name="variable_name",
                type="str",
                allowed_modes={ParameterMode.INPUT, ParameterMode.OUTPUT, ParameterMode.PROPERTY},
                tooltip="Name of the variable to retrieve",
            )
        )

        self.value_param = Parameter(
            name="value",
            type=ParameterTypeBuiltin.ALL.value,
            allowed_modes={ParameterMode.OUTPUT},
            tooltip="The value of the workflow variable",
        )
        self.add_parameter(self.value_param)

        self._add_scope_parameter()

    def process(self) -> None:
        variable_name = self.get_parameter_value(self.variable_name_param.name)

        # This can throw if the variable doesn't exist.
        variable = get_variable(node_name=self.name, variable_name=variable_name, scope=self._get_scope())
        var_value = variable.value

        self.set_parameter_value(self.value_param.name, var_value)

        # Set the output values.
        self.parameter_output_values[self.variable_name_param.name] = variable_name
        self.parameter_output_values[self.value_param.name] = var_value

    def _is_stale(self) -> bool:
        # Stale if the variable's value differs from what we last emitted.
        variable_name = self.get_parameter_value(self.variable_name_param.name)
        variable = get_variable(node_name=self.name, variable_name=variable_name, scope=self._get_scope())
        return variable.value != self.get_parameter_value(self.value_param.name)
