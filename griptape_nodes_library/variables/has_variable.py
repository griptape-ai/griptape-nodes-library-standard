from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import VariableAccess

from griptape_nodes_library.variables.base_variable_node import BaseVariableNode
from griptape_nodes_library.variables.variable_utils import has_variable


class HasVariable(BaseVariableNode):
    # process() only calls HasVariableRequest; the existence check is a read with no side effects.
    VARIABLE_ACCESS = VariableAccess.READ

    def __init__(
        self,
        name: str,
        metadata: dict[Any, Any] | None = None,
    ) -> None:
        super().__init__(name, metadata)

        # Unlike the other variable nodes, HasVariable intentionally does NOT use
        # _add_variable_name_parameter's Options dropdown of existing variables: the whole point of
        # the node is to test whether a variable exists, so it must accept arbitrary names —
        # including ones that don't exist yet — typed directly or driven by a connection.
        self.variable_name_param = Parameter(
            name="variable_name",
            type="str",
            allowed_modes={ParameterMode.INPUT, ParameterMode.OUTPUT, ParameterMode.PROPERTY},
            tooltip="Name of the variable to check for existence",
        )
        self.add_parameter(self.variable_name_param)

        self.exists_param = Parameter(
            name="exists",
            type="bool",
            default_value=False,
            allowed_modes={ParameterMode.OUTPUT},
            tooltip="Whether the workflow variable exists",
        )
        self.add_parameter(self.exists_param)

        self._add_scope_parameter()

    def process(self) -> None:
        variable_name = self.get_parameter_value(self.variable_name_param.name)

        # This can throw.
        exists = has_variable(node_name=self.name, variable_name=variable_name, scope=self._get_scope())

        self.set_parameter_value(self.exists_param.name, exists)

        # Set output values.
        self.parameter_output_values[self.exists_param.name] = exists
        self.parameter_output_values[self.variable_name_param.name] = variable_name

    def _is_stale(self) -> bool:
        # Stale if the variable's existence differs from what we last emitted.
        variable_name = self.get_parameter_value(self.variable_name_param.name)
        var_exists = has_variable(node_name=self.name, variable_name=variable_name, scope=self._get_scope())
        return var_exists != self.get_parameter_value(self.exists_param.name)
