from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode, ParameterTypeBuiltin

from griptape_nodes_library.execution.base_pass_through import BasePassThroughNode


class DotNode(BasePassThroughNode):
    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)

        self.value_param = Parameter(
            name="value",
            input_types=[ParameterTypeBuiltin.ANY.value],
            output_type=ParameterTypeBuiltin.ALL.value,
            default_value=None,
            tooltip="Pass-through value",
            allowed_modes={ParameterMode.INPUT, ParameterMode.OUTPUT},
            hide_property=True,
        )
        self.add_parameter(self.value_param)

    def get_pass_thru_parameter(self) -> Parameter:
        return self.value_param
