from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import BaseNode, DataNode

from griptape_nodes_library.json.json_utils import to_json_value


class ToJson(DataNode):
    def __init__(
        self,
        name: str,
        metadata: dict[Any, Any] | None = None,
        value: str = "",
    ) -> None:
        super().__init__(name, metadata)

        self.add_parameter(
            Parameter(
                name="from",
                default_value=value,
                input_types=["any"],
                tooltip="The data to convert",
                allowed_modes={ParameterMode.INPUT},
            )
        )
        self.add_parameter(
            Parameter(
                name="output",
                default_value=value,
                output_type="json",
                type="json",
                tooltip="The converted data as json",
                hide_property=True,
                allowed_modes={ParameterMode.OUTPUT, ParameterMode.PROPERTY},
            )
        )

    def after_incoming_connection(
        self,
        source_node: BaseNode,
        source_parameter: Parameter,
        target_parameter: Parameter,
    ) -> None:
        pass

    def _convert_to_json(self) -> None:
        """Convert the input value to JSON and update output."""
        input_value = self.get_parameter_value("from")
        result = to_json_value(self.name, input_value)
        self.parameter_output_values["output"] = result

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        if parameter.name == "from":
            try:
                self._convert_to_json()
            except ValueError:
                # Report the error when the node runs, not while the input is being edited or propagated.
                self.parameter_output_values["output"] = None
        return super().after_value_set(parameter, value)

    def process(self) -> None:
        """Process the node during execution."""
        self._convert_to_json()
