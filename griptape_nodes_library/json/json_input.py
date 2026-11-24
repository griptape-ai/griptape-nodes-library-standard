from typing import Any

from griptape_nodes.exe_types.core_types import (
    Parameter,
    ParameterMode,
)
from griptape_nodes.exe_types.node_types import DataNode

from griptape_nodes_library.json.json_utils import to_json_value


class JsonInput(DataNode):
    """Create a JSON node."""

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)

        # Add a parameter for a list of keys
        self.add_parameter(
            Parameter(
                name="json",
                input_types=["json", "str", "dict"],
                type="json",
                default_value="{}",
                tooltip="Json Data",
                allowed_modes={ParameterMode.PROPERTY, ParameterMode.OUTPUT},
            )
        )

    def process(self) -> None:
        json_data = self.get_parameter_value("json")
        result = to_json_value(self.name, json_data)
        self.parameter_output_values["json"] = result
