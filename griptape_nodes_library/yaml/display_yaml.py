from typing import Any

from griptape_nodes.exe_types.node_types import DataNode
from griptape_nodes.exe_types.param_types.parameter_yaml import ParameterYaml


class DisplayYaml(DataNode):
    """Create a YAML Display node."""

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)

        self.add_parameter(
            ParameterYaml(
                name="yaml",
                tooltip="YAML Data",
                default_value="",
                placeholder_text="Enter YAML data here",
            )
        )

    def process(self) -> None:
        pass
