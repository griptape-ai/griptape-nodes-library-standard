from typing import Any

from griptape_nodes.exe_types.core_types import (
    ParameterMode,
)
from griptape_nodes.exe_types.node_types import DataNode
from griptape_nodes.exe_types.param_types.parameter_yaml import ParameterYaml
from ruamel.yaml import YAML

_yaml = YAML()


class YamlInput(DataNode):
    """Create a YAML node."""

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)

        self.add_parameter(
            ParameterYaml(
                name="yaml",
                tooltip="Yaml Data",
                allowed_modes={ParameterMode.PROPERTY, ParameterMode.OUTPUT},
            )
        )

    def process(self) -> None:
        yaml_data = self.get_parameter_value("yaml")
        self.parameter_output_values["yaml"] = yaml_data
