from typing import Any

from griptape_nodes.exe_types.core_types import ParameterMode
from griptape_nodes.exe_types.node_types import DataNode
from griptape_nodes.exe_types.param_types.parameter_xml import ParameterXml


class XmlInput(DataNode):
    """Create an XML node."""

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)

        self.add_parameter(
            ParameterXml(
                name="xml",
                tooltip="XML Data",
                allowed_modes={ParameterMode.PROPERTY, ParameterMode.OUTPUT},
            )
        )

    def process(self) -> None:
        self.parameter_output_values["xml"] = self.get_parameter_value("xml")
