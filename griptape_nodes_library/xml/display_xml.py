from typing import Any

from griptape_nodes.exe_types.node_types import DataNode
from griptape_nodes.exe_types.param_types.parameter_xml import ParameterXml


class DisplayXml(DataNode):
    """Display an XML node."""

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)

        self.add_parameter(
            ParameterXml(
                name="xml",
                tooltip="XML Data",
                default_value="",
            )
        )

    def process(self) -> None:
        pass
