from typing import Any

from griptape_nodes.exe_types.core_types import ParameterMode
from griptape_nodes.exe_types.node_types import DataNode
from griptape_nodes.exe_types.param_types.parameter_html import ParameterHtml


class HtmlInput(DataNode):
    """Create an HTML node."""

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)

        self.add_parameter(
            ParameterHtml(
                name="html",
                tooltip="HTML Data",
                allowed_modes={ParameterMode.PROPERTY, ParameterMode.OUTPUT},
            )
        )

    def process(self) -> None:
        self.parameter_output_values["html"] = self.get_parameter_value("html")
