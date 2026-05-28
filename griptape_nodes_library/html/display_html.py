from typing import Any

from griptape_nodes.exe_types.node_types import DataNode
from griptape_nodes.exe_types.param_types.parameter_html import ParameterHtml


class DisplayHtml(DataNode):
    """Display an HTML node."""

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)

        self.add_parameter(
            ParameterHtml(
                name="html",
                tooltip="HTML Data",
                default_value="",
                placeholder_text="Enter HTML data here",
            )
        )

    def process(self) -> None:
        pass
