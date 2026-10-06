from typing import Literal

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import DataNode

VariantType = Literal["info", "warning", "error", "success", "tip", "note", "help", "docs", "link", "cloud-upload"]


class BaseTool(DataNode):
    """Base tool node for creating agent tools.

    This node provides a generic implementation for emitting agent tool configs with configurable parameters.

    Attributes:
        off_prompt (bool): Accepted for compatibility with saved workflows; ignored.
        tool (BaseTool): A dictionary representation of the created tool.
    """

    def __init__(self, name: str, metadata: dict | None = None) -> None:
        super().__init__(name, metadata)

        tool_param = Parameter(
            name="tool",
            input_types=["Tool"],
            type="Tool",
            output_type="Tool",
            default_value=None,
            allowed_modes={ParameterMode.OUTPUT},
            tooltip="Connect this to an Agent's tools input.",
        )
        tool_param.set_badge(
            variant="info",
            title="Tool",
            message="This tool can be provided to an agent to help it perform tasks.",
            hide_clear_button=True,
        )
        self.add_parameter(tool_param)

        self.add_parameter(
            Parameter(
                name="off_prompt",
                input_types=["bool"],
                type="bool",
                output_type="bool",
                default_value=False,
                tooltip="",
            )
        )

    def update_tool_info(self, value: str = "", title: str = "", variant: VariantType = "info") -> None:
        tool_param = self.get_parameter_by_name("tool")
        if tool_param is None:
            return
        tool_param.set_badge(
            variant=variant,
            title=title or None,
            message=value or None,
            hide_clear_button=True,
        )

    def process(self) -> None:
        # The base node has no tool of its own; subclasses emit a `{"tool_type": ...}` config.
        self.parameter_output_values["tool"] = None
