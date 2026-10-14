from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import DataNode

from griptape_nodes_library.llm.agent_state import AgentState
from griptape_nodes_library.llm.tools import ToolType

placeholder_description = "Tool created from an Agent"


class AgentToTool(DataNode):
    """Convert an Agent to a Tool that can be used by other Agents."""

    def __init__(
        self,
        name: str,
        metadata: dict[Any, Any] | None = None,
    ) -> None:
        super().__init__(name, metadata)

        self.add_parameter(
            Parameter(
                name="agent",
                type="Agent",
                input_types=["Agent"],
                tooltip="None",
                default_value=None,
                allowed_modes={ParameterMode.INPUT},
            )
        )

        # Name parameter for the Tool
        self.add_parameter(
            Parameter(
                name="name",
                input_types=["str"],
                type="str",
                allowed_modes={ParameterMode.PROPERTY, ParameterMode.INPUT},
                default_value="Agent Tool",
                tooltip="Name for the Tool",
            )
        )

        def validate_tool_description(_param: Parameter, value: str) -> None:
            if not value:
                msg = f"{self.name} : A meaningful description is critical for an Agent to know when to use this tool."
                raise ValueError(msg)

        # Description parameter for the Tool
        self.add_parameter(
            Parameter(
                name="description",
                input_types=["str"],
                type="str",
                allowed_modes={ParameterMode.PROPERTY, ParameterMode.INPUT},
                ui_options={
                    "multiline": True,
                    "placeholder_text": "Description for what the Tool does",
                },
                tooltip="Description for what the Tool does",
                validators=[validate_tool_description],
            )
        )

        self.add_parameter(
            Parameter(
                name="off_prompt",
                input_types=["bool"],
                type="bool",
                output_type="bool",
                default_value=False,
                tooltip="Ignored. Kept so saved workflows load.",
                ui_options={"hide": True},
            )
        )
        self.add_parameter(
            Parameter(
                name="tool",
                input_types=["Tool"],
                type="Tool",
                output_type="Tool",
                default_value=None,
                allowed_modes={ParameterMode.OUTPUT},
                tooltip="",
            )
        )

    def process(self) -> None:
        """Convert the Agent to a Tool."""
        params = self.parameter_values
        agent = params.get("agent")
        name = params.get("name", "Agent Tool")
        description = params.get("description", placeholder_description)
        off_prompt = params.get("off_prompt", True)

        if agent is None:
            self.parameter_output_values["tool"] = None
            return

        # Store the agent wire (normalized, so older agent formats and their secrets don't leak
        # through). `build_toolsets` rebuilds a runnable agent tool from it at run time.
        self.parameter_output_values["tool"] = {
            "tool_type": ToolType.AGENT_TOOL.value,
            "agent_dict": AgentState.from_wire(agent).to_wire(),
            "name": name,
            "description": description,
            "off_prompt": off_prompt,
        }
