from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import ControlNode
from griptape_nodes.exe_types.param_types.parameter_json import ParameterJson

from griptape_nodes_library.llm.agent_state import AgentState


class DisplayAgentMemory(ControlNode):
    """DisplayAgentMemory Node that displays the memory of an agent."""

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)

        self.agent = Parameter(
            name="agent",
            tooltip="Agent to extract memory from",
            input_types=["Agent"],
            allowed_modes={ParameterMode.INPUT, ParameterMode.OUTPUT},
        )

        self.add_parameter(self.agent)
        self.memory_json = ParameterJson(
            name="memory",
            tooltip="The agent's memory as a JSON object",
            allowed_modes={ParameterMode.OUTPUT},
        )
        self.add_parameter(self.memory_json)

    def _set_output(self, transformed_memory: dict[str, Any]) -> None:
        """Set the output parameter value."""
        self.parameter_output_values["memory"] = transformed_memory
        self.publish_update_to_parameter("memory", transformed_memory)

    def process(self) -> None:
        # Pass the agent wire through unchanged
        agent_value = self.get_parameter_value("agent")
        if agent_value is None:
            self._set_output({"runs": []})
            return

        self.parameter_output_values["agent"] = agent_value
        self._set_output({"runs": AgentState.from_wire(agent_value).runs()})
