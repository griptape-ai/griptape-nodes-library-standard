from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import ControlNode

from griptape_nodes_library.llm.agent_state import AgentState


class ClearAgentMemory(ControlNode):
    """ClearAgentMemory Node that clears an agent's conversation memory."""

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)

        self.agent = Parameter(
            name="agent",
            tooltip="Agent to clear memory for",
            input_types=["Agent"],
            allowed_modes={ParameterMode.INPUT, ParameterMode.OUTPUT},
        )

        self.add_parameter(self.agent)

    def process(self) -> None:
        agent_value = self.get_parameter_value("agent")
        if agent_value is None:
            return

        state = AgentState.from_wire(agent_value)
        state.messages = []

        updated = state.to_wire()
        self.parameter_output_values["agent"] = updated
        self.publish_update_to_parameter("agent", updated)
