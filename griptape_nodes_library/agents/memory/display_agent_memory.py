import json
from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import ControlNode
from griptape_nodes.exe_types.param_types.parameter_json import ParameterJson

from griptape_nodes_library.utils.agent_utils import unwrap_agent


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

    def _get_memory_dict(self) -> dict[str, Any] | None:
        """Get memory directly from the agent wire dict without full reconstruction."""
        agent_value = self.get_parameter_value("agent")
        if agent_value is None:
            return None

        agent_core_dict, _, _ = unwrap_agent(agent_value)
        memory = agent_core_dict.get("conversation_memory")
        if not isinstance(memory, dict):
            return None
        return memory

    def _extract_value_from_artifact(self, artifact: Any) -> str:
        """Extract value from an artifact (dict with 'value' key or string)."""
        if isinstance(artifact, dict):
            value = artifact.get("value", "")
            if not isinstance(value, str):
                return json.dumps(value)
            return value
        if isinstance(artifact, str):
            return artifact
        return ""

    def _transform_runs(self, memory: dict[str, Any]) -> list[dict[str, str]]:
        """Transform memory runs to extract just input/output values."""
        transformed_runs = []
        if "runs" not in memory:
            return transformed_runs

        runs = memory["runs"]
        if not isinstance(runs, list):
            return transformed_runs

        for run in runs:
            if not isinstance(run, dict):
                continue

            input_value = self._extract_value_from_artifact(run.get("input", ""))
            output_value = self._extract_value_from_artifact(run.get("output", ""))

            transformed_runs.append({"input": input_value, "output": output_value})

        return transformed_runs

    def _set_output(self, transformed_memory: dict[str, Any]) -> None:
        """Set the output parameter value."""
        self.parameter_output_values["memory"] = transformed_memory
        self.publish_update_to_parameter("memory", transformed_memory)

    def process(self) -> None:
        # Pass the agent wire through unchanged
        agent_value = self.get_parameter_value("agent")
        if agent_value is not None:
            self.parameter_output_values["agent"] = agent_value

        memory = self._get_memory_dict()
        if memory is None:
            self._set_output({"runs": []})
            return

        transformed_runs = self._transform_runs(memory)
        self._set_output({"runs": transformed_runs})
