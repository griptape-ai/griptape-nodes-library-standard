from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import ControlNode
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString

from griptape_nodes_library.llm.agent_node_support import default_cloud_model_config
from griptape_nodes_library.llm.agent_state import AgentState, messages_from_runs
from griptape_nodes_library.llm.runner import output_to_text, prompt_model


class SummarizeAgentMemory(ControlNode):
    """SummarizeMemory Node that summarizes an agent's conversation memory and replaces it with a single summary run."""

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)

        self.agent = Parameter(
            name="agent",
            tooltip="Agent to summarize memory for",
            input_types=["Agent"],
            allowed_modes={ParameterMode.INPUT, ParameterMode.OUTPUT},
        )

        self.add_parameter(self.agent)

        self.prompt = ParameterString(
            name="prompt",
            tooltip="The prompt to use to summarize the agent's conversation memory",
            multiline=True,
            hide=True,
            default_value="Summarize our conversation. Include specific and useful details about the conversation, but only output only the summary, no other text. Do not include this exchange as part of that summary.",
        )
        self.add_parameter(self.prompt)

        self.summary = ParameterString(
            name="summary",
            tooltip="The summary of the agent's conversation memory",
            multiline=True,
            allow_input=False,
            allow_property=False,
        )

        self.add_parameter(self.summary)

    def process(self) -> None:
        agent_value = self.get_parameter_value("agent")
        if agent_value is None:
            return

        state = AgentState.from_wire(agent_value)
        runs = state.runs()
        if not runs:
            updated = state.to_wire()
            self.parameter_output_values["agent"] = updated
            self.publish_update_to_parameter("agent", updated)
            return

        prompt = self.get_parameter_value("prompt")
        summary_text = output_to_text(
            prompt_model(
                state.model or default_cloud_model_config(),
                prompt,
                rulesets=state.rulesets,
                message_history=messages_from_runs(runs),
            )
        )

        self.parameter_output_values["summary"] = summary_text
        self.publish_update_to_parameter("summary", summary_text)

        updated = state.with_runs([{"input": "conversation summary", "output": summary_text}]).to_wire()
        self.parameter_output_values["agent"] = updated
        self.publish_update_to_parameter("agent", updated)
