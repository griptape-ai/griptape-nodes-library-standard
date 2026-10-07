from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import ControlNode
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString

from griptape_nodes_library.utils.agent_runner import format_output
from griptape_nodes_library.utils.agent_state import AgentState
from griptape_nodes_library.utils.local_agent_runner import LocalAgentRunner


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
        state = AgentState.from_wire(self.get_parameter_value("agent"))
        if state is None:
            return

        if state.messages:
            run = LocalAgentRunner().run(state, [self.get_parameter_value("prompt")])
            summary_text = format_output(run.output)
            self.parameter_output_values["summary"] = summary_text
            self.publish_update_to_parameter("summary", summary_text)
            state.replace_history("conversation summary", summary_text)

        updated = state.to_wire()
        self.parameter_output_values["agent"] = updated
        self.publish_update_to_parameter("agent", updated)
