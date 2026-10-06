from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import BaseNode, ControlNode
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.traits.options import Options

from griptape_nodes_library.llm.agent_state import AgentState


class ReplaceItemInAgentMemory(ControlNode):
    """ReplaceItemInAgentMemory Node that replaces an item in the memory of an agent."""

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)

        self.agent = Parameter(
            name="agent",
            tooltip="Agent to replace item in memory of",
            input_types=["Agent"],
            allowed_modes={ParameterMode.INPUT, ParameterMode.OUTPUT},
        )

        self.add_parameter(self.agent)

        self.memory_to_replace = ParameterString(
            name="memory_to_replace",
            tooltip="The memory to replace. Connect an agent to the node to see the available memories.",
            default_value=None,
            traits={Options(choices=["No memories available"])},
            hide=True,
        )
        self.add_parameter(self.memory_to_replace)

        self.orig_output = ParameterString(
            name="orig_output",
            tooltip="The original output of the memory item to replace",
            multiline=True,
            allow_input=False,
            allow_property=False,
            hide=True,
        )
        self.add_parameter(self.orig_output)

        self.new_input = ParameterString(
            name="new_input",
            tooltip="The new input of the memory item to replace",
            multiline=True,
            hide=True,
            placeholder_text="The new input of the memory item to replace. Leave blank to use the original input.",
        )
        self.add_parameter(self.new_input)
        self.new_output = ParameterString(
            name="new_output",
            tooltip="The new output of the memory item to replace",
            multiline=True,
            hide=True,
            placeholder_text="The new output of the memory item to replace. Leave blank to use the original output.",
        )
        self.add_parameter(self.new_output)

    def _get_state(self) -> AgentState | None:
        """Get the agent state from the parameter value, returning None if unavailable."""
        agent_value = self.get_parameter_value("agent")
        if agent_value is None:
            return None
        return AgentState.from_wire(agent_value)

    def _format_memory_choice(self, index: int, input_value: str, max_length: int = 60) -> str:
        """Format a memory choice with index and truncated input context."""
        if not input_value:
            return f"{index}: (empty input)"
        truncated = input_value[:max_length]
        if len(input_value) > max_length:
            truncated += "..."
        return f"{index}: {truncated}"

    def _update_memory_choices(self) -> None:
        """Update the memory_to_replace dropdown with available memory runs."""
        state = self._get_state()
        runs = state.runs() if state is not None else []
        if not runs:
            self._update_option_choices(
                param="memory_to_replace",
                choices=["No memories available"],
                default="No memories available",
            )
            return

        # Get current selection to preserve it if still valid
        current_choice = self.get_parameter_value("memory_to_replace")
        current_index = None
        if current_choice and current_choice != "No memories available":
            current_index = self._extract_index_from_choice(current_choice)

        choices = []
        for i, run in enumerate(runs):
            choices.append(self._format_memory_choice(i, run["input"]))

        if choices:
            # Preserve current selection if it's still valid, otherwise use first choice
            default_choice = choices[0]
            if current_index is not None and current_index < len(choices):
                # Check if the choice at current_index matches (same index)
                for choice in choices:
                    choice_index = self._extract_index_from_choice(choice)
                    if choice_index == current_index:
                        default_choice = choice
                        break
            self._update_option_choices(param="memory_to_replace", choices=choices, default=default_choice)
        else:
            self._update_option_choices(
                param="memory_to_replace",
                choices=["No memories available"],
                default="No memories available",
            )

    def after_incoming_connection(
        self, source_node: BaseNode, source_parameter: Parameter, target_parameter: Parameter
    ) -> None:
        if target_parameter.name == "agent":
            self.show_parameter_by_name(["memory_to_replace", "orig_output", "new_input", "new_output"])
            # Try to update choices, but value might not be set yet
            self._update_memory_choices()
        return super().after_incoming_connection(source_node, source_parameter, target_parameter)

    def after_incoming_connection_removed(
        self, source_node: BaseNode, source_parameter: Parameter, target_parameter: Parameter
    ) -> None:
        if target_parameter.name == "agent":
            self.hide_parameter_by_name(["memory_to_replace", "orig_output", "new_input", "new_output"])
        return super().after_incoming_connection_removed(source_node, source_parameter, target_parameter)

    def _extract_index_from_choice(self, choice: str) -> int | None:
        """Extract the run index from a formatted choice string like '0: Hey, how's it going?'."""
        if not choice or choice == "No memories available":
            return None
        try:
            index_str = choice.split(":", 1)[0].strip()
            return int(index_str)
        except (ValueError, IndexError):
            return None

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        if parameter.name == "agent":
            # When agent value is set, update the memory choices
            self._update_memory_choices()

        if parameter.name == "memory_to_replace":
            choice = self.get_parameter_value("memory_to_replace")
            if choice is None:
                return super().after_value_set(parameter, value)

            index = self._extract_index_from_choice(choice)
            if index is None:
                return super().after_value_set(parameter, value)

            state = self._get_state()
            runs = state.runs() if state is not None else []
            if index < 0 or index >= len(runs):
                return super().after_value_set(parameter, value)

            output_value = runs[index]["output"]
            self.parameter_output_values["orig_output"] = output_value
            self.publish_update_to_parameter("orig_output", output_value)

        return super().after_value_set(parameter, value)

    def _set_output(self, transformed_memory: dict[str, Any]) -> None:
        """Set the output parameter value."""
        self.parameter_output_values["memory"] = transformed_memory
        self.publish_update_to_parameter("memory", transformed_memory)

    def process(self) -> None:
        state = self._get_state()
        if state is None:
            return

        choice = self.get_parameter_value("memory_to_replace")
        if choice is None or choice == "No memories available":
            return

        index = self._extract_index_from_choice(choice)
        if index is None:
            return

        runs = state.runs()
        if index < 0 or index >= len(runs):
            return

        new_input_value = self.get_parameter_value("new_input")
        new_output_value = self.get_parameter_value("new_output")

        # Use original values if new values are blank (None or empty string)
        if new_input_value is None or (isinstance(new_input_value, str) and new_input_value.strip() == ""):
            input_value = runs[index]["input"]
        else:
            input_value = new_input_value

        if new_output_value is None or (isinstance(new_output_value, str) and new_output_value.strip() == ""):
            output_value = runs[index]["output"]
        else:
            output_value = new_output_value

        runs[index] = {"input": input_value, "output": output_value}
        updated = state.with_runs(runs).to_wire()
        self.parameter_output_values["agent"] = updated
        self.publish_update_to_parameter("agent", updated)
