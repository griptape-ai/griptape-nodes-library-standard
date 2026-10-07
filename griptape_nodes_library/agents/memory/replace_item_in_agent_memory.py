from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import BaseNode, ControlNode
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.traits.options import Options

from griptape_nodes_library.utils.agent_state import AgentState, ConversationTurn

NO_MEMORIES = "No memories available"


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
            traits={Options(choices=[NO_MEMORIES])},
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
        """Read the connected agent, or ``None`` when nothing usable is connected."""
        return AgentState.from_wire(self.get_parameter_value("agent"))

    def _turns(self) -> list[ConversationTurn]:
        state = self._get_state()
        return state.turns() if state is not None else []

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
        choices = [self._format_memory_choice(i, turn.prompt) for i, turn in enumerate(self._turns())]
        if not choices:
            self._update_option_choices(param="memory_to_replace", choices=[NO_MEMORIES], default=NO_MEMORIES)
            return

        # Preserve the current selection if its index still exists, otherwise use the first choice.
        current_index = self._extract_index_from_choice(self.get_parameter_value("memory_to_replace"))
        default_choice = (
            choices[current_index] if current_index is not None and current_index < len(choices) else choices[0]
        )
        self._update_option_choices(param="memory_to_replace", choices=choices, default=default_choice)

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

    def _extract_index_from_choice(self, choice: str | None) -> int | None:
        """Extract the run index from a formatted choice string like '0: Hey, how's it going?'."""
        if not choice or choice == NO_MEMORIES:
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
            index = self._extract_index_from_choice(self.get_parameter_value("memory_to_replace"))
            turns = self._turns()
            if index is not None and 0 <= index < len(turns):
                self.parameter_output_values["orig_output"] = turns[index].response
                self.publish_update_to_parameter("orig_output", turns[index].response)

        return super().after_value_set(parameter, value)

    def _set_output(self, transformed_memory: dict[str, Any]) -> None:
        """Set the output parameter value."""
        self.parameter_output_values["memory"] = transformed_memory
        self.publish_update_to_parameter("memory", transformed_memory)

    def process(self) -> None:
        state = self._get_state()
        if state is None:
            return

        index = self._extract_index_from_choice(self.get_parameter_value("memory_to_replace"))
        if index is None or not 0 <= index < len(state.turns()):
            return

        # Blank new values keep the original side of the turn.
        new_input = self.get_parameter_value("new_input")
        new_output = self.get_parameter_value("new_output")
        state.replace_turn(
            index,
            prompt=new_input if isinstance(new_input, str) and new_input.strip() else None,
            response=new_output if isinstance(new_output, str) and new_output.strip() else None,
        )

        updated = state.to_wire()
        self.parameter_output_values["agent"] = updated
        self.publish_update_to_parameter("agent", updated)
