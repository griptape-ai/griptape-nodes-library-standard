"""Defines the Agent node: chat with an LLM agent, with text and images, and pass it on.

The node creates an agent, or continues one connected from another node, runs it on the
prompt (plus any connected images) and outputs the reply and the updated agent. The agent
travels between nodes as an :class:`~griptape_nodes_library.utils.agent_state.AgentState`
and runs through an :class:`~griptape_nodes_library.utils.agent_runner.AgentRunner`.
"""

import json
from collections.abc import Generator
from typing import Any

from griptape.drivers.prompt.base_prompt_driver import BasePromptDriver
from griptape.drivers.prompt.griptape_cloud import GriptapeCloudPromptDriver
from griptape.drivers.prompt.ollama import OllamaPromptDriver
from griptape_nodes.drivers.cloud_models import MODEL_CHOICES, ProviderID
from griptape_nodes.exe_types.core_types import (
    NodeMessageResult,
    Parameter,
    ParameterGroup,
    ParameterList,
    ParameterMode,
    ParameterType,
)
from griptape_nodes.exe_types.node_types import BaseNode, ControlNode
from griptape_nodes.exe_types.param_components.model_access_component import ModelAccessComponent
from griptape_nodes.exe_types.param_types.parameter_json import ParameterJson
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.retained_mode.events.connection_events import DeleteConnectionRequest
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes, logger
from griptape_nodes.traits.button import Button, ButtonDetailsMessagePayload
from griptape_nodes.traits.options import Options
from jinja2 import Template
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserContent, UserPromptPart

from griptape_nodes_library.utils.agent_runner import (
    AgentRunEvent,
    AgentRunOutput,
    TextDelta,
    ToolCalled,
    ToolReturned,
    format_output,
    image_prompt_content,
)
from griptape_nodes_library.utils.agent_state import AgentState, ProviderKind, ProviderRef, direct_provider_ref
from griptape_nodes_library.utils.agent_tools import griptape_tool_to_pydantic
from griptape_nodes_library.utils.agent_utils import build_tools, ruleset_to_config
from griptape_nodes_library.utils.cloud_credential_utils import (
    missing_credential_message,
    resolve_cloud_api_key,
)
from griptape_nodes_library.utils.cloud_legacy_models import CLOUD_LEGACY_MODEL_VALUES
from griptape_nodes_library.utils.local_agent_runner import LocalAgentRunner
from griptape_nodes_library.utils.model_invocation import require_model_invocation_sync
from griptape_nodes_library.utils.provider_selection_component import ProviderSelectionComponent

# --- Constants ---
API_KEY_ENV_VAR = "GT_CLOUD_API_KEY"
SERVICE = "Griptape"
DEFAULT_MODEL = "claude-sonnet-5"


class Agent(ControlNode):
    """Chat with an agent and stream its reply.

    The node creates a new agent, or continues one passed in on ``agent``, and outputs
    the agent with this turn added so the next node can carry the conversation on.

    Attributes:
        Inherits parameters and methods from ControlNode.
        Defines specific parameters for agent configuration (model, tools, rulesets),
        prompting, context, and output handling.
    """

    def __init__(self, **kwargs) -> None:
        """Initializes the ExampleAgent node, setting up its parameters and UI elements.

        This involves defining input/output parameters, grouping related settings,
        and establishing default values and behaviors.
        """
        super().__init__(**kwargs)

        # -- Converters --
        # Converters modify parameter values before they are used by the node's logic.
        def strip_whitespace(value: str) -> str:
            """Removes leading and trailing whitespace from a string value.

            Args:
                value: The input string.

            Returns:
                The string with whitespace stripped, or the original value if empty/None.
            """
            if not value:
                return value
            return value.strip()

        # --- Parameter Definitions ---

        # Parameter to input an existing agent's state or output the final state.
        self.add_parameter(
            Parameter(
                name="agent",
                type="Agent",
                input_types=["Agent"],
                output_type="Agent",
                tooltip="Create a new agent, or continue a chat with an existing agent.",
                default_value=None,
                allowed_modes={ParameterMode.INPUT, ParameterMode.OUTPUT},
            )
        )

        # Provider selector shown above the model dropdown. The ProviderSelectionComponent
        # installs the Options + refresh Button traits after construction.
        model_provider_param = Parameter(
            name="model_provider",
            type="str",
            default_value="griptape_cloud",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            tooltip="Choose a provider. Refresh to see all configured providers.",
            ui_options={"display_name": "provider"},
        )
        self.add_parameter(model_provider_param)

        # Model selector. The ModelAccessComponent installs the Options + refresh
        # Button traits, decorates each row with the caller's license entitlement
        # (denied Griptape Cloud models are flagged in the dropdown), and gates the
        # run at execute time. Choices update when the provider changes.
        model_param = Parameter(
            name="model",
            input_types=["str", "Prompt Model Config"],
            default_value=DEFAULT_MODEL,
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            tooltip="Choose a model, or connect a Prompt Model Configuration",
            ui_options={"display_name": "prompt model"},
        )
        self.add_parameter(model_param)
        self._model_access = ModelAccessComponent(
            node=self,
            parameter=model_param,
            model_choices=MODEL_CHOICES,
            default_model=DEFAULT_MODEL,
            deprecated_values=CLOUD_LEGACY_MODEL_VALUES,
        )

        self._provider = ProviderSelectionComponent(
            node=self,
            model_provider_param=model_provider_param,
            model_access=self._model_access,
            default_model=DEFAULT_MODEL,
        )

        self.add_parameter(
            ParameterJson(
                name="agent_memory",
                tooltip="The memory of the agent. Can be a simplified format with runs containing input/output, or full conversation_memory JSON.",
                default_value={},
                hide=True,
                hide_property=True,
                allowed_modes={ParameterMode.INPUT},
            )
        )
        # Main prompt input for the agent.
        self.add_parameter(
            ParameterString(
                name="prompt",
                tooltip="The main text prompt to send to the agent.",
                default_value="",
                multiline=True,
                placeholder_text="Talk with the Agent.",
                converters=[strip_whitespace],
                allow_output=False,
            )
        )

        self.add_parameter(
            ParameterList(
                name="images",
                input_types=[
                    "ImageUrlArtifact",
                    "ImageArtifact",
                    "str",
                    "list[ImageUrlArtifact]",
                    "list[ImageArtifact]",
                ],
                default_value=[],
                tooltip="Images to send with the prompt.",
                allowed_modes={ParameterMode.INPUT},
                collapsed=True,
                ui_options={"display_name": "image(s)"},
            )
        )

        # Optional additional context for the prompt.
        self.add_parameter(
            ParameterString(
                name="additional_context",
                tooltip=(
                    "Additional context to provide to the agent.\nEither a string, or dictionary of key-value pairs."
                ),
                default_value="",
                allow_output=False,
                ui_options={"placeholder_text": "Any additional context for the Agent."},
            )
        )

        self.add_parameter(
            ParameterList(
                name="tools",
                input_types=["Tool", "list[Tool]"],
                default_value=[],
                tooltip="Connect Griptape Tools for the agent to use.\nOr connect individual tools.",
                allowed_modes={ParameterMode.INPUT},
                collapsed=True,
            )
        )
        self.add_parameter(
            ParameterList(
                name="rulesets",
                input_types=["str", "Ruleset", "list[Ruleset]"],
                tooltip="Rulesets to apply to the agent to control its behavior.\nConnect Ruleset nodes, or connect a Text Input node to define behavior inline as plain text.",
                default_value=[],
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                collapsed=True,
                ui_options={
                    "placeholder_text": "e.g. Always respond in a friendly tone",
                    "child_prefix": "Behavior",
                    "display_name": "Behaviors (Rulesets)",
                },
            )
        )

        # Parameter for output schema
        self.add_parameter(
            Parameter(
                name="output_schema",
                input_types=["json"],
                type="json",
                tooltip="Optional JSON schema for structured output validation.",
                default_value=None,
                allowed_modes={ParameterMode.INPUT},
                hide_property=True,
            )
        )

        # Parameter for the agent's final text output.
        self.add_parameter(
            ParameterString(
                name="output",
                default_value="",
                tooltip="The final text response from the agent.",
                allowed_modes={ParameterMode.OUTPUT},
                ui_options={"multiline": True, "placeholder_text": "Agent response", "markdown": False},
            )
        )

        # Group for logging information.
        with ParameterGroup(name="Logs") as logs_group:
            Parameter(name="include_details", type="bool", default_value=False, tooltip="Include extra details.")

            Parameter(
                name="logs",
                type="str",
                tooltip="Displays processing logs and detailed events if enabled.",
                ui_options={"multiline": True, "placeholder_text": "Logs"},
                allowed_modes={ParameterMode.OUTPUT},
            )
        logs_group.ui_options = {"hide": True}  # Hide the logs group by default.

        self.add_node_element(logs_group)

    # --- Provider / Model Methods ---
    def _refresh_models_button(
        self, button: Button, button_details: ButtonDetailsMessagePayload
    ) -> NodeMessageResult | None:
        """Refresh the model dropdown for the currently selected provider."""
        return self._provider._refresh_models_button(button, button_details)

    # --- Helper Methods ---

    def _find_runs_in_data(self, data: Any) -> list[dict[str, Any]]:
        """Recursively find 'runs' array in data structure.

        Args:
            data: Any data structure (dict, list, etc.)

        Returns:
            List of run dicts, or empty list if not found.
        """
        if isinstance(data, dict):
            # Check if this dict has a 'runs' key
            if "runs" in data and isinstance(data["runs"], list):
                return data["runs"]

            # Recursively search in all values
            for value in data.values():
                result = self._find_runs_in_data(value)
                if result:
                    return result

        elif isinstance(data, list):
            # Recursively search in list items
            for item in data:
                result = self._find_runs_in_data(item)
                if result:
                    return result

        return []

    def _extract_value_from_artifact(self, artifact: Any) -> str:
        """Extract value from an artifact (dict with 'value' key, list of artifacts, or string).

        Args:
            artifact: Artifact data (dict, list, or string)

        Returns:
            Extracted string value
        """
        if isinstance(artifact, dict):
            return artifact.get("value", "")
        if isinstance(artifact, list):
            # If it's a list of artifacts, concatenate their values
            values = []
            for item in artifact:
                if isinstance(item, dict):
                    values.append(item.get("value", ""))
                else:
                    values.append(str(item) if item else "")
            return "\n".join(values)
        return str(artifact) if artifact else ""

    def _memory_to_messages(self, memory_data: dict[str, Any]) -> list[ModelMessage]:
        """Convert ``agent_memory`` input to conversation messages.

        Finds ``runs`` anywhere in the data (the simplified ``{"runs": [{"input", "output"}]}``
        format or a full griptape ``conversation_memory``) and turns each run into a
        prompt and a text reply.
        """
        runs_data = self._find_runs_in_data(memory_data)
        if not runs_data and "input" in memory_data and "output" in memory_data:
            runs_data = [memory_data]

        messages: list[ModelMessage] = []
        for run_data in runs_data:
            if not isinstance(run_data, dict) or "input" not in run_data or "output" not in run_data:
                continue
            messages.append(
                ModelRequest(parts=[UserPromptPart(content=self._extract_value_from_artifact(run_data["input"]))])
            )
            messages.append(
                ModelResponse(parts=[TextPart(content=self._extract_value_from_artifact(run_data["output"]))])
            )
        return messages

    def _parse_memory_data(self, memory_data: dict[str, Any] | str | None) -> dict[str, Any] | None:
        """Parse and validate memory data.

        Args:
            memory_data: Memory data dict, JSON string, or None.

        Returns:
            Parsed dict or None if invalid/empty.
        """
        if memory_data is None:
            return None

        # Handle string input (JSON that needs parsing)
        if isinstance(memory_data, str):
            if not memory_data.strip():
                return None
            try:
                memory_data = json.loads(memory_data)
            except json.JSONDecodeError:
                return None

        if not isinstance(memory_data, dict):
            return None

        # Skip empty dicts (default value)
        if not memory_data:
            return None

        return memory_data

    def _update_output_type_and_validate_connections(self, new_output_type: str) -> None:
        """Update the output parameter type and remove incompatible connections.

        Args:
            new_output_type: The new output type to set (e.g., "json" or "str")
        """
        output_param = self.get_parameter_by_name("output")
        if output_param is None:
            return

        # Update output parameter type
        output_param.output_type = new_output_type
        output_param.type = new_output_type

        # Get outgoing connections from the output parameter
        connections = GriptapeNodes.FlowManager().get_connections()
        outgoing_for_node = connections.outgoing_index.get(self.name, {})
        connection_ids = outgoing_for_node.get("output", [])

        # Validate type compatibility and remove incompatible connections
        for connection_id in connection_ids:
            connection = connections.connections[connection_id]
            target_param = connection.target_parameter
            target_node = connection.target_node

            # Check if target parameter accepts the new output type
            is_compatible = any(
                ParameterType.are_types_compatible(new_output_type, input_type)
                for input_type in target_param.input_types
            )

            if not is_compatible:
                logger.info(
                    f"Removing incompatible connection: Agent '{self.name}' output ({new_output_type}) to "
                    f"'{target_node.name}.{target_param.name}' (accepts: {target_param.input_types})"
                )

                # Remove the incompatible connection
                GriptapeNodes.handle_request(
                    DeleteConnectionRequest(
                        source_node_name=self.name,
                        source_parameter_name="output",
                        target_node_name=target_node.name,
                        target_parameter_name=target_param.name,
                    )
                )

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        super().after_value_set(parameter, value)
        self._model_access.on_value_set(parameter, value)
        if parameter.name == "model_provider":
            self._provider.on_provider_changed(str(value))

    # --- UI Interaction Hooks ---

    def after_incoming_connection(
        self, source_node: BaseNode, source_parameter: Parameter, target_parameter: Parameter
    ) -> None:
        # If an existing agent is connected, hide parameters related to creating a new one.
        if target_parameter.name == "agent":
            self._provider.hide()
            self.hide_parameter_by_name(["tools", "rulesets"])

        if target_parameter.name == "model" and source_parameter.name == "prompt_model_config":
            # Remove the options trait. Defensive guard so this stays idempotent instead
            # of raising IndexError.
            options_traits = target_parameter.find_elements_by_type(Options)
            if options_traits:
                target_parameter.remove_trait(trait_type=options_traits[0])

            # Check and see if the incoming connection is from a prompt model config or an agent.
            target_parameter.type = source_parameter.type

            # Remove ParameterMode.PROPERTY so it forces the node mark itself dirty & remove the value
            target_parameter.allowed_modes = {ParameterMode.INPUT}

            # Set the display name to be appropriate
            ui_options = target_parameter.ui_options
            ui_options["display_name"] = source_parameter.ui_options.get("display_name", source_parameter.name)
            target_parameter.ui_options = ui_options

        # If additional context is connected, prevent editing via property panel.
        # NOTE: This is a workaround. Ideally this is done automatically.
        if target_parameter.name == "additional_context":
            target_parameter.allowed_modes = {ParameterMode.INPUT}

        if target_parameter.name == "output_schema":
            # When schema is connected, change output type to json and validate connections
            self._update_output_type_and_validate_connections("json")

        # Hide the text field for connected non-string rulesets children to prevent [object Object].
        rulesets_param = self.get_parameter_by_name("rulesets")
        if rulesets_param and target_parameter in rulesets_param.children:
            if source_parameter.output_type in ("Ruleset", "list[Ruleset]"):
                target_parameter.hide_property = True

        return super().after_incoming_connection(source_node, source_parameter, target_parameter)

    def after_incoming_connection_removed(
        self,
        source_node: BaseNode,
        source_parameter: Parameter,
        target_parameter: Parameter,
    ) -> None:
        # If the agent connection is removed, show agent creation parameters.
        if target_parameter.name == "agent":
            self._provider.show()
            self.show_parameter_by_name(["tools", "rulesets", "schema"])

        if target_parameter.name == "output_schema":
            self.set_parameter_value("output_schema", None)
            # When schema is disconnected, change output type back to str and validate connections
            self._update_output_type_and_validate_connections("str")

        if target_parameter.name == "model":
            # Reset the parameter type and re-enable PROPERTY so the user can set it.
            target_parameter.type = "str"
            target_parameter.allowed_modes = {ParameterMode.INPUT, ParameterMode.PROPERTY}

            default_model = self._model_access.pick_permitted_default() or DEFAULT_MODEL
            target_parameter.set_default_value(default_model)
            target_parameter.default_value = default_model
            ui_options = target_parameter.ui_options
            ui_options["display_name"] = "prompt model"
            target_parameter.ui_options = ui_options
            self.set_parameter_value("model", default_model)
            # The connect hook stripped the Options trait; the component reinstalls
            # its Options trait, per-row license decoration, and badge.
            self._model_access.reinstall_options()

        # If the additional context connection is removed, make it editable again.
        # NOTE: This is a workaround. Ideally this is done automatically.
        if target_parameter.name == "additional_context":
            target_parameter.allowed_modes = {ParameterMode.INPUT, ParameterMode.PROPERTY}

        # Restore text field for disconnected rulesets children.
        rulesets_param = self.get_parameter_by_name("rulesets")
        if rulesets_param and target_parameter in rulesets_param.children:
            target_parameter.hide_property = False

        return super().after_incoming_connection_removed(source_node, source_parameter, target_parameter)

    # --- Validation ---
    def validate_before_workflow_run(self) -> list[Exception] | None:
        """Performs pre-run validation checks for the node.

        A Griptape Cloud credential is only required when the node would fall back
        to the default Griptape Cloud prompt driver. A connected agent carries its
        own driver, and a connected prompt driver (Prompt Model Config) supplies its
        own credentials, so neither needs the cloud credential.

        Either credential is accepted: a Griptape Nodes License or a Griptape Cloud
        API key. Griptape Cloud's chat endpoints authenticate a License, so a
        license-only user (no `GT_CLOUD_API_KEY` at all) must not be blocked here.

        Returns:
            A list of Exception objects if validation fails, otherwise None.
        """
        exceptions = []

        # Mirror the driver-selection precedence in process(): a connected agent or a
        # connected prompt driver bypass the default Griptape Cloud driver entirely.
        if not self._provider.uses_griptape_cloud_driver():
            return None

        # Check to see if either credential is set.
        api_key = resolve_cloud_api_key()

        if not api_key:
            msg = missing_credential_message("run the Agent")
            exceptions.append(KeyError(msg))
            return exceptions

        # Return any exceptions
        return exceptions if exceptions else None

    def _handle_additional_context(self, prompt: str, additional_context: str | int | float | dict[str, Any]) -> str:  # noqa: PYI041
        """Integrates additional context into the main prompt string.

        - If context is numeric, it's converted to a string and appended.
        - If context is a string, it's appended on a new line.
        - If context is a dictionary, the prompt is treated as a Jinja2 template
          and rendered with the dictionary as variables.

        Args:
            prompt: The base prompt string.
            additional_context: The context to integrate (str, int, float, dict).

        Returns:
            The potentially modified prompt string.
        """
        context = additional_context
        if isinstance(context, (int, float)):
            # If the additional context is a number, we want to convert it to a string.
            context = str(context)
        if isinstance(context, str):
            prompt += f"\n{context!s}"
        elif isinstance(context, dict):
            prompt = Template(prompt).render(context)
        else:
            # For any other type, convert to string and append
            try:
                context_str = str(context)
                prompt += f"\n{context_str}"
            except Exception:
                # If conversion fails, log warning and continue with original prompt
                msg = f"[WARNING] Unable to process additional_context of type {type(context).__name__}, ignoring."
                logger.warning(msg)
                self.append_value_to_parameter("logs", msg)
        return prompt

    # --- Processing ---
    def process(self) -> Generator[Any, Any, None]:
        """Build the agent state, run it on the prompt, and set the outputs.

        Yields:
            A callable that runs the agent, for the engine to run off the main thread.
        """
        model_input = self.get_parameter_value("model")
        provider_name = self.get_parameter_value("model_provider") or ProviderID.GRIPTAPE_CLOUD
        agent_input = self.get_parameter_value("agent")
        # License-policy runtime gate, scoped to Griptape Cloud models (the only ones the
        # catalog declares) and skipped when an Agent is connected: it supplies its own
        # model, so the node's (hidden, not cleared) dropdown value is stale. The
        # invocation declaration below gates the model that actually runs.
        if agent_input is None and provider_name == ProviderID.GRIPTAPE_CLOUD:
            self._model_access.raise_if_selection_denied()
        include_details = self.get_parameter_value("include_details")

        self.append_value_to_parameter("logs", "[Processing..]\n")

        # Tools arrive as config dicts (rebuilt fresh by each node that runs the agent) or as
        # live griptape tools. Live tools have no config, so they serve this run only.
        raw_tool_inputs = self.get_parameter_list_value("tools")
        _, tool_configs = build_tools([t for t in raw_tool_inputs if isinstance(t, dict)])
        run_only_tools = [
            tool for item in raw_tool_inputs if not isinstance(item, dict) for tool in griptape_tool_to_pydantic(item)
        ]
        if include_details and raw_tool_inputs:
            names = [
                item.get("mcp_server_name", item.get("tool_type", "unknown")) if isinstance(item, dict) else item.name
                for item in raw_tool_inputs
            ]
            self.append_value_to_parameter("logs", f"[Tools]: {', '.join(names)}\n")

        # Strings are promoted to single-rule rulesets named behavior_1, behavior_2, etc.
        ruleset_configs = self._ruleset_configs(self.get_parameter_list_value("rulesets"))
        if include_details and ruleset_configs:
            self.append_value_to_parameter("logs", f"\n[Rulesets]: {', '.join(r['name'] for r in ruleset_configs)}\n")

        state = AgentState.from_wire(agent_input) if isinstance(agent_input, dict) else None
        if state is None:
            state = self._new_state(model_input, provider_name)
            state.tools = tool_configs
            state.rulesets = ruleset_configs
        else:
            # A connected agent keeps its own tools; rulesets connected here add to its own.
            state.rulesets = [*state.rulesets, *ruleset_configs]
        state.output_schema = self.get_parameter_value("output_schema")
        if include_details and state.output_schema is not None:
            self.append_value_to_parameter("logs", "[Schema]: Structured output schema provided\n")

        memory_data = self._parse_memory_data(self.get_parameter_value("agent_memory"))
        if memory_data is not None:
            state.messages = self._memory_to_messages(memory_data)

        prompt = self.get_parameter_value("prompt") or ""
        additional_context = self.get_parameter_value("additional_context")
        if additional_context:
            prompt = self._handle_additional_context(prompt, additional_context)
        if include_details and prompt:
            self.append_value_to_parameter("logs", f"[Prompt]:\n{prompt}\n")

        images: list[UserContent] = [
            image_content
            for image in self.get_parameter_list_value("images")
            if (image_content := image_prompt_content(image)) is not None
        ]
        if include_details and images:
            self.append_value_to_parameter("logs", f"[Images]: {len(images)}\n")

        if (prompt and not prompt.isspace()) or images:
            # Declare the model that actually runs (a connected agent's, not the stale
            # dropdown value) before the network call, so a denied invocation fails closed.
            require_model_invocation_sync(self, state.model)

            content: list[UserContent] = [prompt, *images] if prompt.strip() else images
            self.append_value_to_parameter("logs", "[Started processing agent..]\n")
            run_state = state
            run: AgentRunOutput = yield lambda: self._process(run_state, content, run_only_tools)
            self.append_value_to_parameter("logs", "\n[Finished processing agent.]\n")
            if not run.cancelled:
                self.set_parameter_value("output", format_output(run.output))
            state = run.state
        else:
            self.append_value_to_parameter("logs", "[No prompt provided, creating Agent.]\n")
            self.parameter_output_values["output"] = "Agent created."
        self.parameter_output_values["agent"] = state.to_wire()

    def _process(
        self, state: AgentState, prompt: list[UserContent], run_only_tools: list | None = None
    ) -> AgentRunOutput:
        """Run the agent, streaming its reply to ``output`` and tool use to ``logs``."""
        include_details = self.get_parameter_value("include_details")

        def on_event(event: AgentRunEvent) -> None:
            match event:
                case TextDelta(text=text):
                    self.append_value_to_parameter("output", value=text)
                    if include_details:
                        self.append_value_to_parameter("logs", value=text)
                case ToolCalled(tool_name=name, args=args):
                    if include_details:
                        self.append_value_to_parameter("logs", f"\n[Using tool {name}: {args}]\n")
                case ToolReturned(tool_name=name, is_error=is_error):
                    if include_details:
                        status = "failed" if is_error else "finished"
                        self.append_value_to_parameter("logs", f"\n[Tool {name} {status}]\n")
                case _:
                    msg = f"Unknown agent run event: {event!r}"
                    raise ValueError(msg)

        runner = LocalAgentRunner(extra_tools=list(run_only_tools or []))
        run = runner.run(state, prompt, on_event=on_event, is_cancelled=lambda: self.is_cancellation_requested)
        if run.cancelled:
            self.append_value_to_parameter("logs", "\n[Agent execution cancelled by user.]\n")
        return run

    def _ruleset_configs(self, raw_rulesets: list) -> list[dict[str, Any]]:
        configs = []
        behavior_count = 0
        for ruleset in raw_rulesets:
            if isinstance(ruleset, str):
                if not ruleset.strip():
                    continue
                behavior_count += 1
                ruleset = {"name": f"behavior_{behavior_count}", "rules": [ruleset.strip()]}  # noqa: PLW2901
            config = ruleset_to_config(ruleset)
            if config:
                configs.append(config)
        return configs

    def _new_state(self, model_input: object, provider_name: str) -> AgentState:
        """Return a fresh state for the model selected on this node (dropdown or Prompt Model Config)."""
        if isinstance(model_input, BasePromptDriver):
            return AgentState(provider=_provider_ref_for_driver(model_input), model=model_input.model)
        model = str(model_input or DEFAULT_MODEL)
        if provider_name == ProviderID.GRIPTAPE_CLOUD:
            if model not in self._model_access.model_choices:
                model = DEFAULT_MODEL
            return AgentState(model=model)
        config = next((p for p in self._provider._fetch_providers() if p.name == provider_name), None)
        if config is None:
            msg = f"Provider '{provider_name}' is not configured. Refresh the provider list and pick another."
            raise ValueError(msg)
        return AgentState(
            provider=ProviderRef(
                kind=ProviderKind.OLLAMA if config.type == ProviderID.OLLAMA else ProviderKind.OPENAI_COMPATIBLE,
                name=config.name,
                base_url=config.base_url or "",
                api_key_secret=config.api_key_secret_name,
            ),
            model=model,
        )


def _provider_ref_for_driver(driver: BasePromptDriver) -> ProviderRef:
    """Map a prompt driver from a Prompt Model Config node to the provider the runner calls."""
    if isinstance(driver, GriptapeCloudPromptDriver):
        return ProviderRef()
    if isinstance(driver, OllamaPromptDriver):
        host = (driver.host or "").rstrip("/")
        return ProviderRef(kind=ProviderKind.OLLAMA, name=ProviderID.OLLAMA, base_url=f"{host}/v1" if host else "")
    return direct_provider_ref(type(driver).__name__, getattr(driver, "base_url", None))
