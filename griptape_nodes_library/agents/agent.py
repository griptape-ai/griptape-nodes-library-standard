from typing import Any

from griptape_nodes.drivers.cloud_models import MODEL_CHOICES
from griptape_nodes.exe_types.core_types import (
    NodeMessageResult,
    Parameter,
    ParameterGroup,
    ParameterList,
    ParameterMode,
    ParameterType,
)
from griptape_nodes.exe_types.node_types import AsyncResult, BaseNode, ControlNode
from griptape_nodes.exe_types.param_components.model_access_component import ModelAccessComponent
from griptape_nodes.exe_types.param_types.parameter_json import ParameterJson
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.retained_mode.events.agent_events import ProviderConfig
from griptape_nodes.retained_mode.events.connection_events import DeleteConnectionRequest
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes, logger
from griptape_nodes.traits.button import Button, ButtonDetailsMessagePayload
from griptape_nodes.traits.options import Options
from jinja2 import Template
from pydantic_ai.agent import AgentRunResult

from griptape_nodes_library.llm.agent_node_support import (
    DEFAULT_CLOUD_MODEL,
    parse_agent_memory,
)
from griptape_nodes_library.llm.agent_state import (
    AgentState,
    compact_messages,
    connected_agent_state,
    find_runs,
    messages_from_runs,
)
from griptape_nodes_library.llm.model_config import (
    ModelConfig,
    ModelProvider,
    model_config_for_engine_provider,
    model_config_from_input,
)
from griptape_nodes_library.llm.rulesets import rulesets_from_inputs
from griptape_nodes_library.llm.runner import (
    AgentRunCancelledError,
    RunCallbacks,
    output_to_text,
    output_type_from_schema,
    run_agent,
)
from griptape_nodes_library.llm.tools import build_agent_from_state, tool_configs_from_inputs, tool_display_name
from griptape_nodes_library.utils.cloud_credential_utils import (
    missing_credential_message,
    resolve_cloud_api_key,
)
from griptape_nodes_library.utils.cloud_legacy_models import CLOUD_LEGACY_MODEL_VALUES
from griptape_nodes_library.utils.model_invocation import require_model_invocation_sync
from griptape_nodes_library.utils.provider_selection_component import ProviderSelectionComponent

_GRIPTAPE_CLOUD_PROVIDER = ProviderConfig(name="griptape_cloud", type="griptape_cloud", model="")

# --- Constants ---
DEFAULT_MODEL = DEFAULT_CLOUD_MODEL


class Agent(ControlNode):
    def __init__(self, **kwargs) -> None:
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
        to a Griptape Cloud model. A connected agent carries its own model, and a
        connected Prompt Model Config supplies its own credentials, so neither needs
        the cloud credential.

        Either credential is accepted: a Griptape Nodes License or a Griptape Cloud
        API key. Griptape Cloud's chat endpoints authenticate a License, so a
        license-only user (no `GT_CLOUD_API_KEY` at all) must not be blocked here.

        Returns:
            A list of Exception objects if validation fails, otherwise None.
        """
        exceptions = []

        # Mirror the model-selection precedence in process(): a connected agent or a
        # connected Prompt Model Config bypass Griptape Cloud entirely.
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
    def _resolve_model_config(self, model_input: Any, provider_name: str) -> ModelConfig:
        """Pick the model for a node with no incoming agent: a connected config, or the dropdown selection."""
        connected = model_config_from_input(model_input)
        if connected is not None:
            return connected
        if not isinstance(model_input, str):
            msg = f"Unsupported model value of type {type(model_input).__name__}; choose a model or connect a Prompt Model Config."
            raise TypeError(msg)
        if provider_name == "griptape_cloud":
            if model_input not in self._model_access.model_choices:
                model_input = DEFAULT_MODEL
            return ModelConfig(provider=ModelProvider.GRIPTAPE_CLOUD, model=model_input)
        providers = self._provider._fetch_providers()
        provider_config = next((p for p in providers if p.name == provider_name), _GRIPTAPE_CLOUD_PROVIDER)
        return model_config_for_engine_provider(provider_config, model_input)

    def _build_state(self, model_input: Any, provider_name: str, agent_input: Any) -> AgentState:
        """Combine the incoming agent (if any) with this node's model, tools, and rulesets."""
        include_details = self.get_parameter_value("include_details")

        node_tools = tool_configs_from_inputs(self.get_parameter_list_value("tools"))
        if include_details and node_tools:
            names = ", ".join(tool_display_name(tool) for tool in node_tools)
            self.append_value_to_parameter("logs", f"[Tools]: {names}\n")

        node_rulesets = rulesets_from_inputs(self.get_parameter_list_value("rulesets"))
        if include_details and node_rulesets:
            names = ", ".join(r["name"] for r in node_rulesets)
            self.append_value_to_parameter("logs", f"\n[Rulesets]: {names}\n")

        state = connected_agent_state(agent_input)
        if state is None:
            state = AgentState(model=self._resolve_model_config(model_input, provider_name))
            state.tools = node_tools
            state.rulesets = node_rulesets
        else:
            state.tools = state.tools + [tool for tool in node_tools if tool not in state.tools]
            state.rulesets = state.rulesets + node_rulesets

        memory = parse_agent_memory(self.get_parameter_value("agent_memory"))
        if memory is not None:
            state.messages = messages_from_runs(find_runs(memory))
        return state

    def _build_output_type(self) -> Any:
        output_schema = self.get_parameter_value("output_schema")
        if output_schema is None:
            return str
        try:
            output_type = output_type_from_schema(output_schema)
        except Exception as e:
            msg = f"[ERROR]: Unable to create output schema model: {e}. Try using the `Create Agent Schema` node to generate a schema."
            self.append_value_to_parameter("logs", msg + "\n")
            raise ValueError(msg) from e
        if self.get_parameter_value("include_details"):
            self.append_value_to_parameter("logs", "[Schema]: Structured output schema provided\n")
        return output_type

    def process(self) -> AsyncResult[AgentRunResult[Any] | None]:
        model_input = self.get_parameter_value("model")
        provider_name = self.get_parameter_value("model_provider") or "griptape_cloud"
        agent_input = self.get_parameter_value("agent")
        # License-policy runtime gate, scoped to Griptape Cloud models (the only ones the
        # catalog declares) and skipped when an Agent is connected: it supplies its own
        # model, so the node's (hidden, not cleared) dropdown value is stale. The
        # INVOKE_MODEL declaration below gates the model that actually runs.
        if agent_input is None and provider_name == "griptape_cloud":
            self._model_access.raise_if_selection_denied()
        include_details = self.get_parameter_value("include_details")

        self.append_value_to_parameter("logs", "[Processing..]\n")

        output_type = self._build_output_type()
        state = self._build_state(model_input, provider_name, agent_input)

        prompt = self.get_parameter_value("prompt")
        additional_context = self.get_parameter_value("additional_context")
        if additional_context:
            prompt = self._handle_additional_context(prompt, additional_context)
        if include_details and prompt:
            self.append_value_to_parameter("logs", f"[Prompt]:\n{prompt}\n")

        if prompt and not prompt.isspace():
            if state.model is None:
                msg = "Agent has no model configured."
                raise RuntimeError(msg)
            # Declare the model that will actually run, before any network call, so a denied
            # invocation fails closed. A connected Agent supplies the real model; the node's own
            # `model` parameter keeps its last dropdown value while hidden.
            require_model_invocation_sync(self, state.model.model)

            self.append_value_to_parameter("logs", "[Started processing agent..]\n")
            result = yield lambda: self._process(state, prompt, output_type)
            self.append_value_to_parameter("logs", "\n[Finished processing agent.]\n")
            if result is not None:
                self.set_parameter_value("output", output_to_text(result.output))
                state.messages = compact_messages([*state.messages, *result.new_messages()])
        else:
            self.append_value_to_parameter("logs", "[No prompt provided, creating Agent.]\n")
            self.parameter_output_values["output"] = "Agent created."

        self.parameter_output_values["agent"] = state.to_wire()

    def _process(self, state: AgentState, prompt: str, output_type: Any) -> AgentRunResult[Any] | None:
        """Run the prompt, streaming text into `output` and tool calls into `logs`."""
        include_details = self.get_parameter_value("include_details")

        def on_text(text: str) -> None:
            self.append_value_to_parameter("output", value=text)
            if include_details:
                self.append_value_to_parameter("logs", value=text)

        def on_tool_call(name: str, args_json: str) -> None:
            if include_details:
                self.append_value_to_parameter("logs", f"\n[Using tool {name}: ({args_json})]\n")

        callbacks = RunCallbacks(
            on_text=on_text, on_tool_call=on_tool_call, is_cancelled=lambda: self.is_cancellation_requested
        )
        agent = build_agent_from_state(state, output_type=output_type)
        try:
            return run_agent(agent, prompt, message_history=state.messages, callbacks=callbacks)
        except AgentRunCancelledError:
            self.append_value_to_parameter("logs", "\n[Agent execution cancelled by user.]\n")
            return None
