"""Defines the ExampleAgent node, providing an interface to interact with a Griptape Agent.

This node allows users to create a new Griptape Agent or continue interaction
with an existing one. It defaults to using the Griptape Cloud prompt driver
but supports connecting custom prompt_model_configurations. It handles parameters
for tools, rulesets, prompts, and streams output back to the user interface.
"""

import json
from typing import TYPE_CHECKING, Any, cast  # cast used for handle_request narrowing

from griptape.artifacts import BaseArtifact, ModelArtifact, TextArtifact
from griptape.drivers.prompt.base_prompt_driver import BasePromptDriver
from griptape.drivers.prompt.griptape_cloud import GriptapeCloudPromptDriver
from griptape.drivers.prompt.openai import OpenAiChatPromptDriver as GtOpenAiChatPromptDriver
from griptape.events import (
    ActionChunkEvent,
    FinishActionsSubtaskEvent,
    FinishStructureRunEvent,
    StartStructureRunEvent,
    TextChunkEvent,
)
from griptape.memory.structure import ConversationMemory, Run
from griptape.structures import Structure
from griptape.tasks import PromptTask
from griptape_nodes.exe_types.core_types import (
    NodeMessageResult,
    Parameter,
    ParameterGroup,
    ParameterList,
    ParameterMessage,
    ParameterMode,
    ParameterType,
)
from griptape_nodes.exe_types.node_types import AsyncResult, BaseNode, ControlNode
from griptape_nodes.exe_types.param_types.parameter_json import ParameterJson
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.retained_mode.events.connection_events import DeleteConnectionRequest
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes, logger
from griptape_nodes.traits.button import Button, ButtonDetailsMessagePayload
from griptape_nodes.traits.options import Options

if TYPE_CHECKING:
    from griptape_nodes.retained_mode.events.agent_events import (  # type: ignore[import]
        ListAgentProvidersRequest,
        ListAgentProvidersResultSuccess,
        ListProviderModelsRequest,
        ListProviderModelsResultSuccess,
    )

try:
    from griptape_nodes.retained_mode.events.agent_events import (  # type: ignore[import]
        ListAgentProvidersRequest,
        ListAgentProvidersResultSuccess,
        ListProviderModelsRequest,
        ListProviderModelsResultSuccess,
    )
    _AGENT_PROVIDERS_AVAILABLE = True
except ImportError:
    _AGENT_PROVIDERS_AVAILABLE = False
from jinja2 import Template
from json_schema_to_pydantic import create_model  # pyright: ignore[reportMissingImports]

from griptape_nodes_library.agents.griptape_nodes_agent import GriptapeNodesAgent as GtAgent
from griptape_nodes_library.config.prompt.cloud_models import (
    DEPRECATED_MODELS,
    MODEL_CHOICES,
    MODEL_CHOICES_ARGS,
)
from griptape_nodes_library.utils.agent_utils import (
    build_rulesets_from_configs,
    build_tools,
    ruleset_to_config,
    unwrap_agent,
    wrap_agent,
)
from griptape_nodes_library.utils.error_utils import try_throw_error

# --- Constants ---
API_KEY_ENV_VAR = "GT_CLOUD_API_KEY"
SERVICE = "Griptape"
DEFAULT_MODEL = "claude-sonnet-4-6"


class Agent(ControlNode):
    """A Griptape Node that provides an interface to interact with a Griptape Agent.

    This node facilitates communication with a Griptape Agent, allowing for
    sending prompts and receiving streamed responses. It can initialize a new
    agent or operate on an existing agent representation passed as input.

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
        self.add_node_element(
            ParameterMessage(
                name="model_deprecation_notice",
                title="Model Deprecation Notice",
                variant="info",
                value="",
                traits={
                    Button(
                        full_width=True,
                        on_click=lambda _, __: self.hide_message_by_name("model_deprecation_notice"),
                    )
                },
                button_text="Dismiss",
                hide=True,
            )
        )

        # Provider selector — populated from the engine's configured providers.
        provider_names = self._fetch_provider_names()
        self.add_parameter(
            Parameter(
                name="model_provider",
                type="str",
                default_value=provider_names[0] if provider_names else "griptape_cloud",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                tooltip="Choose a provider. Refresh to see all configured providers.",
                traits={
                    Options(choices=provider_names),
                    Button(
                        icon="list-restart",
                        size="icon",
                        variant="secondary",
                        on_click=self._refresh_providers_button,
                    ),
                },
                ui_options={"display_name": "provider"},
            )
        )

        # Model selector — choices update when the provider changes.
        self.add_parameter(
            Parameter(
                name="model",
                input_types=["str", "Prompt Model Config"],
                default_value=DEFAULT_MODEL,
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                tooltip="Choose a model, or connect a Prompt Model Configuration",
                traits={
                    Options(choices=MODEL_CHOICES),
                    Button(
                        icon="list-restart",
                        size="icon",
                        variant="secondary",
                        on_click=self._refresh_models_button,
                    ),
                },
                ui_options={"display_name": "prompt model", "data": MODEL_CHOICES_ARGS},
            )
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
                ui_options={"hide_property": True},
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

    def before_value_set(
        self,
        parameter: Parameter,
        value: Any,
    ) -> Any:
        if parameter.name == "model":
            # Only run deprecation check for string model names. When a prompt driver
            # is connected, value is a BasePromptDriver; "value in DEPRECATED_MODELS"
            # would hash it and raise (drivers are unhashable). See #3713.
            if isinstance(value, str) and value in DEPRECATED_MODELS:
                replacement = DEPRECATED_MODELS[value]
                message = self.get_message_by_name_or_element_id("model_deprecation_notice")
                if message is None:
                    raise RuntimeError("model_deprecation_notice message element not found")  # noqa: TRY003, EM101
                message.value = f"The '{value}' model has been deprecated. The model has been updated to '{replacement}'. Please save your workflow to apply this change."
                self.show_message_by_name("model_deprecation_notice")
                value = replacement
            elif isinstance(value, str):
                self.hide_message_by_name("model_deprecation_notice")

        # Call the parent implementation and return the result
        return super().before_value_set(
            parameter,
            value,
        )

    def _build_tool_exchange(self, subtask_events: list) -> str:
        """Build a verified tool-use record from FinishActionsSubtaskEvents.

        This is prepended to the stored run output so memory faithfully reflects
        every tool call that happened, giving downstream agents clear evidence.
        """
        lines = ["[Verified tool use:"]
        for event in subtask_events:
            if event.subtask_thought:
                lines.append(f"  Thought: {event.subtask_thought}")
            for action in event.subtask_actions or []:
                name = action.get("name", "?")
                path = action.get("path", "")
                tool_input = action.get("input", {})
                prefix = f"{path}/" if path else ""
                lines.append(f"  Tool: {prefix}{name}")
                if tool_input:
                    lines.append(f"  Input: {json.dumps(tool_input)}")
            if event.task_output:
                result = event.task_output.to_text()
                if len(result) > 400:
                    result = result[:400] + "…"
                lines.append(f"  Result: {result}")
        lines.append("]")
        return "\n".join(lines) + "\n\n"

    # --- Provider / Model Methods ---

    def _fetch_providers(self) -> list[dict]:
        """Fetch configured providers from the engine, falling back to griptape_cloud only."""
        _FALLBACK = [{"name": "griptape_cloud", "type": "griptape_cloud"}]
        if not _AGENT_PROVIDERS_AVAILABLE:
            return _FALLBACK
        try:
            result = GriptapeNodes.handle_request(ListAgentProvidersRequest())
            if not isinstance(result, ListAgentProvidersResultSuccess):
                return _FALLBACK
            return cast(ListAgentProvidersResultSuccess, result).providers or _FALLBACK
        except Exception:
            return _FALLBACK

    def _fetch_provider_names(self) -> list[str]:
        """Return ordered provider names for the provider dropdown."""
        providers = self._fetch_providers()
        return [p["name"] for p in providers] or ["griptape_cloud"]

    def _resolve_provider_api_key(self, provider_config: dict) -> str:
        """Resolve the API key for a provider config.

        Provider configs carry api_key_secret_name (the name of a secret in the
        secrets manager) rather than a raw api_key value.  Fall back to
        "not-needed" for providers that don't require a key (e.g. Ollama).
        """
        secret_name = provider_config.get("api_key_secret_name", "")
        if secret_name:
            return GriptapeNodes.SecretsManager().get_secret(secret_name, should_error_on_not_found=False) or "not-needed"
        return "not-needed"

    def _fetch_models_for_provider(self, provider_name: str) -> list[str]:
        """Return the model list for a given provider name."""
        if not _AGENT_PROVIDERS_AVAILABLE:
            return MODEL_CHOICES
        try:
            providers = self._fetch_providers()
            provider_config = next((p for p in providers if p["name"] == provider_name), None)
            if provider_config is None:
                return MODEL_CHOICES
            result = GriptapeNodes.handle_request(
                ListProviderModelsRequest(
                    provider=provider_config.get("type", provider_name),
                    base_url=provider_config.get("base_url", ""),
                    api_key=self._resolve_provider_api_key(provider_config),
                )
            )
            if isinstance(result, ListProviderModelsResultSuccess):
                return cast(ListProviderModelsResultSuccess, result).models or MODEL_CHOICES
        except Exception:
            pass
        return MODEL_CHOICES

    def _update_model_choices_for_provider(self, provider_name: str) -> None:
        """Swap the model dropdown choices for the given provider."""
        models = self._fetch_models_for_provider(provider_name)
        default = models[0] if models else DEFAULT_MODEL
        self._update_option_choices(param="model", choices=models, default=default)
        # The frontend renders the dropdown from the "data" ui_options key (not "simple_dropdown").
        # Update it explicitly so the frontend receives the change notification.
        param = self.get_parameter_by_name("model")
        if param:
            if provider_name == "griptape_cloud":
                new_data = MODEL_CHOICES_ARGS
            else:
                new_data = [{"name": m, "icon": "", "args": {}} for m in models]
            param.update_ui_options_key("data", new_data)

    def _refresh_providers_button(
        self, button: Button, button_details: ButtonDetailsMessagePayload
    ) -> NodeMessageResult | None:  # noqa: ARG002
        """Refresh the provider dropdown from the engine."""
        provider_names = self._fetch_provider_names()
        current = self.get_parameter_value("model_provider") or "griptape_cloud"
        default = current if current in provider_names else (provider_names[0] if provider_names else "griptape_cloud")
        self._update_option_choices(param="model_provider", choices=provider_names, default=default)
        return None

    def _refresh_models_button(
        self, button: Button, button_details: ButtonDetailsMessagePayload
    ) -> NodeMessageResult | None:  # noqa: ARG002
        """Refresh the model dropdown for the currently selected provider."""
        provider_name = self.get_parameter_value("model_provider") or "griptape_cloud"
        self._update_model_choices_for_provider(provider_name)
        return None

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

    def _convert_memory_to_runs(self, memory_data: dict[str, Any]) -> list[Run]:
        """Convert memory data to a list of Run objects.

        Finds 'runs' anywhere in the data structure, extracts input/output values,
        and creates Run objects.

        Args:
            memory_data: Memory data dict in any format (simplified or full conversation_memory).

        Returns:
            List of Run objects, or empty list if conversion fails.
        """
        runs = []

        if not isinstance(memory_data, dict):
            return runs

        # Find runs anywhere in the data structure
        runs_data = self._find_runs_in_data(memory_data)

        # If no runs found, check if it's a single run format
        if not runs_data:
            if "input" in memory_data and "output" in memory_data:
                runs_data = [memory_data]
            else:
                return runs

        for run_data in runs_data:
            if not isinstance(run_data, dict):
                continue

            if "input" not in run_data or "output" not in run_data:
                continue

            input_data = run_data["input"]
            output_data = run_data["output"]

            # Extract values from artifacts
            input_value = self._extract_value_from_artifact(input_data)
            output_value = self._extract_value_from_artifact(output_data)

            runs.append(
                Run(
                    input=TextArtifact(value=input_value),
                    output=TextArtifact(value=output_value),
                )
            )

        return runs

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

    def _is_simplified_format(self, memory_data: dict[str, Any]) -> bool:
        """Check if memory data is in simplified format.

        Simplified format has direct string inputs/outputs, full format has nested artifacts.

        Args:
            memory_data: Memory data dict.

        Returns:
            True if simplified format, False if full format.
        """
        if "runs" not in memory_data or not isinstance(memory_data["runs"], list) or not memory_data["runs"]:
            return False

        first_run = memory_data["runs"][0]
        if not isinstance(first_run, dict):
            return False

        input_data = first_run.get("input")
        # Simplified format has direct strings, full format has nested dicts with "type" and "value"
        return isinstance(input_data, str) or (isinstance(input_data, dict) and "type" not in input_data)

    def _apply_memory_via_from_dict(self, agent: GtAgent, memory_data: dict[str, Any]) -> bool:
        """Apply memory using ConversationMemory.from_dict().

        Args:
            agent: The agent to apply memory to.
            memory_data: Memory data dict in full format.

        Returns:
            True if successful, False otherwise.
        """
        if not hasattr(ConversationMemory, "from_dict"):
            return False

        if agent.conversation_memory is None:
            return False

        try:
            # Preserve the original memory driver if it exists
            original_driver = agent.conversation_memory.conversation_memory_driver
            new_memory = ConversationMemory.from_dict(memory_data)
            # Restore the original driver to maintain persistence
            if original_driver is not None:
                new_memory.conversation_memory_driver = original_driver
            agent.conversation_memory = new_memory
            return True  # noqa: TRY300
        except (ValueError, TypeError, AttributeError):
            # Fall back to manual conversion if from_dict() fails with expected errors
            return False

    def _apply_memory_to_agent(self, agent: GtAgent, memory_data: dict[str, Any] | str | None) -> None:
        """Apply memory data to an agent's conversation memory.

        Uses ConversationMemory.from_dict() if available, otherwise falls back to manual conversion.

        Args:
            agent: The agent to apply memory to.
            memory_data: Memory data dict, JSON string, or None to skip.
        """
        # Failure cases first
        if agent.conversation_memory is None:
            return

        parsed_data = self._parse_memory_data(memory_data)
        if parsed_data is None:
            return

        # Try to use ConversationMemory.from_dict() only for full format
        if not self._is_simplified_format(parsed_data) and self._apply_memory_via_from_dict(agent, parsed_data):
            return

        # Success path - manual conversion
        if agent.conversation_memory is None:
            return

        runs = self._convert_memory_to_runs(parsed_data)
        if not runs:
            # If no valid runs, clear the memory
            agent.conversation_memory.runs = []
        else:
            # Success path - replace memory with new runs
            agent.conversation_memory.runs = runs

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
        if parameter.name == "model_provider":
            provider_name = str(value)
            models = self._fetch_models_for_provider(provider_name)
            default = models[0] if models else DEFAULT_MODEL
            self._update_option_choices(param="model", choices=models, default=default)
            new_data = (
                MODEL_CHOICES_ARGS
                if provider_name == "griptape_cloud"
                else [{"name": m, "icon": "", "args": {}} for m in models]
            )
            param = self.get_parameter_by_name("model")
            if param:
                param.update_ui_options_key("data", new_data)

    # --- UI Interaction Hooks ---

    def after_incoming_connection(
        self, source_node: BaseNode, source_parameter: Parameter, target_parameter: Parameter
    ) -> None:
        # If an existing agent is connected, hide parameters related to creating a new one.
        if target_parameter.name == "agent":
            params_to_toggle = ["model", "model_provider", "tools", "rulesets"]
            self.hide_parameter_by_name(params_to_toggle)

        if target_parameter.name == "model" and source_parameter.name == "prompt_model_config":
            # Remove the options trait
            target_parameter.remove_trait(trait_type=target_parameter.find_elements_by_type(Options)[0])

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
            params_to_toggle = ["model", "model_provider", "tools", "rulesets", "schema"]
            self.show_parameter_by_name(params_to_toggle)

        if target_parameter.name == "output_schema":
            self.set_parameter_value("output_schema", None)
            # When schema is disconnected, change output type back to str and validate connections
            self._update_output_type_and_validate_connections("str")

        if target_parameter.name == "model":
            # Reset the parameter type
            target_parameter.type = "str"

            # Enable PROPERTY so the user can set it
            target_parameter.allowed_modes = {ParameterMode.INPUT, ParameterMode.PROPERTY}

            # Sometimes the value is not set to the default value - these are all attempts to get it to work.
            target_parameter.set_default_value(DEFAULT_MODEL)
            target_parameter.default_value = DEFAULT_MODEL
            self.set_parameter_value("model", DEFAULT_MODEL)

            # Restore choices for the currently selected provider.
            current_provider = self.get_parameter_value("model_provider") or "griptape_cloud"
            restored_models = self._fetch_models_for_provider(current_provider)
            self._update_option_choices(param="model", choices=restored_models, default=DEFAULT_MODEL)

            # Change the display name to be appropriate
            ui_options = target_parameter.ui_options
            ui_options["display_name"] = "prompt model"
            target_parameter.ui_options = ui_options

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

        The Griptape Cloud API key is only required when the node would fall back
        to the default Griptape Cloud prompt driver. A connected agent carries its
        own driver, and a connected prompt driver (Prompt Model Config) supplies its
        own credentials, so neither needs the cloud key.

        Returns:
            A list of Exception objects if validation fails, otherwise None.
        """
        exceptions = []

        # Mirror the driver-selection precedence in process(): a connected agent or a
        # connected prompt driver bypass the default Griptape Cloud driver entirely.
        if not self._uses_griptape_cloud_driver():
            return None

        # Check to see if the API key is set.
        api_key = GriptapeNodes.SecretsManager().get_secret(API_KEY_ENV_VAR)

        if not api_key:
            msg = f"{API_KEY_ENV_VAR} is not defined"
            exceptions.append(KeyError(msg))
            return exceptions

        # Return any exceptions
        return exceptions if exceptions else None

    def _uses_griptape_cloud_driver(self) -> bool:
        """Return True when the node will build the default Griptape Cloud prompt driver.

        A connected agent supplies its own driver, a connected prompt driver
        (``model`` resolved to a ``BasePromptDriver``) supplies its own credentials,
        and a non-griptape_cloud provider uses an OpenAI-compatible driver — none
        of these need the Griptape Cloud API key.
        """
        if self.get_parameter_value("agent") is not None:
            return False
        model_input = self.get_parameter_value("model")
        if isinstance(model_input, BasePromptDriver):
            return False
        provider_name = self.get_parameter_value("model_provider") or "griptape_cloud"
        return provider_name == "griptape_cloud"

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
    def process(self) -> AsyncResult[Structure]:  # noqa: C901, PLR0915, PLR0912
        """Executes the main logic of the node asynchronously.

        Sets up the Griptape Agent (either new or from input), configures the
        prompt driver, prepares the prompt with context, and then yields
        a lambda function to perform the actual agent interaction via `_process`.
        Handles setting output parameters after execution.

        Yields:
            A lambda function wrapping the call to `_process` for asynchronous execution.

        Returns:
            An AsyncResult indicating the structure being processed (the agent).
        """
        model_input = self.get_parameter_value("model")
        provider_name = self.get_parameter_value("model_provider") or "griptape_cloud"
        agent = None
        include_details = self.get_parameter_value("include_details")
        default_prompt_driver = GriptapeCloudPromptDriver(
            model=DEFAULT_MODEL,
            api_key=GriptapeNodes.SecretsManager().get_secret(API_KEY_ENV_VAR),
            stream=True,
        )

        # Initialize the logs parameter
        self.append_value_to_parameter("logs", "[Processing..]\n")

        # Get any tools — may be live objects or serializable config dicts (e.g. MCPTool)
        raw_tool_inputs = self.get_parameter_list_value("tools")
        live_tools, tool_configs = build_tools(raw_tool_inputs)
        tools = live_tools
        if include_details and live_tools:
            names = []
            for item in raw_tool_inputs:
                if isinstance(item, dict):
                    names.append(item.get("mcp_server_name", item.get("tool_type", "unknown")))
                else:
                    names.append(item.name)
            self.append_value_to_parameter("logs", f"[Tools]: {', '.join(names)}\n")

        # Get any rulesets — convert live objects to serializable configs so they survive chaining.
        # Strings are auto-promoted to single-rule rulesets named behavior_1, behavior_2, etc.
        raw_rulesets = self.get_parameter_list_value("rulesets")
        str_counter = 0
        promoted_rulesets = []
        for r in raw_rulesets:
            if isinstance(r, str) and r.strip():
                str_counter += 1
                promoted_rulesets.append({"name": f"behavior_{str_counter}", "rules": [r.strip()]})
            else:
                promoted_rulesets.append(r)
        ruleset_configs: list = [c for c in (ruleset_to_config(r) for r in promoted_rulesets) if c]
        rulesets = build_rulesets_from_configs(ruleset_configs)
        if include_details and rulesets:
            self.append_value_to_parameter(
                "logs",
                f"\n[Rulesets]: {', '.join([r.name for r in rulesets])}\n",
            )

        # Get the output schema
        output_schema = self.get_parameter_value("output_schema")
        pydantic_schema = None
        if output_schema is not None:
            try:
                pydantic_schema = create_model(output_schema)
            except Exception as e:
                msg = f"[ERROR]: Unable to create output schema model: {e}. Try using the `Create Agent Schema` node to generate a schema."
                self.append_value_to_parameter("logs", msg + "\n")
                logger.error(msg)
                raise

        if include_details and pydantic_schema:
            self.append_value_to_parameter("logs", "[Schema]: Structured output schema provided\n")

        # Get the prompt
        prompt = self.get_parameter_value("prompt")

        # Use any additional context provided by the user.
        additional_context = self.get_parameter_value("additional_context")
        if additional_context:
            prompt = self._handle_additional_context(prompt, additional_context)

        # If the user has connected a prompt, we want to show it in the logs.
        if include_details and prompt:
            self.append_value_to_parameter("logs", f"[Prompt]:\n{prompt}\n")

        # If an agent is provided, we'll use and ensure it's using a PromptTask
        # If a prompt_driver is provided, we'll use that
        # If neither are provided, we'll create a new one with the selected model.
        # Otherwise, we'll just use the default model
        agent_input = self.get_parameter_value("agent")
        incoming_provider: dict | None = None
        if isinstance(agent_input, dict):
            # Unwrap the new-format wrapper (or handle old raw agent dict gracefully).
            agent_core_dict, incoming_tool_configs, incoming_ruleset_configs = unwrap_agent(agent_input)
            incoming_provider = agent_input.get("provider")  # non-GTC provider config forwarded by upstream node
            agent = GtAgent().from_dict(agent_core_dict)
            # Rebuild tools from the incoming config so the live connection is fresh.
            if incoming_tool_configs:
                incoming_live_tools, _ = build_tools(incoming_tool_configs)
                if incoming_live_tools and agent.tasks:
                    cast(PromptTask, agent.tasks[0]).tools = incoming_live_tools
                tool_configs = incoming_tool_configs  # carry forward for output wrap
            # Merge incoming rulesets with any rulesets connected at this node; set directly on _rulesets.
            ruleset_configs = incoming_ruleset_configs + ruleset_configs
            agent._rulesets = build_rulesets_from_configs(ruleset_configs)
            # make sure the agent is using a PromptTask — replace rather than add to avoid two tasks
            if not isinstance(agent.tasks[0], PromptTask):
                agent.tasks[0] = PromptTask(prompt_driver=default_prompt_driver, output_schema=pydantic_schema)
            else:
                agent.tasks[0].output_schema = pydantic_schema
            # Rebuild the prompt driver for non-GTC providers — griptape strips api_key during serialization.
            if incoming_provider:
                rebuilt_driver = GtOpenAiChatPromptDriver(
                    model=cast(PromptTask, agent.tasks[0]).prompt_driver.model,
                    base_url=incoming_provider.get("base_url", ""),
                    api_key=incoming_provider.get("api_key") or "not-needed",
                    stream=True,
                )
                cast(PromptTask, agent.tasks[0]).prompt_driver = rebuilt_driver
        elif isinstance(model_input, BasePromptDriver):
            agent = GtAgent(prompt_driver=model_input, tools=tools, rulesets=rulesets, output_schema=pydantic_schema)
        elif isinstance(model_input, str):
            if provider_name == "griptape_cloud":
                if model_input not in MODEL_CHOICES:
                    model_input = DEFAULT_MODEL
                # Get the appropriate args (stream setting, structured output strategy, etc.)
                args = next((model["args"] for model in MODEL_CHOICES_ARGS if model["name"] == model_input), {})
                args = {k: v for k, v in args.items() if v is not None}
                prompt_driver = GriptapeCloudPromptDriver(
                    model=model_input,
                    api_key=GriptapeNodes.SecretsManager().get_secret(API_KEY_ENV_VAR),
                    **args,
                )
            else:
                # Non-Griptape-Cloud provider: resolve config and use the OpenAI-compatible driver.
                providers = self._fetch_providers()
                provider_config = next((p for p in providers if p["name"] == provider_name), {})
                base_url = provider_config.get("base_url", "")
                api_key = self._resolve_provider_api_key(provider_config)
                prompt_driver = GtOpenAiChatPromptDriver(
                    model=model_input,
                    base_url=base_url,
                    api_key=api_key,
                    stream=True,
                )
            agent = GtAgent(prompt_driver=prompt_driver, tools=tools, rulesets=rulesets, output_schema=pydantic_schema)

        if agent is None:
            msg = "Agent was not initialized"
            raise RuntimeError(msg)

        # Apply memory if provided
        agent_memory = self.get_parameter_value("agent_memory")
        if agent_memory is not None:
            self._apply_memory_to_agent(agent, agent_memory)

        if prompt and not prompt.isspace():
            # Run the agent asynchronously
            self.append_value_to_parameter("logs", "[Started processing agent..]\n")
            yield lambda: self._process(agent, prompt)
            self.append_value_to_parameter("logs", "\n[Finished processing agent.]\n")
            try_throw_error(agent.output)
            # Settle the output field to the final answer only — not the [Verified tool use: ...]
            # block we prepend to memory for downstream agents.  _process() saves the raw answer
            # in self._last_raw_output before modifying memory; fall back to memory when no tools ran.
            raw_output = getattr(self, "_last_raw_output", None)
            if raw_output is not None:
                self.set_parameter_value("output", raw_output)
                self._last_raw_output = None
            elif agent is not None and agent.conversation_memory and agent.conversation_memory.runs:
                self.set_parameter_value("output", agent.conversation_memory.runs[-1].output.to_text())
        else:
            self.append_value_to_parameter("logs", "[No prompt provided, creating Agent.]\n")
            self.parameter_output_values["output"] = "Agent created."
        # Clear tools from the live agent before serializing — MCPTool connections are not
        # serializable. They're rebuilt from tool_configs when the next node unwraps.
        if agent.tasks:
            cast(PromptTask, agent.tasks[0]).tools = []
        wrapper = wrap_agent(agent.to_dict(), tool_configs, ruleset_configs)
        # Forward provider credentials so downstream nodes can rebuild the driver —
        # griptape strips api_key from non-GTC drivers during to_dict() serialization.
        if provider_name != "griptape_cloud":
            providers = self._fetch_providers()
            p = next((x for x in providers if x["name"] == provider_name), {})
            wrapper["provider"] = {
                "name": provider_name,
                "base_url": p.get("base_url", ""),
                "api_key": self._resolve_provider_api_key(p),
            }
        elif incoming_provider:
            # Passthrough: this node uses griptape_cloud but an upstream agent used a non-GTC provider.
            wrapper["provider"] = incoming_provider
        self.parameter_output_values["agent"] = wrapper

    def _process(self, agent: GtAgent, prompt: BaseArtifact | str) -> Structure:  # noqa: C901, PLR0912
        """Performs the synchronous, streaming interaction with the Griptape Agent.

        Iterates through events generated by `agent.run_stream`, updating the
        'output' parameter with text chunks and the 'logs' parameter with
        action details (if enabled).

        Normally we would use the pattern:
        for artifact in Stream(agent).run(prompt):
        But for this example, we'll use the run_stream method to get the events so we can
        show the user when the Agent is using a tool.

        Args:
            agent: The configured Griptape Agent instance.
            prompt: The final prompt string or BaseArtifact to send to the agent.

        Returns:
            The agent structure after processing.
        """
        include_details = self.get_parameter_value("include_details")

        args = [prompt] if prompt else []
        structure_id_stack = []
        active_structure_id = None
        subtask_events: list[FinishActionsSubtaskEvent] = []

        task = agent.tasks[0]
        if not isinstance(task, PromptTask):
            msg = "Agent must have a PromptTask"
            raise TypeError(msg)
        prompt_driver = task.prompt_driver
        prompt_driver.stream = True
        if prompt_driver.stream:
            for event in agent.run_stream(
                *args,
                event_types=[
                    StartStructureRunEvent,
                    TextChunkEvent,
                    ActionChunkEvent,
                    FinishActionsSubtaskEvent,
                    FinishStructureRunEvent,
                ],
            ):
                if isinstance(event, StartStructureRunEvent):
                    active_structure_id = event.structure_id
                    structure_id_stack.append(active_structure_id)
                if isinstance(event, FinishStructureRunEvent):
                    structure_id_stack.pop()
                    active_structure_id = structure_id_stack[-1] if structure_id_stack else None

                # If an Agent uses other Agents (via `StructureRunTool`), we will receive those events too.
                # We want to ignore those events and only show the events for this node's Agent.
                # TODO: https://github.com/griptape-ai/griptape-nodes/issues/984
                if agent.id == active_structure_id:
                    # Check for cancellation request
                    if self.is_cancellation_requested:
                        self.append_value_to_parameter("logs", "\n[Agent execution cancelled by user.]\n")
                        return agent

                    if isinstance(event, TextChunkEvent):
                        self.append_value_to_parameter("output", value=event.token)
                        if include_details:
                            self.append_value_to_parameter("logs", value=event.token)

                    if isinstance(event, ActionChunkEvent) and event.name and event.tag:
                        if include_details:
                            self.append_value_to_parameter("logs", f"\n[Using tool {event.name}: ({event.path})]\n")

                    # Capture completed subtask exchanges for faithful memory reconstruction.
                    if isinstance(event, FinishActionsSubtaskEvent) and event.subtask_parent_task_id == task.id:
                        subtask_events.append(event)

            # Prepend verified tool-use record to memory so downstream agents have
            # clear evidence of every tool call.  Save the raw final answer first so
            # the output field shows just the answer, not the metadata block.
            if subtask_events and agent.conversation_memory and agent.conversation_memory.runs:
                exchange = self._build_tool_exchange(subtask_events)
                last_run = agent.conversation_memory.runs[-1]
                self._last_raw_output = last_run.output.to_text()
                agent.conversation_memory.runs[-1] = Run(
                    input=TextArtifact(value=last_run.input.to_text()),
                    output=TextArtifact(value=exchange + self._last_raw_output),
                )

        else:
            agent.run(*args)
            agent_output = agent.output
            if isinstance(agent_output, ModelArtifact):
                self.set_parameter_value("output", agent_output.value.model_dump_json())
            else:
                self.set_parameter_value("output", str(agent_output))
            try_throw_error(agent.output)

        return agent
