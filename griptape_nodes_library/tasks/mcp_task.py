import time
from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterList, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, BaseNode, SuccessFailureNode
from griptape_nodes.exe_types.param_components.model_access_component import ModelAccessComponent
from griptape_nodes.exe_types.param_types.parameter_int import ParameterInt
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.retained_mode.events.agent_events import ProviderConfig
from griptape_nodes.retained_mode.griptape_nodes import logger
from griptape_nodes.traits.button import Button, ButtonDetailsMessagePayload
from griptape_nodes.traits.options import Options
from pydantic_ai.toolsets import AbstractToolset

from griptape_nodes_library.llm.agent_state import AgentState, compact_messages, is_agent_value
from griptape_nodes_library.llm.model_config import ModelConfig, model_config_for_engine_provider
from griptape_nodes_library.llm.task_support import (
    TaskRunResult,
    cloud_model_config,
    run_task_agent,
)
from griptape_nodes_library.llm.tools import ToolType, build_toolset, build_toolsets
from griptape_nodes_library.utils.cloud_legacy_models import cloud_legacy_values_for
from griptape_nodes_library.utils.mcp_utils import (
    get_available_mcp_servers,
    get_server_config,
    validate_mcp_server,
)
from griptape_nodes_library.utils.model_invocation import require_model_invocation_sync
from griptape_nodes_library.utils.provider_selection_component import ProviderSelectionComponent

_GRIPTAPE_CLOUD_PROVIDER = ProviderConfig(name="griptape_cloud", type="griptape_cloud", model="")

# Curated subset of the Cloud chat catalog for the MCP task node. Not every Cloud
# model works reliably with tool-calling, so this list is narrower than the full
# MODEL_CHOICES_ARGS the Agent node offers.
MCP_TASK_MODEL_CHOICES = [
    "claude-sonnet-5",
    "claude-opus-5",
    "claude-haiku-4-5",
    "gemini-3.6-flash",
    "gemini-3.1-pro",
    "gemini-3-flash",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gpt-5.5",
    "gpt-5.4",
    "gpt-5.2",
    "gpt-5.2-chat",
    "gpt-5.1",
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-4o",
    "deepseek-v3",
]

DEFAULT_MODEL = MCP_TASK_MODEL_CHOICES[0]

MCP_TASK_LEGACY_MODEL_VALUES = cloud_legacy_values_for(MCP_TASK_MODEL_CHOICES)


def _ruleset_from_rules_string(rules_string: str | None, server_name: str) -> dict | None:
    """Return an MCP ruleset, or None for blank input. See #307."""
    if not rules_string or not rules_string.strip():
        return None
    return {"name": f"mcp_{server_name}_rules", "rules": [rules_string.strip()]}


class MCPTaskNode(SuccessFailureNode):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        mcp_servers = get_available_mcp_servers()
        default_mcp_server = mcp_servers[0] if mcp_servers else None
        self.add_parameter(
            ParameterString(
                name="mcp_server_name",
                default_value=default_mcp_server,
                tooltip="Select an MCP server to use",
                traits={
                    Options(choices=mcp_servers),
                    Button(
                        full_width=False,
                        icon="refresh-cw",
                        size="icon",
                        variant="secondary",
                        on_click=self._reload_mcp_servers,
                    ),
                },
                placeholder_text="Select MCP server...",
            )
        )
        self.add_parameter(
            Parameter(
                name="agent",
                input_types=["Agent"],
                type="Agent",
                default_value=None,
                tooltip="Optional agent to use - helpful if you want to continue interaction with an existing agent.",
            )
        )

        # Provider + model selectors for the agent this node runs when no `agent` is connected.
        # Wired exactly as the Agent node wires its own: the ProviderSelectionComponent installs
        # the Options + refresh Button traits on `model_provider`, and the ModelAccessComponent
        # owns those traits on `model`, decorates each row with the caller's license entitlement,
        # and gates the selection at execute time. Both are hidden while an agent is connected,
        # since that agent carries its own model.
        model_provider_param = Parameter(
            name="model_provider",
            type="str",
            default_value="griptape_cloud",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            tooltip="Choose a provider. Refresh to see all configured providers.",
            ui_options={"display_name": "provider"},
        )
        self.add_parameter(model_provider_param)

        model_param = Parameter(
            name="model",
            type="str",
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
            model_choices=MCP_TASK_MODEL_CHOICES,
            default_model=DEFAULT_MODEL,
            deprecated_values=MCP_TASK_LEGACY_MODEL_VALUES,
        )
        self._provider_selection = ProviderSelectionComponent(
            node=self,
            model_provider_param=model_provider_param,
            model_access=self._model_access,
            default_model=DEFAULT_MODEL,
        )

        self.add_parameter(
            ParameterInt(
                name="max_subtasks",
                default_value=20,
                hide=True,
                tooltip="The maximum number of subtasks to allow.",
                min_val=1,
                max_val=100,
            )
        )
        self.add_parameter(
            ParameterString(
                name="prompt",
                default_value=None,
                tooltip="The prompt to use",
                multiline=True,
                placeholder_text="Input text to process",
            )
        )
        self.add_parameter(
            ParameterList(
                name="context",
                tooltip="Additional context to add to the prompt",
                input_types=["Any"],
                allowed_modes={ParameterMode.INPUT},
            )
        )
        self.output = ParameterString(
            name="output",
            default_value=None,
            tooltip="The output of the task",
            allowed_modes={ParameterMode.OUTPUT},
            multiline=True,
            markdown=True,
            placeholder_text="The results of the MCP task will be displayed here.",
        )
        self.add_parameter(self.output)

        self._create_status_parameters(
            result_details_tooltip="Details about the MCP task execution result",
            result_details_placeholder="Details on the MCP task execution will be presented here.",
            parameter_group_initially_collapsed=True,
        )

    def _reload_mcp_servers(self, button: Button, button_details: ButtonDetailsMessagePayload) -> None:  # noqa: ARG002
        try:
            mcp_servers = get_available_mcp_servers()

            if mcp_servers:
                current_value = self.get_parameter_value("mcp_server_name")
                if current_value in mcp_servers:
                    default_value = current_value
                else:
                    default_value = mcp_servers[0]

                self._update_option_choices("mcp_server_name", mcp_servers, default_value)
                msg = f"{self.name}: Refreshed MCP servers: {len(mcp_servers)} servers available"
                logger.info(f"Refreshed MCP servers: {len(mcp_servers)} servers available")
            else:
                self._update_option_choices("mcp_server_name", ["No MCP servers available"], "No MCP servers available")
                msg = f"{self.name}: No MCP servers available"
                logger.info(msg)

        except Exception as e:
            msg = f"{self.name}: Failed to reload MCP servers: {e}"
            logger.error(msg)

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        """Keep the model dropdown's denial badge and choices in step with the selection."""
        super().after_value_set(parameter, value)
        self._model_access.on_value_set(parameter, value)
        if parameter.name == "model_provider":
            self._provider_selection.on_provider_changed(str(value))

    def after_incoming_connection(
        self, source_node: BaseNode, source_parameter: Parameter, target_parameter: Parameter
    ) -> None:
        # A connected agent carries its own model, so the node's own selectors no
        # longer decide anything -- hide them rather than leave a stale choice on display.
        if target_parameter.name == "agent":
            self._provider_selection.hide()

        if target_parameter.name == "model" and source_parameter.name == "prompt_model_config":
            # A connected Prompt Model Config supplies the driver, so the dropdown is no
            # longer a choice. Drop the Options trait (defensively, so reconnecting stays
            # idempotent) and switch the parameter to input-only, mirroring the Agent node.
            options_traits = target_parameter.find_elements_by_type(Options)
            if options_traits:
                target_parameter.remove_trait(trait_type=options_traits[0])

            target_parameter.type = source_parameter.type
            target_parameter.allowed_modes = {ParameterMode.INPUT}

            ui_options = target_parameter.ui_options
            ui_options["display_name"] = source_parameter.ui_options.get("display_name", source_parameter.name)
            target_parameter.ui_options = ui_options

        return super().after_incoming_connection(source_node, source_parameter, target_parameter)

    def after_incoming_connection_removed(
        self, source_node: BaseNode, source_parameter: Parameter, target_parameter: Parameter
    ) -> None:
        if target_parameter.name == "agent":
            self._provider_selection.show()

        if target_parameter.name == "model":
            # Reset the parameter type and re-enable PROPERTY so the user can set it again.
            target_parameter.type = "str"
            target_parameter.allowed_modes = {ParameterMode.INPUT, ParameterMode.PROPERTY}

            default_model = self._model_access.pick_permitted_default() or DEFAULT_MODEL
            target_parameter.set_default_value(default_model)
            target_parameter.default_value = default_model
            ui_options = target_parameter.ui_options
            ui_options["display_name"] = "prompt model"
            target_parameter.ui_options = ui_options
            self.set_parameter_value("model", default_model)
            # The connect hook stripped the Options trait; the component reinstalls its
            # Options trait, per-row license decoration, and badge.
            self._model_access.reinstall_options()

        return super().after_incoming_connection_removed(source_node, source_parameter, target_parameter)

    def validate_before_node_run(self) -> list[Exception] | None:
        exceptions = []

        # Get parameter values
        mcp_server_name = self.get_parameter_value("mcp_server_name")
        prompt = self.get_parameter_value("prompt")

        if not prompt:
            msg = f"{self.name}: No prompt provided. Please enter a prompt to process."
            exceptions.append(ValueError(msg))

        if mcp_server_name:
            is_valid, error_msg = validate_mcp_server(mcp_server_name)
            if not is_valid:
                msg = f"{self.name}: {error_msg}"
                exceptions.append(ValueError(msg))

        return exceptions if exceptions else None

    def process(self) -> AsyncResult:
        self._clear_execution_status()
        self._set_failure_output_values()
        self.publish_update_to_parameter("output", "")

        # OFFER_MODEL gate on the dropdown, re-queried against live policy so a selection
        # that was permitted when the node was built but has since been denied stops here --
        # before the MCP server connection below, which is the expensive part. Skipped when an
        # agent is connected (it brings its own model, so the hidden dropdown value is stale)
        # and when the provider is not Griptape Cloud (its models are outside the catalog the
        # policy gates). Routed through the status parameters rather than raised, matching how
        # this node reports every other failure; INVOKE_MODEL still gates the actual call.
        if self._provider_selection.uses_griptape_cloud_driver():
            denial = self._model_access.selection_denial()
            if denial is not None:
                self._set_status_results(was_successful=False, result_details=f"FAILURE: {denial.reason()}")
                logger.error(f"{self.name}: {denial.reason()}")
                return

        # Get parameter values
        mcp_server_name = self.get_parameter_value("mcp_server_name")
        prompt = self.get_parameter_value("prompt")
        context = self.get_parameter_list_value("context")

        if context:
            prompt += f"\n{context!s}"

        server_config = get_server_config(mcp_server_name)
        if server_config is None:
            error_details = f"MCP server '{mcp_server_name}' not found or not enabled"
            self._set_status_results(was_successful=False, result_details=f"FAILURE: {error_details}")
            logger.error(f"{self.name}: {error_details}")
            return

        setup = self._setup_agent()
        if setup is None:
            error_details = "Failed to setup agent"
            self._set_status_results(was_successful=False, result_details=f"FAILURE: {error_details}")
            logger.error(f"{self.name}: {error_details}")
            return
        state, model_config = setup

        mcp_ruleset = _ruleset_from_rules_string(server_config.get("rules"), mcp_server_name)
        rulesets = [*state.rulesets, mcp_ruleset] if mcp_ruleset else list(state.rulesets)
        mcp_tool_config = {
            "tool_type": ToolType.MCP,
            "mcp_server_name": mcp_server_name,
            "server_config": server_config,
        }

        yield lambda: self._execute_with_streaming(
            state, model_config, mcp_tool_config, rulesets, prompt, mcp_server_name
        )

    def _setup_agent(self) -> tuple[AgentState, ModelConfig] | None:
        """The incoming agent's state and the model this run uses, or None if either cannot be resolved."""
        try:
            agent_input = self.get_parameter_value("agent")
            state = AgentState.from_wire(agent_input) if is_agent_value(agent_input) else AgentState()
            model_config = state.model or self._create_model_config()
        except Exception as e:
            self._handle_failure_exception(e)
            return None
        return state, model_config

    def _build_toolsets(
        self, state: AgentState, mcp_tool_config: dict, mcp_server_name: str
    ) -> list[AbstractToolset[Any]]:
        """The incoming agent's toolsets plus the MCP server's."""
        mcp_toolset = build_toolset(mcp_tool_config)
        if mcp_toolset is None:
            msg = f"Failed to create MCP tool for server '{mcp_server_name}'"
            raise RuntimeError(msg)
        return [*build_toolsets(state.tools), mcp_toolset]

    def _execute_with_streaming(  # noqa: PLR0913
        self,
        state: AgentState,
        model_config: ModelConfig,
        mcp_tool_config: dict,
        rulesets: list[dict],
        prompt: str,
        mcp_server_name: str,
    ) -> None:
        try:
            toolsets = self._build_toolsets(state, mcp_tool_config, mcp_server_name)
            execution_start = time.time()
            logger.debug(f"MCPTaskNode '{self.name}': Starting agent execution with MCP tool...")
            result = self._process_with_streaming(model_config, state, toolsets, rulesets, prompt)
            execution_time = time.time() - execution_start
            logger.debug(f"MCPTaskNode '{self.name}': Agent execution completed in {execution_time:.2f}s")

            self._set_success_output_values(result, state, model_config)
            success_details = f"Successfully executed MCP task with server '{mcp_server_name}'"
            self._set_status_results(was_successful=True, result_details=f"SUCCESS: {success_details}")
            logger.info(f"MCPTaskNode '{self.name}': {success_details}")

        except Exception as execution_error:
            error_details = f"MCP task execution failed: {execution_error}"
            self._set_status_results(was_successful=False, result_details=f"FAILURE: {error_details}")
            self._handle_failure_exception(execution_error)

    def _process_with_streaming(
        self,
        model_config: ModelConfig,
        state: AgentState,
        toolsets: list[AbstractToolset[Any]],
        rulesets: list[dict],
        prompt: str,
    ) -> TaskRunResult:
        """Run the agent, streaming its text and tool use into `output`, like the Agent node."""
        # License-policy gate immediately before the model call, reading the model off the
        # config actually in use -- the node's own `model` parameter is not a trustworthy
        # source, since it keeps its last dropdown value (hidden, not cleared) while a
        # connected agent supplies the real model. The util resolves the provider model
        # id to its stable catalog key (via this node's `model_usage`) before declaring. This is
        # the fail-closed backstop behind the dropdown's OFFER_MODEL gate, and the only gate for
        # a model that never came from the dropdown at all.
        require_model_invocation_sync(self, model_config.model)

        def on_text(token: str) -> None:
            self.append_value_to_parameter("output", value=token)

        def on_tool_call(tool_name: str, _args: str) -> None:
            self.append_value_to_parameter("output", f"\n[Using tool {tool_name}]\n")

        return run_task_agent(
            model_config,
            prompt or None,
            rulesets=rulesets,
            toolsets=toolsets,
            message_history=state.messages,
            on_text=on_text,
            on_tool_call=on_tool_call,
        )

    def _create_model_config(self) -> ModelConfig:
        """The model config for the selected provider and model.

        A connected Prompt Model Config is the model, so it is used as-is. Otherwise the
        model comes from the dropdown: Griptape Cloud gets a Cloud config (the preset
        for the model family is applied when the model is built), and a third-party
        provider gets a config for that provider's endpoint.
        """
        model_input = self.get_parameter_value("model")
        connected = ModelConfig.from_wire(model_input)
        if connected is not None:
            return connected

        provider_name = self.get_parameter_value("model_provider") or "griptape_cloud"
        if provider_name != "griptape_cloud":
            providers = self._provider_selection._fetch_providers()
            provider_config = next((p for p in providers if p.name == provider_name), _GRIPTAPE_CLOUD_PROVIDER)
            return model_config_for_engine_provider(provider_config, str(model_input))

        # A model outside the Cloud catalog cannot be resolved to a catalog key, so the
        # license gate would fail closed on it. Fall back to the declared default instead.
        model = model_input if model_input in self._model_access.model_choices else DEFAULT_MODEL
        return cloud_model_config(str(model))

    def _set_success_output_values(self, result: TaskRunResult, state: AgentState, model_config: ModelConfig) -> None:
        self.parameter_output_values["output"] = result.text
        # The MCP toolset is rebuilt per run and not carried downstream, so the agent keeps the incoming tools.
        self.parameter_output_values["agent"] = AgentState(
            model=model_config, messages=compact_messages(result.messages), tools=state.tools, rulesets=state.rulesets
        ).to_wire()

    def _set_failure_output_values(self) -> None:
        self.parameter_output_values["output"] = ""
