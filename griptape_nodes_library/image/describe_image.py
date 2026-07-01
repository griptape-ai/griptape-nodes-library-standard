import json
from typing import Any, cast

from griptape.artifacts import ImageUrlArtifact, ModelArtifact
from griptape.drivers.prompt.base_prompt_driver import BasePromptDriver
from griptape.drivers.prompt.griptape_cloud_prompt_driver import GriptapeCloudPromptDriver
from griptape.structures import Structure
from griptape.tasks import PromptTask
from griptape_nodes.exe_types.core_types import (
    NodeMessageResult,
    Parameter,
    ParameterList,
    ParameterMode,
    ParameterType,
)
from griptape_nodes.exe_types.node_types import AsyncResult, BaseNode, ControlNode
from griptape_nodes.exe_types.param_types.parameter_bool import ParameterBool
from griptape_nodes.exe_types.param_types.parameter_json import ParameterJson
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.retained_mode.events.access_events import (
    QueryModelAccessForNodeRequest,
    QueryModelAccessForNodeResultSuccess,
)
from griptape_nodes.retained_mode.events.connection_events import CreateConnectionRequest, DeleteConnectionRequest
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes, logger
from griptape_nodes.retained_mode.managers.authorization_checkpoint import CheckpointDenial
from griptape_nodes.traits.button import Button, ButtonDetailsMessagePayload
from griptape_nodes.traits.options import Options
from json_schema_to_pydantic import create_model  # pyright: ignore[reportMissingImports]

from griptape_nodes_library.agents.griptape_nodes_agent import GriptapeNodesAgent as GtAgent
from griptape_nodes_library.utils.agent_utils import (
    build_rulesets_from_configs,
    build_tools,
    restore_provider_driver,
    unwrap_agent,
    wrap_agent,
)
from griptape_nodes_library.utils.error_utils import try_throw_error
from griptape_nodes_library.utils.image_utils import load_image_from_url_artifact

SERVICE = "Griptape"
API_KEY_URL = "https://cloud.griptape.ai/configuration/api-keys"
API_KEY_ENV_VAR = "GT_CLOUD_API_KEY"
MODEL_CHOICES = [
    "gpt-5.2",
    "gpt-5.1",
    "gpt-5",
    "gpt-5-mini",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-4o",
    "o4-mini",
    "o3",
    "claude-opus-4-7",
    "claude-sonnet-4-6",
    "claude-4-5-sonnet",
    "gemini-3.1-pro",
    "gemini-2.5-pro",
]
DEFAULT_MODEL = MODEL_CHOICES[0]


class DescribeImage(ControlNode):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # Cache the per-provider denial map at node-init so after_value_set can
        # toggle the badge without a round-trip on every selection change.
        # process() re-asks the engine so a hook decision that flipped between
        # creation and run still wins.
        self._denial_by_provider_id: dict[str, CheckpointDenial] = self._fetch_denial_map()
        default_model = self._pick_default_model()

        self.add_parameter(
            Parameter(
                name="agent",
                type="Agent",
                output_type="Agent",
                tooltip="An agent that will be used to describe the image(s).",
                default_value=None,
                allowed_modes={ParameterMode.INPUT, ParameterMode.OUTPUT},
            )
        )
        self.add_parameter(
            Parameter(
                name="model",
                input_types=["str", "Prompt Model Config"],
                type="str",
                output_type="str",
                default_value=default_model,
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                tooltip="Choose a model, or connect a Prompt Model Configuration or an Agent",
                traits={
                    Options(choices=list(MODEL_CHOICES)),
                    Button(
                        icon="list-restart",
                        size="icon",
                        variant="secondary",
                        on_click=self._on_refresh_access_click,
                        tooltip="Refresh available models",
                    ),
                },
                ui_options=self._model_ui_options(),
            )
        )
        self.add_parameter(
            ParameterList(
                name="images",
                input_types=["ImageUrlArtifact", "ImageArtifact", "str"],
                default_value=None,
                tooltip="The image(s) to be described",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"display_name": "image(s)", "collapsed": True},
            )
        )
        self.add_parameter(
            ParameterString(
                name="prompt",
                tooltip="Explain how you'd like the image(s) to be described.",
                default_value="",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                placeholder_text="Explain the various aspects of the image(s) you want to be described.",
                multiline=True,
                ui_options={"display_name": "description prompt"},
            ),
        )

        self.add_parameter(
            ParameterBool(
                name="description_only",
                tooltip="Only return the description of the image, no conversation",
                default_value=True,
            )
        )

        # Parameter for output schema
        self.add_parameter(
            ParameterJson(
                name="output_schema",
                tooltip="Optional JSON schema for structured output validation.",
                default_value=None,
                allowed_modes={ParameterMode.INPUT},
                hide_property=True,
            )
        )
        self.add_parameter(
            ParameterString(
                name="output",
                tooltip="None",
                default_value=None,
                allowed_modes={ParameterMode.OUTPUT},
                multiline=True,
                placeholder_text="The description of the image",
                ui_options={"display_name": "output"},
            )
        )

        self.add_parameter(
            Parameter(
                name="image",
                input_types=["ImageUrlArtifact", "ImageArtifact"],
                default_value=None,
                tooltip="Deprecated. Use images.",
                allowed_modes={ParameterMode.INPUT},
                hide=True,
            )
        )

        # Reflect the initial selection: a node born with a denied default
        # gets the badge immediately.
        self._update_model_access_badge(default_model)

    def _fetch_denial_map(self) -> dict[str, CheckpointDenial]:
        """Return `{provider_model_id: CheckpointDenial}` for every denied catalog model.

        Asks the engine which of the node's declared catalog models the
        authorization hook denies, keyed by the upstream provider's name (which
        is what the dropdown stores). Also populates ``_catalog_id_by_provider_id``
        so the runtime check in ``process()`` can translate the dropdown name back
        to the catalog id the engine policy actually matches on. Empty on engine
        failure -- internal errors must not silently strip the dropdown.
        """
        self._catalog_id_by_provider_id: dict[str, str] = {}
        result = GriptapeNodes.handle_request(QueryModelAccessForNodeRequest(node_type=type(self).__name__))
        if not isinstance(result, QueryModelAccessForNodeResultSuccess):
            return {}
        denials: dict[str, CheckpointDenial] = {}
        for verdict in result.verdicts:
            if verdict.provider_model_id is not None:
                self._catalog_id_by_provider_id[verdict.provider_model_id] = verdict.model_id
                if verdict.denial is not None:
                    denials[verdict.provider_model_id] = verdict.denial
        return denials

    def _pick_default_model(self) -> str:
        """Return `DEFAULT_MODEL` if it's allowed; otherwise the first allowed entry.

        Falls back to `DEFAULT_MODEL` when every entry is denied so the dropdown
        still has a value the user can see and the warning notice can render.
        """
        if DEFAULT_MODEL not in self._denial_by_provider_id:
            return DEFAULT_MODEL
        for choice in MODEL_CHOICES:
            if choice not in self._denial_by_provider_id:
                return choice
        return DEFAULT_MODEL

    def _model_ui_options(self) -> dict[str, Any]:
        """Build the `ui_options` dict for the model parameter, including per-row decoration."""
        data: list[dict[str, str]] = []
        for choice in MODEL_CHOICES:
            if choice in self._denial_by_provider_id:
                data.append({"name": choice, "icon": "shield-off", "subtitle": "Not permitted by your license"})
            else:
                data.append({"name": choice})
        return {
            "display_name": "prompt model",
            "data": data,
            "dropdown_row_icons": True,
            "dropdown_row_subtitles": True,
        }

    def _fetch_runtime_denial(self, model_provider_id: str) -> CheckpointDenial | None:
        """Ask the engine whether this specific model is permitted, right now.

        Translates the dropdown name (e.g. ``"gpt-5.2"``) to the catalog id the
        engine policy matches on (e.g. ``"gtc_gpt_5_2"``) via the mapping built
        at init time. Falls through to ``None`` on any failure -- a missing
        mapping means the dropdown got out of sync with the manifest; an engine
        ``Failure`` means an internal error. In neither case do we want to gate
        user work.
        """
        catalog_id = self._catalog_id_by_provider_id.get(model_provider_id)
        if catalog_id is None:
            return None
        result = GriptapeNodes.handle_request(
            QueryModelAccessForNodeRequest(
                node_type=type(self).__name__,
                candidate_model_ids=[catalog_id],
            )
        )
        if not isinstance(result, QueryModelAccessForNodeResultSuccess) or not result.verdicts:
            return None
        return result.verdicts[0].denial

    def _on_refresh_access_click(
        self, _button: Button, _button_details: ButtonDetailsMessagePayload
    ) -> NodeMessageResult | None:
        """Re-query the engine and refresh the dropdown decoration + current-selection badge.

        Fires when the user clicks the inline refresh button next to the model
        dropdown. Useful when a license / permission state has changed under the
        running engine (e.g. the user upgraded their plan) and the artist wants
        the dropdown to reflect it without recreating the node or reloading the
        workflow.
        """
        self._denial_by_provider_id = self._fetch_denial_map()
        model_param = self.get_parameter_by_name("model")
        if model_param is not None:
            model_param.update_ui_options(self._model_ui_options())
        self._update_model_access_badge(self.get_parameter_value("model"))
        return None

    def _update_output_type_and_validate_connections(self, new_output_type: str) -> None:
        output_param = self.get_parameter_by_name("output")
        if output_param is None:
            return

        output_param.output_type = new_output_type
        output_param.type = new_output_type

        connections = GriptapeNodes.FlowManager().get_connections()
        outgoing_for_node = connections.outgoing_index.get(self.name, {})
        connection_ids = outgoing_for_node.get("output", [])

        for connection_id in connection_ids:
            connection = connections.connections[connection_id]
            target_param = connection.target_parameter
            target_node = connection.target_node

            is_compatible = any(
                ParameterType.are_types_compatible(new_output_type, input_type)
                for input_type in target_param.input_types
            )

            if not is_compatible:
                logger.info(
                    f"Removing incompatible connection: DescribeImage '{self.name}' output ({new_output_type}) to "
                    f"'{target_node.name}.{target_param.name}' (accepts: {target_param.input_types})"
                )

                GriptapeNodes.handle_request(
                    DeleteConnectionRequest(
                        source_node_name=self.name,
                        source_parameter_name="output",
                        target_node_name=target_node.name,
                        target_parameter_name=target_param.name,
                    )
                )

    def set_parameter_value(
        self,
        param_name: str,
        value: Any,
        *,
        initial_setup: bool = False,
        emit_change: bool = True,
        skip_before_value_set: bool = False,
    ) -> None:
        if param_name == "image" and value is not None:
            logger.info(
                f"DescribeImage '{self.name}': 'image' parameter is deprecated. Migrating value to 'images' parameter."
            )
            images_list = self.get_parameter_by_name("images")
            assert isinstance(images_list, ParameterList)
            child = images_list.add_child_parameter()
            connections = GriptapeNodes.FlowManager().get_connections()
            image_conn_ids = connections.incoming_index.get(self.name, {}).get("image", [])
            if image_conn_ids:
                conn = connections.connections[image_conn_ids[0]]
                GriptapeNodes.handle_request(
                    DeleteConnectionRequest(
                        source_node_name=conn.source_node.name,
                        source_parameter_name=conn.source_parameter.name,
                        target_node_name=self.name,
                        target_parameter_name="image",
                    )
                )
                GriptapeNodes.handle_request(
                    CreateConnectionRequest(
                        source_node_name=conn.source_node.name,
                        source_parameter_name=conn.source_parameter.name,
                        target_node_name=self.name,
                        target_parameter_name=child.name,
                    )
                )
            super().set_parameter_value(
                child.name,
                value,
                initial_setup=initial_setup,
                emit_change=emit_change,
                skip_before_value_set=skip_before_value_set,
            )
            return
        super().set_parameter_value(
            param_name,
            value,
            initial_setup=initial_setup,
            emit_change=emit_change,
            skip_before_value_set=skip_before_value_set,
        )

    def validate_before_workflow_run(self) -> list[Exception] | None:
        # TODO: https://github.com/griptape-ai/griptape-nodes/issues/871
        exceptions = []
        api_key = GriptapeNodes.SecretsManager().get_secret(API_KEY_ENV_VAR)
        # No need for the api key. These exceptions caught on other nodes.
        if self.parameter_values.get("agent", None) and self.parameter_values.get("driver", None):
            return None
        if not api_key:
            msg = f"{API_KEY_ENV_VAR} is not defined"
            exceptions.append(KeyError(msg))
            return exceptions
        return exceptions if exceptions else None

    def after_incoming_connection(
        self,
        source_node: BaseNode,
        source_parameter: Parameter,
        target_parameter: Parameter,
    ) -> None:
        if target_parameter.name == "agent":
            self.hide_parameter_by_name("model")

        if target_parameter.name == "output_schema":
            self._update_output_type_and_validate_connections("json")

        if target_parameter.name == "model" and source_parameter.name == "prompt_model_config":
            # Check and see if the incoming connection is from a prompt model config or an agent.
            target_parameter.type = source_parameter.type
            # Remove ParameterMode.PROPERTY so it forces the node mark itself dirty & remove the value
            target_parameter.allowed_modes = {ParameterMode.INPUT}

            target_parameter.remove_trait(trait_type=target_parameter.find_elements_by_type(Options)[0])
            ui_options = target_parameter.ui_options
            ui_options["display_name"] = source_parameter.ui_options.get("display_name", source_parameter.name)
            target_parameter.ui_options = ui_options

        return super().after_incoming_connection(source_node, source_parameter, target_parameter)

    def after_incoming_connection_removed(
        self,
        source_node: BaseNode,
        source_parameter: Parameter,
        target_parameter: Parameter,
    ) -> None:
        if target_parameter.name == "agent":
            self.show_parameter_by_name("model")
        if target_parameter.name == "output_schema":
            self.set_parameter_value("output_schema", None)
            self._update_output_type_and_validate_connections("str")
        # Check and see if the incoming connection is from an agent. If so, we'll hide the model parameter
        if target_parameter.name == "model":
            target_parameter.type = "str"
            # Enable PROPERTY so the user can set it
            target_parameter.allowed_modes = {ParameterMode.INPUT, ParameterMode.PROPERTY}

            # Refresh the cached denial map first so the reinstalled dropdown reflects
            # the current policy, then reinstall Options with the full choice set --
            # denied entries stay in the list with their `shield-off` decoration via
            # _model_ui_options(), matching the parameter's original construction.
            self._denial_by_provider_id = self._fetch_denial_map()
            default_model = self._pick_default_model()
            target_parameter.add_trait(Options(choices=list(MODEL_CHOICES)))
            target_parameter.set_default_value(default_model)
            target_parameter.default_value = default_model
            target_parameter.update_ui_options(self._model_ui_options())
            self.set_parameter_value("model", default_model)

        return super().after_incoming_connection_removed(source_node, source_parameter, target_parameter)

    def _update_model_access_badge(self, value: Any) -> None:
        """Set or clear the `model` parameter's badge based on the selected value's policy verdict.

        Reads the cached ``_denial_by_provider_id`` (built at init) so this is a
        local map lookup, not a round-trip to the engine. ``process()`` re-asks
        the engine at run-time so a hook decision that changed between node
        creation and run is still enforced.
        """
        param = self.get_parameter_by_name("model")
        if param is None:
            return
        if not isinstance(value, str):
            # Driver / Agent connection in flight -- the dropdown isn't the truth.
            param.clear_badge()
            return
        denial = self._denial_by_provider_id.get(value)
        if denial is None:
            param.clear_badge()
            return
        param.set_badge(
            variant="error",
            title="Model Not Permitted",
            message=f"Model `{value}` is not permitted. Running this node will fail.\n\nReason(s): {denial.reason()}",
            icon="shield-off",
        )

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        super().after_value_set(parameter, value)
        if parameter.name == "model":
            self._update_model_access_badge(value)

    def process(self) -> AsyncResult[Structure]:  # noqa: C901, PLR0915, PLR0912
        # Get the parameters from the node
        params = self.parameter_values
        model_input = self.get_parameter_value("model")

        # Runtime entitlement gate. We re-ask the engine rather than reading the
        # cached _denial_by_provider_id so a hook decision that changed between
        # node creation and run is honored. Only checks string-valued model
        # selections; connected driver objects (BasePromptDriver) bypass this --
        # they carry their own model identity that the node doesn't introspect.
        if isinstance(model_input, str):
            denial = self._fetch_runtime_denial(model_input)
            if denial is not None:
                msg = f"Cannot run {type(self).__name__}: '{model_input}' is not permitted. {denial.reason()}"
                raise RuntimeError(msg)

        agent = None

        default_prompt_driver = GriptapeCloudPromptDriver(
            model=DEFAULT_MODEL,
            api_key=GriptapeNodes.SecretsManager().get_secret(API_KEY_ENV_VAR),
            stream=False,  # TODO: enable once https://github.com/griptape-ai/griptape-cloud/issues/1593 is resolved
        )

        output_schema = self.get_parameter_value("output_schema")
        pydantic_schema = None
        if output_schema is not None:
            schema_value = output_schema
            if isinstance(schema_value, str):
                if not schema_value.strip():
                    schema_value = None
                else:
                    try:
                        schema_value = json.loads(schema_value)
                    except json.JSONDecodeError as e:
                        msg = (
                            f"DescribeImage '{self.name}': Unable to parse output_schema as JSON: {e}. "
                            "Try using the `Create Agent Schema` node to generate a schema."
                        )
                        logger.error(msg)
                        raise

            if schema_value is not None and not isinstance(schema_value, dict):
                msg = (
                    f"DescribeImage '{self.name}': output_schema must be a JSON schema object (dict) "
                    f"or a JSON string, got: {type(schema_value).__name__}"
                )
                logger.error(msg)
                raise TypeError(msg)

            if schema_value is not None:
                try:
                    pydantic_schema = create_model(schema_value)
                except Exception as e:
                    msg = (
                        f"DescribeImage '{self.name}': Unable to create output schema model: {e}. "
                        "Try using the `Create Agent Schema` node to generate a schema."
                    )
                    logger.error(msg)
                    raise

        # If an agent is provided, we'll use and ensure it's using a PromptTask
        # If a prompt_driver is provided, we'll use that
        # If neither are provided, we'll create a new one with the selected model.
        # Otherwise, we'll just use the default model
        tool_configs: list = []
        ruleset_configs: list = []
        agent_value = self.get_parameter_value("agent")
        if isinstance(agent_value, dict):
            agent_core_dict, tool_configs, ruleset_configs = unwrap_agent(agent_value)
            agent = GtAgent().from_dict(agent_core_dict)
            restore_provider_driver(agent, agent_value)
            # Rebuild tools and rulesets from configs so they're fresh for this run.
            if tool_configs:
                live_tools, _ = build_tools(tool_configs)
                if live_tools and agent.tasks:
                    cast(PromptTask, agent.tasks[0]).tools = live_tools
            if ruleset_configs:
                agent._rulesets = build_rulesets_from_configs(ruleset_configs)
            # make sure the agent is using a PromptTask — replace rather than add to avoid two tasks
            if not isinstance(agent.tasks[0], PromptTask):
                agent.tasks[0] = PromptTask(prompt_driver=default_prompt_driver, output_schema=pydantic_schema)
            else:
                agent.tasks[0].output_schema = pydantic_schema
        elif isinstance(model_input, BasePromptDriver):
            agent = GtAgent(prompt_driver=model_input, output_schema=pydantic_schema)
        elif isinstance(model_input, str):
            if model_input not in MODEL_CHOICES:
                model_input = DEFAULT_MODEL
            prompt_driver = GriptapeCloudPromptDriver(
                model=model_input,
                api_key=GriptapeNodes.SecretsManager().get_secret(API_KEY_ENV_VAR),
                stream=False,  # TODO: enable once https://github.com/griptape-ai/griptape-cloud/issues/1593 is resolved
            )
            agent = GtAgent(prompt_driver=prompt_driver, output_schema=pydantic_schema)
        else:
            # If the agent is not provided, we'll create a new one with a default prompt driver
            agent = GtAgent(prompt_driver=default_prompt_driver, output_schema=pydantic_schema)

        prompt = params.get("prompt", "")
        if prompt == "":
            prompt = "Describe the image"

        get_description_only = self.get_parameter_value("description_only")
        if get_description_only:
            prompt += "\n\nOutput image description only."

        image_artifacts = [
            load_image_from_url_artifact(img) if isinstance(img, ImageUrlArtifact) else img
            for img in (self.get_parameter_value("images") or [])
            if img is not None
        ]

        if not image_artifacts:
            self.parameter_output_values["output"] = "No image provided"
            return

        # Run the agent
        yield lambda: agent.run([prompt, *image_artifacts])
        agent_output = agent.output
        output_value = agent_output.value
        if isinstance(agent_output, ModelArtifact):
            output_value = agent_output.value.model_dump()

        self.parameter_output_values["output"] = output_value

        # Insert a false memory to prevent the base64
        memory_output = output_value
        if isinstance(memory_output, (dict, list)):
            memory_output = json.dumps(memory_output, ensure_ascii=False)
        agent.insert_false_memory(prompt=prompt, output=str(memory_output))
        try_throw_error(agent.output)

        # Clear live tools before serializing, then wrap with configs for downstream nodes.
        if agent.tasks:
            cast(PromptTask, agent.tasks[0]).tools = []
        self.parameter_output_values["agent"] = wrap_agent(
            agent.to_dict(),
            tool_configs,
            ruleset_configs,
            provider=agent_value.get("provider") if isinstance(agent_value, dict) else None,
        )
