import json
from typing import Any, cast

from griptape.artifacts import ImageUrlArtifact, ModelArtifact
from griptape.drivers.prompt.base_prompt_driver import BasePromptDriver
from griptape.drivers.prompt.griptape_cloud_prompt_driver import GriptapeCloudPromptDriver
from griptape.drivers.prompt.openai import OpenAiChatPromptDriver as GtOpenAiChatPromptDriver
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
from griptape_nodes.exe_types.param_components.model_access_component import ModelAccessComponent
from griptape_nodes.exe_types.param_types.parameter_bool import ParameterBool
from griptape_nodes.exe_types.param_types.parameter_json import ParameterJson
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.retained_mode.events.agent_events import (
    ListAgentProvidersRequest,
    ListAgentProvidersResultSuccess,
    ListProviderModelsRequest,
    ListProviderModelsResultSuccess,
    ProviderConfig,
)
from griptape_nodes.retained_mode.events.connection_events import CreateConnectionRequest, DeleteConnectionRequest
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes, logger
from griptape_nodes.traits.button import Button, ButtonDetailsMessagePayload
from griptape_nodes.traits.options import Options
from json_schema_to_pydantic import create_model  # pyright: ignore[reportMissingImports]

from griptape_nodes_library.agents.griptape_nodes_agent import GriptapeNodesAgent as GtAgent
from griptape_nodes_library.config.prompt.cloud_models import MODEL_CHOICES_ARGS
from griptape_nodes_library.utils.agent_utils import (
    build_rulesets_from_configs,
    build_tools,
    restore_provider_driver,
    unwrap_agent,
    wrap_agent,
)
from griptape_nodes_library.utils.error_utils import try_throw_error
from griptape_nodes_library.utils.image_utils import load_image_from_url_artifact
from griptape_nodes_library.utils.model_invocation import declare_model_invocation_sync

_GRIPTAPE_CLOUD_PROVIDER = ProviderConfig(name="griptape_cloud", type="griptape_cloud", model="")

SERVICE = "Griptape"
API_KEY_URL = "https://cloud.griptape.ai/configuration/api-keys"
API_KEY_ENV_VAR = "GT_CLOUD_API_KEY"

# Vision-capable models available on Griptape Cloud.
GTC_VISION_MODEL_CHOICES = [
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
DEFAULT_MODEL = GTC_VISION_MODEL_CHOICES[0]


class DescribeImage(ControlNode):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

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

        model_param = Parameter(
            name="model",
            input_types=["str", "Prompt Model Config"],
            type="str",
            output_type="str",
            default_value=DEFAULT_MODEL,
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            tooltip="Choose a model, or connect a Prompt Model Configuration or an Agent",
            ui_options={"display_name": "prompt model"},
        )
        self.add_parameter(model_param)
        # License-policy helper: adds Options + refresh Button traits, applies per-row
        # decoration + badge, exposes query_for_denial / raise_if_denied, and
        # relocates the stored value to a permitted alternative if DEFAULT_MODEL is denied.
        self._model_access = ModelAccessComponent(
            node=self,
            parameter=model_param,
            model_choices=GTC_VISION_MODEL_CHOICES,
            default_model=DEFAULT_MODEL,
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

    # --- Provider / model helpers (mirrors agent.py) ---
    # TODO: extract into ProviderSelectionComponent shared with Agent
    # https://github.com/griptape-ai/griptape-nodes-library-standard/issues/442

    def _fetch_providers(self) -> list[ProviderConfig]:
        _FALLBACK = [_GRIPTAPE_CLOUD_PROVIDER]
        try:
            result = GriptapeNodes.handle_request(ListAgentProvidersRequest())
            if not isinstance(result, ListAgentProvidersResultSuccess):
                return _FALLBACK
            return cast(ListAgentProvidersResultSuccess, result).providers or _FALLBACK
        except Exception:
            return _FALLBACK

    def _fetch_provider_names(self) -> list[str]:
        providers = self._fetch_providers()
        return [p.name for p in providers] or ["griptape_cloud"]

    def _resolve_provider_api_key(self, provider_config: "ProviderConfig") -> str:
        secret_name = provider_config.api_key_secret_name or ""
        if secret_name:
            return (
                GriptapeNodes.SecretsManager().get_secret(secret_name, should_error_on_not_found=False) or "not-needed"
            )
        return "not-needed"

    def _fetch_models_for_provider(self, provider_name: str) -> list[str]:
        try:
            providers = self._fetch_providers()
            provider_config = next((p for p in providers if p.name == provider_name), None)
            if provider_config is None:
                return GTC_VISION_MODEL_CHOICES
            result = GriptapeNodes.handle_request(
                ListProviderModelsRequest(
                    provider=provider_config.type,
                    base_url=provider_config.base_url or "",
                    api_key=self._resolve_provider_api_key(provider_config),
                )
            )
            if isinstance(result, ListProviderModelsResultSuccess):
                return cast(ListProviderModelsResultSuccess, result).models or GTC_VISION_MODEL_CHOICES
        except Exception:
            pass
        return GTC_VISION_MODEL_CHOICES

    def _update_model_choices_for_provider(self, provider_name: str) -> None:
        if provider_name == "griptape_cloud":
            # Use a curated vision-only subset rather than the full GTC model list.
            models = GTC_VISION_MODEL_CHOICES
            vision_names = set(GTC_VISION_MODEL_CHOICES)
            new_data = [entry for entry in MODEL_CHOICES_ARGS if entry["name"] in vision_names]
        else:
            models = self._fetch_models_for_provider(provider_name)
            new_data = [{"name": m, "icon": "", "args": {}} for m in models]
        default = models[0] if models else DEFAULT_MODEL
        self._update_option_choices(param="model", choices=models, default=default)
        param = self.get_parameter_by_name("model")
        if param:
            param.update_ui_options_key("data", new_data)

    def _refresh_providers_button(
        self,
        button: Button,
        button_details: ButtonDetailsMessagePayload,  # noqa: ARG002
    ) -> NodeMessageResult | None:
        provider_names = self._fetch_provider_names()
        current = self.get_parameter_value("model_provider") or "griptape_cloud"
        default = current if current in provider_names else (provider_names[0] if provider_names else "griptape_cloud")
        self._update_option_choices(param="model_provider", choices=provider_names, default=default)
        return None

    def _uses_griptape_cloud_driver(self) -> bool:
        if self.get_parameter_value("agent") is not None:
            return False
        if isinstance(self.get_parameter_value("model"), BasePromptDriver):
            return False
        provider_name = self.get_parameter_value("model_provider") or "griptape_cloud"
        return provider_name == "griptape_cloud"

    # --- Connection / UI helpers ---

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
        exceptions = []
        if not self._uses_griptape_cloud_driver():
            return None
        api_key = GriptapeNodes.SecretsManager().get_secret(API_KEY_ENV_VAR)
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
            self.hide_parameter_by_name("model_provider")

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
            self.show_parameter_by_name("model_provider")
        if target_parameter.name == "output_schema":
            self.set_parameter_value("output_schema", None)
            self._update_output_type_and_validate_connections("str")
        # Check and see if the incoming connection is from an agent. If so, we'll hide the model parameter
        if target_parameter.name == "model":
            target_parameter.type = "str"
            # Enable PROPERTY so the user can set it
            target_parameter.allowed_modes = {ParameterMode.INPUT, ParameterMode.PROPERTY}

            default_model = self._model_access.pick_permitted_default() or DEFAULT_MODEL
            target_parameter.set_default_value(default_model)
            target_parameter.default_value = default_model
            ui_options = target_parameter.ui_options
            ui_options["display_name"] = "prompt model"
            target_parameter.ui_options = ui_options
            self.set_parameter_value("model", default_model)
            # Helper reinstalls its Options trait + decoration + badge on the freshly-uncovered
            # parameter (the incoming-connection handler stripped Options when the driver connected).
            self._model_access.reinstall_options()

        return super().after_incoming_connection_removed(source_node, source_parameter, target_parameter)

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        super().after_value_set(parameter, value)
        if parameter.name == "model":
            self._model_access.on_value_changed(value)
        elif parameter.name == "model_provider":
            self._update_model_choices_for_provider(str(value))

    def process(self) -> AsyncResult[Structure]:  # noqa: C901, PLR0915, PLR0912
        # Get the parameters from the node
        params = self.parameter_values
        model_input = self.get_parameter_value("model")
        provider_name = self.get_parameter_value("model_provider") or "griptape_cloud"
        agent_value = self.get_parameter_value("agent")

        # License-policy runtime gate, skipped when an Agent is connected: it supplies its
        # own driver, so the node's (hidden, not cleared) dropdown value is stale. The
        # INVOKE_MODEL declaration below gates the model that actually runs.
        if agent_value is None:
            self._model_access.raise_if_denied(model_input)

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

        tool_configs: list = []
        ruleset_configs: list = []
        provider_info: dict | None = None
        if isinstance(agent_value, dict):
            agent_core_dict, tool_configs, ruleset_configs = unwrap_agent(agent_value)
            agent = GtAgent().from_dict(agent_core_dict)
            restore_provider_driver(agent, agent_value)
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
        elif provider_name != "griptape_cloud":
            providers = self._fetch_providers()
            non_gtc_provider_config = next((p for p in providers if p.name == provider_name), None)
            if non_gtc_provider_config is None:
                msg = f"DescribeImage '{self.name}': provider '{provider_name}' not found in configured providers."
                raise ValueError(msg)
            api_key = self._resolve_provider_api_key(non_gtc_provider_config)
            base_url = non_gtc_provider_config.base_url or ""
            prompt_driver = GtOpenAiChatPromptDriver(
                model=model_input if isinstance(model_input, str) else DEFAULT_MODEL,
                base_url=base_url,
                api_key=api_key,
                stream=True,
            )
            provider_info = {"name": provider_name, "base_url": base_url, "api_key": api_key}
            agent = GtAgent(prompt_driver=prompt_driver, output_schema=pydantic_schema)
        elif isinstance(model_input, str):
            if model_input not in self._model_access.model_choices:
                model_input = DEFAULT_MODEL
            prompt_driver = GriptapeCloudPromptDriver(
                model=model_input,
                api_key=GriptapeNodes.SecretsManager().get_secret(API_KEY_ENV_VAR),
                stream=False,  # TODO: enable once https://github.com/griptape-ai/griptape-cloud/issues/1593 is resolved
            )
            agent = GtAgent(prompt_driver=prompt_driver, output_schema=pydantic_schema)
        else:
            agent = GtAgent(prompt_driver=default_prompt_driver, output_schema=pydantic_schema)

        prompt = params.get("prompt", "")
        if prompt == "":
            prompt = "Describe the image"

        get_description_only = self.get_parameter_value("description_only")
        if get_description_only:
            prompt += "\n\nOutput image description only."

        # Flatten nested lists — a ParameterList child may receive a list of artifacts
        # when connected to an output that produces List[ImageUrlArtifact].
        raw_images = self.get_parameter_value("images") or []
        flat_images: list = []
        for img in raw_images:
            if isinstance(img, list):
                flat_images.extend(img)
            else:
                flat_images.append(img)

        image_artifacts = []
        for img in flat_images:
            if img is None or img == "":
                continue
            if isinstance(img, ImageUrlArtifact):
                image_artifacts.append(load_image_from_url_artifact(img))
            elif isinstance(img, str) and img.strip():
                # String path or URL — load as image bytes rather than passing as text.
                image_artifacts.append(load_image_from_url_artifact(ImageUrlArtifact(img)))
            else:
                image_artifacts.append(img)

        if not image_artifacts:
            self.parameter_output_values["output"] = "No image provided"
            return

        # Declare the model that will actually run. Every construction branch above
        # ends with the concrete prompt driver installed on the agent's PromptTask,
        # so read the model from there. The node's own `model` parameter is not a
        # trustworthy source: it keeps its last dropdown value (hidden, not cleared)
        # while a connected Agent supplies the real driver. The util resolves the
        # provider model id to its stable catalog key (via the node's model_usage)
        # before declaring. Declare before the network call below so a denied
        # invocation fails closed rather than reaching the provider.
        model = cast(PromptTask, agent.tasks[0]).prompt_driver.model
        declaration = declare_model_invocation_sync(self, model)
        if declaration.failed():
            details = str(
                declaration.result_details
                or f"DescribeImage '{self.name}': invocation of model '{model}' was not permitted."
            )
            raise RuntimeError(details)

        # Run the agent
        yield lambda: agent.run([prompt, *image_artifacts])
        agent_output = agent.output
        output_value = agent_output.value
        if isinstance(agent_output, ModelArtifact):
            output_value = agent_output.value.model_dump()

        self.parameter_output_values["output"] = output_value

        # Replace the run's image bytes with text — without this, those bytes get serialized
        # into conversation history and resent on every downstream API call.
        memory_output = output_value
        if isinstance(memory_output, (dict, list)):
            memory_output = json.dumps(memory_output, ensure_ascii=False)
        agent.insert_false_memory(prompt=prompt, output=str(memory_output))
        try_throw_error(agent.output)

        # Clear live tools before serializing, then wrap with configs for downstream nodes.
        if agent.tasks:
            cast(PromptTask, agent.tasks[0]).tools = []

        incoming_provider = agent_value.get("provider") if isinstance(agent_value, dict) else None

        self.parameter_output_values["agent"] = wrap_agent(
            agent.to_dict(),
            tool_configs,
            ruleset_configs,
            provider=provider_info or incoming_provider,
        )
