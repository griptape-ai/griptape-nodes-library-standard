import json
from dataclasses import replace
from typing import Any

from griptape_nodes.drivers.cloud_models import VISION_MODEL_CHOICES
from griptape_nodes.exe_types.core_types import (
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
from griptape_nodes.retained_mode.events.agent_events import ProviderConfig
from griptape_nodes.retained_mode.events.connection_events import CreateConnectionRequest, DeleteConnectionRequest
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes, logger
from griptape_nodes.traits.options import Options
from pydantic_ai.agent import AgentRunResult

from griptape_nodes_library.llm.agent_state import AgentState, connected_agent_state, messages_from_runs
from griptape_nodes_library.llm.content import image_content
from griptape_nodes_library.llm.model_config import (
    ModelConfig,
    ModelProvider,
    model_config_for_engine_provider,
    model_config_from_input,
)
from griptape_nodes_library.llm.runner import output_to_text, output_type_from_schema, run_agent
from griptape_nodes_library.llm.tools import build_agent_from_state
from griptape_nodes_library.utils.cloud_credential_utils import (
    missing_credential_message,
    resolve_cloud_api_key,
)
from griptape_nodes_library.utils.cloud_legacy_models import cloud_legacy_values_for
from griptape_nodes_library.utils.model_invocation import require_model_invocation_sync
from griptape_nodes_library.utils.provider_selection_component import ProviderSelectionComponent

SERVICE = "Griptape"
API_KEY_URL = "https://cloud.griptape.ai/configuration/api-keys"
API_KEY_ENV_VAR = "GT_CLOUD_API_KEY"

# Vision-capable models available on Griptape Cloud, derived from the catalog's
# per-model `vision` flag so this list cannot drift from the models it describes.
GTC_VISION_MODEL_CHOICES = VISION_MODEL_CHOICES
DEFAULT_MODEL = GTC_VISION_MODEL_CHOICES[0]

# The shared Griptape Cloud table, narrowed to the vision-capable models this node offers.
LEGACY_MODEL_VALUES = cloud_legacy_values_for(GTC_VISION_MODEL_CHOICES)


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
            deprecated_values=LEGACY_MODEL_VALUES,
        )

        self._provider = ProviderSelectionComponent(
            node=self,
            model_provider_param=model_provider_param,
            model_access=self._model_access,
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
        if not self._provider.uses_griptape_cloud_driver():
            return None
        api_key = resolve_cloud_api_key()
        if not api_key:
            msg = missing_credential_message("describe an image")
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
            self._provider.hide()

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
            self._provider.show()
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
        self._model_access.on_value_set(parameter, value)
        if parameter.name == "model_provider":
            self._provider.on_provider_changed(str(value))

    def _parse_output_schema(self) -> dict | None:
        schema_value = self.get_parameter_value("output_schema")
        if isinstance(schema_value, str):
            if not schema_value.strip():
                return None
            try:
                schema_value = json.loads(schema_value)
            except json.JSONDecodeError as e:
                msg = (
                    f"DescribeImage '{self.name}': Unable to parse output_schema as JSON: {e}. "
                    "Try using the `Create Agent Schema` node to generate a schema."
                )
                raise ValueError(msg) from e
        if schema_value is not None and not isinstance(schema_value, dict):
            msg = (
                f"DescribeImage '{self.name}': output_schema must be a JSON schema object (dict) "
                f"or a JSON string, got: {type(schema_value).__name__}"
            )
            raise TypeError(msg)
        return schema_value

    def _third_party_model_config(self, provider_name: str, model: str) -> ModelConfig:
        providers: list[ProviderConfig] = self._provider._fetch_providers()
        provider_config = next((p for p in providers if p.name == provider_name), None)
        if provider_config is None:
            msg = f"DescribeImage '{self.name}': provider '{provider_name}' not found in configured providers."
            raise ValueError(msg)
        return model_config_for_engine_provider(provider_config, model)

    def _resolve_model_config(self, state: AgentState | None) -> ModelConfig:
        """The model that will run: a connected Agent's, a connected Prompt Model Config, or the dropdown selection."""
        if state is not None and state.model is not None:
            return state.model
        model_input = self.get_parameter_value("model")
        connected = model_config_from_input(model_input)
        if connected is not None:
            return connected
        model_name = model_input or DEFAULT_MODEL
        provider_name = self.get_parameter_value("model_provider") or "griptape_cloud"
        if provider_name != "griptape_cloud":
            return self._third_party_model_config(provider_name, model_name)
        if model_name not in self._model_access.model_choices:
            model_name = DEFAULT_MODEL
        return ModelConfig(provider=ModelProvider.GRIPTAPE_CLOUD, model=model_name)

    def _collect_image_contents(self) -> list:
        # Flatten nested lists: a ParameterList child may receive a list of artifacts
        # when connected to an output that produces List[ImageUrlArtifact].
        flat_images: list = []
        for img in self.get_parameter_value("images") or []:
            if isinstance(img, list):
                flat_images.extend(img)
            else:
                flat_images.append(img)
        return [
            image_content(img)
            for img in flat_images
            if img is not None and not (isinstance(img, str) and not img.strip())
        ]

    def process(self) -> AsyncResult[AgentRunResult[Any]]:
        agent_value = self.get_parameter_value("agent")
        provider_name = self.get_parameter_value("model_provider") or "griptape_cloud"

        # License-policy runtime gate, scoped to Griptape Cloud models (the only ones the
        # catalog declares) and skipped when an Agent is connected: it supplies its own
        # model, so the node's (hidden, not cleared) dropdown value is stale. The
        # INVOKE_MODEL declaration below gates the model that actually runs.
        if agent_value is None and provider_name == "griptape_cloud":
            self._model_access.raise_if_selection_denied()

        output_schema = self._parse_output_schema()

        prompt = self.get_parameter_value("prompt") or "Describe the image"
        if self.get_parameter_value("description_only"):
            prompt += "\n\nOutput image description only."

        image_contents = self._collect_image_contents()
        if not image_contents:
            self.parameter_output_values["output"] = "No image provided"
            return

        state = connected_agent_state(agent_value)
        model_config = self._resolve_model_config(state)
        agent_state = replace(state, model=model_config) if state is not None else AgentState(model=model_config)
        try:
            output_type = output_type_from_schema(output_schema) if output_schema else str
        except Exception as e:
            logger.error(
                "DescribeImage '%s': Unable to create output schema model: %s. "
                "Try using the `Create Agent Schema` node to generate a schema.",
                self.name,
                e,
            )
            raise

        # Declare the model that will actually run, read from the resolved config rather than
        # the node's `model` parameter, which keeps its last dropdown value (hidden, not cleared)
        # while a connected Agent supplies the real model. The util resolves the provider model
        # id to its stable catalog key before declaring. Declare before the network call below
        # so a denied invocation fails closed rather than reaching the provider.
        require_model_invocation_sync(self, model_config.model)

        # Build in the worker thread: building a Cloud model makes a blocking engine round trip.
        result = yield lambda: run_agent(
            build_agent_from_state(agent_state, output_type=output_type),
            [prompt, *image_contents],
            message_history=agent_state.messages,
        )
        output = result.output
        output_text = output_to_text(output)
        self.parameter_output_values["output"] = output if output_schema else output_text

        # Store the run as text: image bytes in conversation history would bloat the saved
        # workflow and be resent on every downstream API call.
        run_messages = messages_from_runs([{"input": prompt, "output": output_text}])
        self.parameter_output_values["agent"] = replace(
            agent_state, messages=[*agent_state.messages, *run_messages]
        ).to_wire()
