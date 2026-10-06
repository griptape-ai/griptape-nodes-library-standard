from typing import Any

from griptape.artifacts import ImageUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterGroup, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, BaseNode, ControlNode
from griptape_nodes.exe_types.param_components.model_access_component import ModelAccessComponent
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_types.parameter_bool import ParameterBool
from griptape_nodes.exe_types.param_types.parameter_image import ParameterImage
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.traits.options import Options
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

from griptape_nodes_library.llm.agent_state import AgentState
from griptape_nodes_library.llm.image_generation import ImageGenerationConfig, ImageProvider, generate_image
from griptape_nodes_library.llm.model_config import ModelConfig, ModelProvider
from griptape_nodes_library.llm.runner import prompt_model
from griptape_nodes_library.utils.cloud_credential_utils import (
    missing_credential_message,
    resolve_cloud_api_key,
)
from griptape_nodes_library.utils.model_invocation import require_model_invocation_sync

API_KEY_ENV_VAR = "GT_CLOUD_API_KEY"
SERVICE = "Griptape"
MODEL_CHOICES = [
    "gpt-image-1-mini",
    "gpt-image-1.5",
]
AVAILABLE_SIZES = ["1024x1024", "1536x1024", "1024x1536"]
DEFAULT_MODEL = MODEL_CHOICES[0]
DEFAULT_SIZE = AVAILABLE_SIZES[0]
ENHANCEMENT_MODEL = "gpt-4o"
ENHANCEMENT_INSTRUCTIONS = """
Enhance the following prompt for an image generation engine. Return only the image generation prompt.
Include unique details that make the subject stand out.
Specify a specific depth of field, and time of day.
Use dust in the air to create a sense of depth.
Use a slight vignetting on the edges of the image.
Use a color palette that is complementary to the subject.
Focus on qualities that will make this the most professional looking photo in the world.
IMPORTANT: Output must be a single, raw prompt string for an image generation model. Do not include any preamble, explanation, or conversational language."""
GENERATED_IMAGE_MEMORY = 'I created an image based on your prompt.\n<THOUGHT>\nmeta={"used_tool": True, "tool": "GenerateImageTool"}\n</THOUGHT>'

# GPT-4o is the separate prompt-enhancement model, not an image-model choice.
LEGACY_MODEL_VALUES = {
    "GPT Image 1 Mini": "gpt-image-1-mini",
    "GPT Image 1.5": "gpt-image-1.5",
    "dall-e-3": "gpt-image-1-mini",
    "gpt-image-1": "gpt-image-1-mini",
    "gtc_gpt_image_1_5": "gpt-image-1.5",
    "gtc_gpt_image_1_mini": "gpt-image-1-mini",
}


class GenerateImage(ControlNode):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # TODO: https://github.com/griptape-ai/griptape-nodes/issues/720
        self._has_connection_to_prompt = False

        self.add_parameter(
            Parameter(
                name="agent",
                type="Agent",
                input_types=["Agent"],
                output_type="Agent",
                tooltip="None",
                default_value=None,
                allowed_modes={ParameterMode.INPUT, ParameterMode.OUTPUT},
            )
        )
        model_param = Parameter(
            name="model",
            input_types=["str", "Image Generation Driver"],
            type="str",
            default_value=DEFAULT_MODEL,
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            tooltip="Select the model you want to use from the available options.",
            ui_options={"display_name": "image model"},
        )
        self.add_parameter(model_param)
        # License-policy helper: adds Options + refresh Button traits, applies per-row
        # decoration + badge, exposes query_for_denial / raise_if_denied, and
        # relocates the stored value to a permitted alternative if DEFAULT_MODEL is denied.
        self._model_access = ModelAccessComponent(
            node=self,
            parameter=model_param,
            model_choices=MODEL_CHOICES,
            default_model=DEFAULT_MODEL,
            deprecated_values=LEGACY_MODEL_VALUES,
        )
        self.add_parameter(
            ParameterString(
                name="prompt",
                tooltip="None",
                default_value="",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                multiline=True,
                placeholder_text="Enter your image generation prompt here.",
            )
        )
        self.add_parameter(
            ParameterString(
                name="image_size",
                default_value=DEFAULT_SIZE,
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                tooltip="Select the size of the generated image.",
                traits={Options(choices=AVAILABLE_SIZES)},
            )
        )

        self.add_parameter(
            ParameterBool(
                name="enhance_prompt",
                tooltip="None",
                default_value=False,
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            ParameterImage(
                name="output",
                tooltip="None",
                default_value=None,
                allowed_modes={ParameterMode.PROPERTY, ParameterMode.OUTPUT},
                ui_options={"pulse_on_run": True},
                settable=False,  # Ensures this serializes on save, but don't let user set it.
            )
        )

        self._output_file = ProjectFileParameter(
            node=self,
            name="output_file",
            default_filename="generated.png",
        )
        self._output_file.add_parameter()

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

    def validate_before_workflow_run(self) -> list[Exception] | None:
        # TODO: https://github.com/griptape-ai/griptape-nodes/issues/871
        exceptions = []
        api_key = resolve_cloud_api_key()
        if not api_key:
            # If we have an agent or a driver, the lack of API key will be surfaced on them, not us.
            agent_val = self.parameter_values.get("agent", None)
            driver_val = self.parameter_values.get("driver", None)
            if agent_val is None and driver_val is None:
                msg = missing_credential_message("generate an image")
                exceptions.append(KeyError(msg))

        prompt_error = self.validate_empty_parameter(param="prompt")
        if prompt_error and not self._has_connection_to_prompt:
            exceptions.append(prompt_error)

        return exceptions if exceptions else None

    def after_value_set(
        self,
        parameter: Parameter,
        value: Any,
    ) -> None:
        """Certain options are only available for certain models."""
        if parameter.name == "output_format":
            if value == "jpeg":
                self.show_parameter_by_name("output_compression")
            else:
                self.hide_parameter_by_name("output_compression")

        if parameter.name == "model":
            # "model" supports either a string OR an Image Generation Driver. Only strings serialize.
            if isinstance(value, str):
                # Strings can serialize.
                parameter.serializable = True
            else:
                # It's an Image Generation Driver, which we canNOT serialize.
                parameter.serializable = False
            self._model_access.on_value_set(parameter, value)

        return super().after_value_set(parameter, value)

    def process(self) -> AsyncResult:
        params = self.parameter_values

        orig_prompt = self.get_parameter_value("prompt")

        exception = self.validate_empty_parameter(param="prompt")
        if exception:
            raise exception

        # License-policy runtime gate for the image model. Non-string values (a connected
        # Image Generation Driver) bypass it: they carry their own model identity. The
        # INVOKE_MODEL declarations below still gate the models that actually run.
        self._model_access.raise_if_selection_denied()

        agent_input = self.get_parameter_value("agent")
        state = AgentState.from_wire(agent_input)
        if state.model is None:
            state.model = ModelConfig(provider=ModelProvider.GRIPTAPE_CLOUD, model=ENHANCEMENT_MODEL)

        prompt = self._build_context(state, orig_prompt)

        enhance_prompt = params.get("enhance_prompt", False)

        if enhance_prompt:
            self.append_value_to_parameter("logs", "Enhancing prompt...\n")
            # This runs the agent's own model (the default gpt-4o, or a connected agent's) -- a
            # model invocation distinct from the image-generation model below, and one no
            # dropdown selects. Declare it so a denied invocation fails closed before the call.
            enhance_model = state.model
            require_model_invocation_sync(self, enhance_model.model, purpose="prompt enhancement")
            # The model call blocks, so hand it to the engine to run in the background
            # via `yield lambda` and resume when it's done.
            prompt = yield lambda: prompt_model(
                enhance_model, [ENHANCEMENT_INSTRUCTIONS, prompt], rulesets=state.rulesets
            )
            self.append_value_to_parameter("logs", "Finished enhancing prompt...\n")
        else:
            self.append_value_to_parameter("logs", "Prompt enhancement disabled.\n")

        model_input = self.get_parameter_value("model")
        image_config = ImageGenerationConfig.from_wire(model_input)
        if image_config is None:
            if not isinstance(model_input, str) or model_input not in self._model_access.model_choices:
                model_input = DEFAULT_MODEL
            image_config = ImageGenerationConfig(
                provider=ImageProvider.GRIPTAPE_CLOUD,
                model=model_input,
                image_size=self.get_parameter_value("image_size"),
            )

        # The image model is settled above. The util resolves its provider model id to the
        # stable catalog key (via the node's model_usage) before declaring. Declare before the
        # network call below so a denied invocation fails closed here.
        require_model_invocation_sync(self, image_config.model)

        self.append_value_to_parameter("logs", "Starting processing image..\n")
        yield lambda: self._create_image(image_config, prompt)
        self.append_value_to_parameter("logs", "Finished processing image.\n")

        # Record the exchange as a short text memory: the image itself is huge, so the agent
        # is told it used a tool instead.
        state.messages = [
            *state.messages,
            ModelRequest(parts=[UserPromptPart(content=orig_prompt)]),
            ModelResponse(parts=[TextPart(content=GENERATED_IMAGE_MEMORY)]),
        ]
        self.parameter_output_values["agent"] = state.to_wire()

    @staticmethod
    def _build_context(state: AgentState, prompt: str) -> str:
        """Prefix `prompt` with the agent's conversation, since the image model has no memory."""
        context = ""
        runs = state.runs()
        if runs:
            lines = []
            for run in runs:
                if run["input"]:
                    lines.append(f"User: {run['input']}")
                if run["output"]:
                    lines.append(f"Assistant: {run['output']}")
            context = f"<Conversation History>\n{chr(10).join(lines)}</Conversation History>\n"
        if prompt:
            context = f"{context}\nUser:\n{prompt}\n"
        return context

    def after_incoming_connection(
        self,
        source_node: BaseNode,
        source_parameter: Parameter,
        target_parameter: Parameter,
    ) -> None:
        """Callback after a Connection has been established TO this Node."""
        # Record a connection to the prompt Parameter so that node validation doesn't get aggro
        if target_parameter.name == "prompt":
            self._has_connection_to_prompt = True
            # hey.. what if we just remove the property mode from the prompt parameter?
            if ParameterMode.PROPERTY in target_parameter.allowed_modes:
                target_parameter.allowed_modes = target_parameter.allowed_modes - {ParameterMode.PROPERTY}

        if target_parameter.name == "model" and source_parameter.name == "image_model_config":
            # Check and see if the incoming connection is from a image model config.
            target_parameter.type = source_parameter.type
            target_parameter.remove_trait(trait_type=target_parameter.find_elements_by_type(Options)[0])
            ui_options = target_parameter.ui_options
            ui_options["display_name"] = source_parameter.name
            target_parameter.ui_options = ui_options
            target_parameter.allowed_modes = {ParameterMode.INPUT}

            self.hide_parameter_by_name("image_size")

        return super().after_incoming_connection(source_node, source_parameter, target_parameter)

    def after_incoming_connection_removed(
        self,
        source_node: BaseNode,
        source_parameter: Parameter,
        target_parameter: Parameter,
    ) -> None:
        """Callback after a Connection TO this Node was REMOVED."""
        # Remove the state maintenance of the connection to the prompt Parameter
        if target_parameter.name == "prompt":
            self._has_connection_to_prompt = False
            # If we have no connections to the prompt parameter, add the property mode back
            target_parameter.allowed_modes = target_parameter.allowed_modes | {ParameterMode.PROPERTY}

        # Check and see if the incoming connection is from an agent. If so, we'll hide the model parameter
        if target_parameter.name == "model":
            target_parameter.type = "str"
            # Enable PROPERTY so the user can set it
            target_parameter.allowed_modes = {ParameterMode.INPUT, ParameterMode.PROPERTY}

            default_model = self._model_access.pick_permitted_default() or DEFAULT_MODEL
            target_parameter.set_default_value(default_model)
            target_parameter.default_value = default_model
            ui_options = target_parameter.ui_options
            ui_options["display_name"] = "model"
            target_parameter.ui_options = ui_options
            self.set_parameter_value("model", default_model)
            # Helper reinstalls its Options trait + decoration + badge on the freshly-uncovered
            # parameter (the incoming-connection handler stripped Options when the driver connected).
            self._model_access.reinstall_options()
            self.show_parameter_by_name("image_size")

        return super().after_incoming_connection_removed(source_node, source_parameter, target_parameter)

    def _create_image(self, config: ImageGenerationConfig, prompt: str) -> None:
        image_bytes = generate_image(config, prompt)
        dest = self._output_file.build_file()
        saved = dest.write_bytes(image_bytes)
        url_artifact = ImageUrlArtifact(value=saved.location)
        self.publish_update_to_parameter("output", url_artifact)
