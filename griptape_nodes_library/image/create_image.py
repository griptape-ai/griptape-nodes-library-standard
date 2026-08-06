from typing import Any, cast

import requests
from griptape.artifacts import BaseArtifact, ImageUrlArtifact
from griptape.drivers.image_generation.base_image_generation_driver import BaseImageGenerationDriver
from griptape.drivers.image_generation.griptape_cloud import GriptapeCloudImageGenerationDriver
from griptape.drivers.prompt.griptape_cloud import GriptapeCloudPromptDriver
from griptape.tasks import PromptImageGenerationTask, PromptTask
from griptape_nodes.exe_types.core_types import Parameter, ParameterGroup, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, BaseNode, ControlNode
from griptape_nodes.exe_types.param_components.model_access_component import ModelAccessComponent
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_types.parameter_bool import ParameterBool
from griptape_nodes.exe_types.param_types.parameter_image import ParameterImage
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.options import Options

from griptape_nodes_library.agents.griptape_nodes_agent import GriptapeNodesAgent as GtAgent
from griptape_nodes_library.utils.agent_utils import restore_provider_driver, unwrap_agent, wrap_agent
from griptape_nodes_library.utils.error_utils import try_throw_error
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

# Migrates values saved before the dropdown stored the provider's own model id. "dall-e-3"
# and "gpt-image-1" predate this node's own MODEL_CHOICES history and were folded in
# from the DEPRECATED_MODELS dict this replaces. "GPT-4o" / "gpt-4o" are deliberately
# excluded even though the generated catalog table lists them: this node's dropdown
# never offered "gpt-4o" as an image model (it's the hardcoded model of the separate
# prompt-enhancement driver below), and its catalog key gtc_gpt_4o is not one of this
# node's own MODEL_CHOICES, so mapping to it here would fail ModelAccessComponent's
# construction-time validation that every deprecated_values target is a current choice.
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

    def validate_before_workflow_run(self) -> list[Exception] | None:
        # TODO: https://github.com/griptape-ai/griptape-nodes/issues/871
        exceptions = []
        api_key = GriptapeNodes.SecretsManager().get_secret(API_KEY_ENV_VAR)
        if not api_key:
            # If we have an agent or a driver, the lack of API key will be surfaced on them, not us.
            agent_val = self.parameter_values.get("agent", None)
            driver_val = self.parameter_values.get("driver", None)
            if agent_val is None and driver_val is None:
                msg = f"{API_KEY_ENV_VAR} is not defined"
                exceptions.append(KeyError(msg))

        # Validate that we have a prompt.
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
            # "model" supports either a string OR an Image Generation Driver. We can serialize strings, but not driver objects.
            if isinstance(value, str):
                # Strings can serialize.
                parameter.serializable = True
            else:
                # It's an Image Generation Driver, which we canNOT serialize.
                parameter.serializable = False
            self._model_access.on_value_set(parameter, value)

        return super().after_value_set(parameter, value)

    def process(self) -> AsyncResult:
        # Get the parameters from the node
        params = self.parameter_values

        # Validate that we have a prompt.
        orig_prompt = self.get_parameter_value("prompt")

        exception = self.validate_empty_parameter(param="prompt")
        if exception:
            raise exception

        # License-policy runtime gate for the image model. Non-string values (a connected
        # Image Generation Driver) bypass it: they carry their own model identity. The
        # INVOKE_MODEL declarations below still gate the models that actually run.
        self._model_access.raise_if_selection_denied()

        agent_input = self.get_parameter_value("agent")
        tool_configs: list = []
        ruleset_configs: list = []
        if not agent_input:
            prompt_driver = GriptapeCloudPromptDriver(
                model="gpt-4o",
                api_key=GriptapeNodes.SecretsManager().get_secret(API_KEY_ENV_VAR),
                stream=True,
            )
            agent = GtAgent(prompt_driver=prompt_driver)
        else:
            agent_core_dict, tool_configs, ruleset_configs = unwrap_agent(agent_input)
            agent = GtAgent.from_dict(agent_core_dict)
            restore_provider_driver(agent, agent_input)

        # Add some context to the prompt based on the agent's conversation memory.
        # We use this because otherwise the agent will not have the context of the prompt.
        # This is due to the fact that when you temporarily swap the task from a prompt_task to an image generation task,
        # the context is lost.
        prompt = agent.build_context(prompt=orig_prompt)

        # Check if we have a connection to the prompt parameter
        enhance_prompt = params.get("enhance_prompt", False)

        if enhance_prompt:
            self.append_value_to_parameter("logs", "Enhancing prompt...\n")
            # This runs the agent's own prompt driver (the default gpt-4o driver, or a
            # connected agent's) -- a model invocation distinct from the image-generation
            # driver below, and one no dropdown selects, so its model comes from the task
            # driver. Declare it so a denied invocation fails closed before the call.
            enhance_model = cast(PromptTask, agent.tasks[0]).prompt_driver.model
            require_model_invocation_sync(self, enhance_model, purpose="prompt enhancement")
            # agent.run is a blocking operation that will hold up the rest of the engine.
            # By using `yield lambda`, the engine can run this in the background and resume when it's done.
            result = yield lambda: agent.run(
                [
                    """
Enhance the following prompt for an image generation engine. Return only the image generation prompt.
Include unique details that make the subject stand out.
Specify a specific depth of field, and time of day.
Use dust in the air to create a sense of depth.
Use a slight vignetting on the edges of the image.
Use a color palette that is complementary to the subject.
Focus on qualities that will make this the most professional looking photo in the world.
IMPORTANT: Output must be a single, raw prompt string for an image generation model. Do not include any preamble, explanation, or conversational language.""",
                    prompt,
                ]
            )
            self.append_value_to_parameter("logs", "Finished enhancing prompt...\n")
            prompt = result.output
        else:
            self.append_value_to_parameter("logs", "Prompt enhancement disabled.\n")
        # Initialize driver kwargs with required parameters
        kwargs = {}

        # Driver
        model_input = self.get_parameter_value("model")
        driver = None
        if isinstance(model_input, BaseImageGenerationDriver):
            driver = model_input
        elif isinstance(model_input, str):
            if model_input not in self._model_access.model_choices:
                model_input = DEFAULT_MODEL
            driver = GriptapeCloudImageGenerationDriver(
                model=model_input,
                image_size=self.get_parameter_value("image_size"),
                api_key=GriptapeNodes.SecretsManager().get_secret(API_KEY_ENV_VAR),
                # Don't retry on HTTP errors, we want to fail fast.
                ignored_exception_types=(requests.exceptions.HTTPError,),
            )
        else:
            driver = GriptapeCloudImageGenerationDriver(
                model=DEFAULT_MODEL,
                image_size=self.get_parameter_value("image_size"),
                api_key=GriptapeNodes.SecretsManager().get_secret(API_KEY_ENV_VAR),
                ignored_exception_types=(requests.HTTPError,),
            )

        kwargs["image_generation_driver"] = driver

        # The image generation driver is settled above -- every branch produces a
        # concrete BaseImageGenerationDriver whose `model` is a required field. The util
        # resolves that provider model id to its stable catalog key (via the node's
        # model_usage) before declaring. Declare before swapping in the task (and the
        # network call it triggers below) so a denied invocation fails closed here.
        require_model_invocation_sync(self, driver.model)

        # Set new Image Generation Task
        # Cool trick to swap the task of the agent from PromptTask to ImageGenerationTask
        agent.swap_task(PromptImageGenerationTask(**kwargs))

        # Run the agent asynchronously
        self.append_value_to_parameter("logs", "Starting processing image..\n")
        yield lambda: self._create_image(agent, prompt)
        self.append_value_to_parameter("logs", "Finished processing image.\n")

        # Create a false memory for the agent
        # This is because the agent will have the base64 image in its memory, which is huge.
        # So we replace it with a simple, false memory - but tell it is used a tool.
        agent.insert_false_memory(
            prompt=orig_prompt, output="I created an image based on your prompt.", tool="GenerateImageTool"
        )

        # Restore the task
        # Now restore the original prompt task for the agent.
        agent.restore_task()

        # Output the agent
        if agent.tasks:
            cast(PromptTask, agent.tasks[0]).tools = []
        provider = agent_input.get("provider") if isinstance(agent_input, dict) else None
        self.parameter_output_values["agent"] = wrap_agent(
            agent.to_dict(), tool_configs, ruleset_configs, provider=provider
        )

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

    def _create_image(self, agent: GtAgent, prompt: BaseArtifact | str) -> None:
        agent.run(prompt)
        dest = self._output_file.build_file()
        saved = dest.write_bytes(agent.output.to_bytes())
        url_artifact = ImageUrlArtifact(value=saved.location)
        self.publish_update_to_parameter("output", url_artifact)
        try_throw_error(agent.output)
