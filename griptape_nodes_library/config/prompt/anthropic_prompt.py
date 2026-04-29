"""Defines the AnthropicPrompt node for configuring the Anthropic Prompt Driver.

This module provides the `AnthropicPrompt` class, which allows users
to configure and utilize the Anthropic prompt service within the Griptape
Nodes framework. It inherits common prompt parameters from `BasePrompt`, sets
Anthropic specific model options, requires an Anthropic API key via
node configuration, and instantiates the `GtAnthropicPromptDriver`.
"""

from typing import Any

from griptape.drivers.prompt.anthropic import AnthropicPromptDriver as GtAnthropicPromptDriver
from griptape_nodes.exe_types.core_types import Parameter, ParameterMessage
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.button import Button

from griptape_nodes_library.config.prompt.base_prompt import BasePrompt

# --- Constants ---

SERVICE = "Anthropic"
API_KEY_URL = "https://console.anthropic.com/settings/keys"
API_KEY_ENV_VAR = "ANTHROPIC_API_KEY"
MODEL_CHOICES = [
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
]
DEFAULT_MODEL = MODEL_CHOICES[2]  # claude-sonnet-4-6

# Models that do not support sampling parameters (temperature, top_p, top_k)
MODELS_WITHOUT_SAMPLING_PARAMS = {"claude-opus-4-7"}

# Deprecated models and their replacements
DEPRECATED_MODELS = {
    # Dated model versions
    "claude-3-5-sonnet-20241022": "claude-sonnet-4-6",
    "claude-3-5-sonnet-20240620": "claude-sonnet-4-6",
    "claude-3-5-haiku-20241022": "claude-haiku-4-5-20251001",
    "claude-3-opus-20240229": "claude-opus-4-6",
    "claude-3-sonnet-20240229": "claude-sonnet-4-6",
    "claude-3-haiku-20240307": "claude-haiku-4-5-20251001",
    # -latest variants
    "claude-3-7-sonnet-latest": "claude-sonnet-4-6",
    "claude-3-5-sonnet-latest": "claude-sonnet-4-6",
    "claude-3-5-opus-latest": "claude-opus-4-6",
    "claude-3-5-haiku-latest": "claude-haiku-4-5-20251001",
}


class AnthropicPrompt(BasePrompt):
    """Node for configuring and providing an Anthropic Prompt Driver.

    Inherits from `BasePrompt` to leverage common LLM parameters. This node
    customizes the available models to those supported by Anthropic, requires an
    Anthropic API key set in the node's configuration under the 'Anthropic'
    service, and potentially handles parameter conversions specific to the
    Anthropic driver (like min_p to top_p).

    The `process` method uses the `_get_common_driver_args` helper, adds
    Anthropic-specific configurations, and instantiates the
    `AnthropicPromptDriver`.
    """

    def __init__(self, **kwargs) -> None:
        """Initializes the AnthropicPrompt node.

        Calls the superclass initializer, then modifies the inherited 'model'
        parameter to use Anthropic specific models and sets a default.
        """
        super().__init__(**kwargs)

        # --- Customize Inherited Parameters ---

        # Update the 'model' parameter for Anthropic specifics.
        self._update_option_choices(param="model", choices=MODEL_CHOICES, default=DEFAULT_MODEL)

        # Add deprecation notice message element right after model parameter
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

        # Replace `min_p` with `top_p` for Anthropic.
        self._replace_param_by_name(
            param_name="min_p", new_param_name="top_p", tooltip=None, default_value=0.9, ui_options=None
        )

        # Remove the 'seed' parameter as it's not directly used by AnthropicPromptDriver.
        self.remove_parameter_element_by_name("seed")

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        if parameter.name == "model" and isinstance(value, str):
            if value in MODELS_WITHOUT_SAMPLING_PARAMS:
                self.hide_parameter_by_name("top_p")
                self.hide_parameter_by_name("temperature")
                self.hide_parameter_by_name("top_k")
            else:
                self.show_parameter_by_name("top_p")
                self.show_parameter_by_name("temperature")
                self.show_parameter_by_name("top_k")

        super().after_value_set(parameter, value)

    def before_value_set(
        self,
        parameter: Parameter,
        value: Any,
    ) -> Any:
        if parameter.name == "model":
            if isinstance(value, str) and value in DEPRECATED_MODELS:
                replacement = DEPRECATED_MODELS[value]
                message = self.get_message_by_name_or_element_id("model_deprecation_notice")
                if message is None:
                    raise RuntimeError("model_deprecation_notice message element not found")  # noqa: TRY003, EM101
                message.value = f"The '{value}' model has been deprecated. The model has been updated to '{replacement}'. Please save your workflow to apply this change."
                self.show_message_by_name("model_deprecation_notice")
                value = replacement
            else:
                self.hide_message_by_name("model_deprecation_notice")

        return super().before_value_set(parameter, value)

    def process(self) -> None:
        """Processes the node configuration to create an AnthropicPromptDriver.

        Retrieves parameter values, uses the `_get_common_driver_args` helper
        for common settings, then adds Anthropic-specific arguments like API key
        and model. Handles the conversion of `min_p` to `top_p` if `min_p` is set.
        Finally, instantiates the `AnthropicPromptDriver` and assigns it to the
        'prompt_model_config' output parameter.

        Raises:
            KeyError: If the Anthropic API key is not found in the node configuration.
        """
        # Retrieve all parameter values set on the node.
        params = self.parameter_values

        # --- Get Common Driver Arguments ---
        # Use the helper method from BasePrompt. This gets temperature, stream,
        # max_attempts, max_tokens, use_native_tools, min_p, top_k if they are set.
        common_args = self._get_common_driver_args(params)

        # --- Prepare Anthropic Specific Arguments ---
        specific_args = {}

        # Retrieve the mandatory API key.
        specific_args["api_key"] = GriptapeNodes.SecretsManager().get_secret(API_KEY_ENV_VAR)

        # Get the selected model.
        specific_args["model"] = self.get_parameter_value("model")

        # temperature and top_p are deprecated for some models (e.g. claude-opus-4-7)
        model = self.get_parameter_value("model")
        if model in MODELS_WITHOUT_SAMPLING_PARAMS:
            common_args.pop("temperature", None)
            common_args.pop("top_k", None)
        else:
            top_p = self.get_parameter_value("top_p")
            if top_p is not None:
                specific_args["top_p"] = top_p

        response_format = self.get_parameter_value("response_format")
        if response_format == "json_object":
            response_format = {"type": "json_object"}
            specific_args["response_format"] = response_format

        # --- Combine Arguments and Instantiate Driver ---
        # Combine common arguments (potentially modified) with Anthropic specific arguments.
        all_kwargs = {**common_args, **specific_args}

        # Create the Anthropic prompt driver instance.
        driver = GtAnthropicPromptDriver(**all_kwargs)

        # Set the output parameter 'prompt_model_config'.
        self.parameter_output_values["prompt_model_config"] = driver

    def validate_before_workflow_run(self) -> list[Exception] | None:
        """Validates that the Anthropic API key is configured correctly.

        Calls the base class helper `_validate_api_key` with Anthropic-specific
        configuration details.
        """
        return self._validate_api_key(
            service_name=SERVICE,
            api_key_env_var=API_KEY_ENV_VAR,
            api_key_url=API_KEY_URL,
        )
