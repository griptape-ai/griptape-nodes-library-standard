"""Defines the CoherePrompt node for configuring a Cohere prompt model.

This module provides the `CoherePrompt` class, which allows users
to configure and utilize the Cohere prompt service within the Griptape
Nodes framework. It inherits common prompt parameters from `BasePrompt`, sets
Cohere specific model options, requires a Cohere API key via
node configuration, and emits a `ModelConfig`.
"""

from griptape_nodes_library.config.prompt.base_prompt import BasePrompt
from griptape_nodes_library.llm.model_config import ModelProvider

# --- Constants ---

SERVICE = "Cohere"
API_KEY_URL = "https://dashboard.cohere.com/api-keys"
API_KEY_ENV_VAR = "COHERE_API_KEY"
MODEL_CHOICES = ["command-r-plus"]
DEFAULT_MODEL = MODEL_CHOICES[0]

# Migrates values saved before the dropdown stored the provider's own model id.
LEGACY_MODEL_VALUES = {
    "Command R+": "command-r-plus",
    "cohere_command_r_plus": "command-r-plus",
}


class CoherePrompt(BasePrompt):
    """Node for configuring a Cohere prompt model.

    Inherits from `BasePrompt` to leverage common LLM parameters. This node
    customizes the available models to those supported by Cohere, requires a
    Cohere API key set in the node's configuration under the 'Cohere' service, and potentially handles parameter conversions specific to the
    Cohere (like min_p to top_p).

    The `process` method turns the configured parameters into a `ModelConfig` that
    names the API key secret, and assigns it to the 'prompt_model_config' output parameter.
    """

    def __init__(self, **kwargs) -> None:
        """Initializes the CoherePrompt node.

        Calls the superclass initializer, then modifies the inherited 'model'
        parameter to use Anthropic specific models and sets a default.
        """
        super().__init__(**kwargs)

        # --- Customize Inherited Parameters ---

        # Offer Cohere's models as a license-filtered dropdown.
        self._install_model_access(
            model_choices=MODEL_CHOICES, default_model=DEFAULT_MODEL, deprecated_values=LEGACY_MODEL_VALUES
        )

        # Replace `min_p` with `top_p` for Cohere.
        self._replace_param_by_name(
            param_name="min_p", new_param_name="p", tooltip=None, default_value=0.9, ui_options=None
        )
        self._replace_param_by_name(param_name="top_k", new_param_name="k")

        # Remove the 'seed' parameter as it's not directly used by Cohere.
        self.remove_parameter_element_by_name("seed")
        self.remove_parameter_element_by_name("response_format")

    def process(self) -> None:
        """Emits the `ModelConfig` for the selected Cohere model.

        Cohere's `p` and `k` parameters map to the standard `top_p` and `top_k` settings.
        Fails closed if the license denies the selected model. The API key is
        referenced by secret name only.
        """
        self._raise_if_model_denied()

        settings = self._common_settings()
        for param_name, setting_name in (("p", "top_p"), ("k", "top_k")):
            value = self.get_parameter_value(param_name)
            if value is not None:
                settings[setting_name] = value

        config = self._build_model_config(
            ModelProvider.COHERE,
            self._get_selected_model_id(),
            settings=settings,
            api_key_secret=API_KEY_ENV_VAR,
        )
        self.parameter_output_values["prompt_model_config"] = config

    def validate_before_workflow_run(self) -> list[Exception] | None:
        """Validates that the Cohere API key is configured correctly.

        Calls the base class helper `_validate_api_key` with Cohere-specific
        configuration details.
        """
        return self._validate_api_key(
            service_name=SERVICE,
            api_key_env_var=API_KEY_ENV_VAR,
            api_key_url=API_KEY_URL,
        )
