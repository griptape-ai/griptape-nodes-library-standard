"""Defines the GrokPrompt node for configuring a Grok prompt model.

This module provides the `GrokPrompt` class, which allows users
to configure and utilize the Grok prompt service within the Griptape
Nodes framework. It inherits common prompt parameters from `BasePrompt`, sets
Grok specific model options, requires a Grok API key via
node configuration, and emits a `ModelConfig`.
"""

from griptape_nodes_library.config.prompt.base_prompt import BasePrompt
from griptape_nodes_library.llm.model_config import ModelProvider

# --- Constants ---

SERVICE = "Grok"
API_KEY_URL = "https://console.x.ai"
API_KEY_ENV_VAR = "GROK_API_KEY"
MODEL_CHOICES = [
    "grok-3-beta",
    "grok-3-fast-beta",
    "grok-3-mini-beta",
    "grok-3-mini-fast-beta",
    "grok-2-vision-1212",
]
DEFAULT_MODEL = MODEL_CHOICES[0]

# Migrates values saved before the dropdown stored the provider's own model id.
LEGACY_MODEL_VALUES = {
    "Grok 2 Vision": "grok-2-vision-1212",
    "Grok 3 Beta": "grok-3-beta",
    "Grok 3 Fast Beta": "grok-3-fast-beta",
    "Grok 3 Mini Beta": "grok-3-mini-beta",
    "Grok 3 Mini Fast Beta": "grok-3-mini-fast-beta",
    "xai_grok_2_vision_1212": "grok-2-vision-1212",
    "xai_grok_3_beta": "grok-3-beta",
    "xai_grok_3_fast_beta": "grok-3-fast-beta",
    "xai_grok_3_mini_beta": "grok-3-mini-beta",
    "xai_grok_3_mini_fast_beta": "grok-3-mini-fast-beta",
}


class GrokPrompt(BasePrompt):
    """Node for configuring a Grok prompt model.

    Inherits from `BasePrompt` to leverage common LLM parameters. This node
    customizes the available models to those supported by Grok,
    removes parameters not applicable to Grok (like 'seed'), and
    requires a Grok API key to be set in the node's configuration
    under the 'Grok' service.

    The `process` method turns the configured parameters into a `ModelConfig` that
    names the API key secret, and assigns it to the 'prompt_model_config' output parameter.
    """

    def __init__(self, **kwargs) -> None:
        """Initializes the GrokPrompt node.

        Calls the superclass initializer, then modifies the inherited 'model'
        parameter to use Grok specific models and sets a default.
        It also removes the 'seed' parameter inherited from `BasePrompt` as it's
        not directly supported by the Grok.
        """
        super().__init__(**kwargs)

        # --- Customize Inherited Parameters ---

        # Offer Grok's models as a license-filtered dropdown.
        self._install_model_access(
            model_choices=MODEL_CHOICES, default_model=DEFAULT_MODEL, deprecated_values=LEGACY_MODEL_VALUES
        )

        # Remove `top_k` parameter as it's not used by Grok.
        self.remove_parameter_element_by_name("seed")
        self.remove_parameter_element_by_name("top_k")

        # Replace `min_p` with `top_p` for Grok.
        self._replace_param_by_name(param_name="min_p", new_param_name="top_p", default_value=0.9)

    def process(self) -> None:
        """Emits the `ModelConfig` for the selected Grok model.

        Fails closed if the license denies the selected model. The API key is
        referenced by secret name only.
        """
        self._raise_if_model_denied()

        config = self._build_model_config(
            ModelProvider.GROK,
            self._get_selected_model_id(),
            api_key_secret=API_KEY_ENV_VAR,
        )
        self.parameter_output_values["prompt_model_config"] = config

    def validate_before_workflow_run(self) -> list[Exception] | None:
        """Validates that the Grok API key is configured correctly.

        Calls the base class helper `_validate_api_key` with Grok-specific
        configuration details.
        """
        return self._validate_api_key(
            service_name=SERVICE,
            api_key_env_var=API_KEY_ENV_VAR,
            api_key_url=API_KEY_URL,
        )
