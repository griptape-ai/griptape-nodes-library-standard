"""Defines the AnthropicPrompt node for configuring a Anthropic prompt model.

This module provides the `AnthropicPrompt` class, which allows users
to configure and utilize the Anthropic prompt service within the Griptape
Nodes framework. It inherits common prompt parameters from `BasePrompt`, sets
Anthropic specific model options, requires an Anthropic API key via
node configuration, and emits a `ModelConfig`.
"""

from griptape_nodes_library.config.prompt.base_prompt import BasePrompt
from griptape_nodes_library.llm.model_config import ModelProvider

# --- Constants ---

SERVICE = "Anthropic"
API_KEY_URL = "https://console.anthropic.com/settings/keys"
API_KEY_ENV_VAR = "ANTHROPIC_API_KEY"
MODEL_CHOICES = [
    "claude-opus-4-7",
    "claude-sonnet-4-6",
    "claude-haiku-4-5",
]
DEFAULT_MODEL = MODEL_CHOICES[1]  # claude-sonnet-4-6

# Migrates values saved before the dropdown stored the provider's own model id.
LEGACY_MODEL_VALUES = {
    "Claude Haiku 4.5": "claude-haiku-4-5",
    "Claude Opus 4.7": "claude-opus-4-7",
    "Claude Sonnet 4.6": "claude-sonnet-4-6",
    "gtc_claude_haiku_4_5": "claude-haiku-4-5",
    "gtc_claude_opus_4_7": "claude-opus-4-7",
    "gtc_claude_sonnet_4_6": "claude-sonnet-4-6",
    # Dated model versions
    "claude-3-5-sonnet-20241022": "claude-sonnet-4-6",
    "claude-3-5-sonnet-20240620": "claude-sonnet-4-6",
    "claude-3-5-haiku-20241022": "claude-haiku-4-5",
    "claude-3-opus-20240229": "claude-opus-4-7",
    "claude-3-sonnet-20240229": "claude-sonnet-4-6",
    "claude-3-haiku-20240307": "claude-haiku-4-5",
    "claude-3-7-sonnet-20250219": "claude-sonnet-4-6",
    "claude-sonnet-4-20250514": "claude-sonnet-4-6",
    "claude-opus-4-20250514": "claude-opus-4-7",
    # -latest variants
    "claude-3-7-sonnet-latest": "claude-sonnet-4-6",
    "claude-3-5-sonnet-latest": "claude-sonnet-4-6",
    "claude-3-5-opus-latest": "claude-opus-4-7",
    "claude-3-5-haiku-latest": "claude-haiku-4-5",
    # Superseded current-generation models
    "claude-haiku-4-5-20251001": "claude-haiku-4-5",
    "claude-opus-4-6": "claude-opus-4-7",
}


class AnthropicPrompt(BasePrompt):
    """Node for configuring a Anthropic prompt model.

    Inherits from `BasePrompt` to leverage common LLM parameters. This node
    customizes the available models to those supported by Anthropic, requires an
    Anthropic API key set in the node's configuration under the 'Anthropic'
    service, and potentially handles parameter conversions specific to the
    Anthropic (like min_p to top_p).

    The `process` method turns the configured parameters into a `ModelConfig` that
    names the API key secret, and assigns it to the 'prompt_model_config' output parameter.
    """

    def __init__(self, **kwargs) -> None:
        """Initializes the AnthropicPrompt node.

        Calls the superclass initializer, then modifies the inherited 'model'
        parameter to use Anthropic specific models and sets a default.
        """
        super().__init__(**kwargs)

        # --- Customize Inherited Parameters ---

        # Offer Anthropic's models as a license-filtered dropdown.
        self._install_model_access(
            model_choices=MODEL_CHOICES, default_model=DEFAULT_MODEL, deprecated_values=LEGACY_MODEL_VALUES
        )

        # Replace `min_p` with `top_p` for Anthropic.
        self._replace_param_by_name(
            param_name="min_p", new_param_name="top_p", tooltip=None, default_value=0.9, ui_options=None
        )

        # Remove the 'seed' parameter as it's not directly used by Anthropic.
        self.remove_parameter_element_by_name("seed")

    def process(self) -> None:
        """Emits the `ModelConfig` for the selected Anthropic model.

        Fails closed if the license denies the selected model. The API key is
        referenced by secret name only.

        Note: the old driver's `response_format` was read from a parameter this node
        never defined, so it never reached the API and is not carried over.
        """
        self._raise_if_model_denied()

        config = self._build_model_config(
            ModelProvider.ANTHROPIC,
            self._get_selected_model_id(),
            api_key_secret=API_KEY_ENV_VAR,
        )
        self.parameter_output_values["prompt_model_config"] = config

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
