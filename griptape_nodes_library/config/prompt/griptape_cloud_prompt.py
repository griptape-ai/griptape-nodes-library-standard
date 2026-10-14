"""Defines the GriptapeCloudPrompt node for configuring a Griptape Cloud prompt model.

This module provides the `GriptapeCloudPrompt` class, which allows users
to configure and utilize the Griptape Cloud prompt service within the Griptape
Nodes framework. It inherits common prompt parameters from `BasePrompt`, sets
Griptape Cloud specific model options, requires a Griptape Cloud API key via
node configuration, and emits a `ModelConfig`.
"""

from typing import Any

from griptape_nodes.drivers.cloud_models import (
    MODEL_CHOICES,
    MODEL_CHOICES_ARGS,
    O_SERIES_MODELS,
)
from griptape_nodes.exe_types.core_types import Parameter

from griptape_nodes_library.config.prompt.base_prompt import BasePrompt
from griptape_nodes_library.llm.model_config import ModelProvider
from griptape_nodes_library.utils.cloud_credential_utils import (
    missing_credential_message,
    resolve_cloud_api_key,
)
from griptape_nodes_library.utils.cloud_legacy_models import CLOUD_LEGACY_MODEL_VALUES

# --- Constants ---

SERVICE = "Griptape"
BASE_URL = "https://cloud.griptape.ai"
API_KEY_URL = f"{BASE_URL}/configuration/api-keys"
DEFAULT_MODEL = "gpt-4.1-mini"

API_KEY_ENV_VAR = "GT_CLOUD_API_KEY"

# Catalog model args that map onto generation settings; the rest (stream, ...) are griptape-only.
GENERATION_SETTINGS = frozenset({"temperature", "top_p", "top_k", "seed", "max_tokens"})


class GriptapeCloudPrompt(BasePrompt):
    """Node for configuring a Griptape Cloud prompt model.

    Inherits from `BasePrompt` to leverage common LLM parameters. This node
    customizes the available models to those supported by Griptape Cloud,
    removes parameters not applicable to Griptape Cloud (like 'seed'), and
    requires a Griptape Cloud API key to be set in the node's configuration
    under the 'Griptape' service.

    The `process` method turns the configured parameters into a `ModelConfig` that
    names the secrets to use, and assigns it to the 'prompt_model_config' output parameter.
    """

    def __init__(self, **kwargs) -> None:
        """Initializes the GriptapeCloudPrompt node.

        Calls the superclass initializer, then modifies the inherited 'model'
        parameter to use Griptape Cloud specific models and sets a default.
        It also removes the 'seed' parameter inherited from `BasePrompt` as it's
        not directly supported by the Griptape Cloud implementation.
        """
        super().__init__(**kwargs)

        # --- Customize Inherited Parameters ---

        # Offer Griptape Cloud's models as a license-filtered dropdown.
        self._install_model_access(
            model_choices=MODEL_CHOICES, default_model=DEFAULT_MODEL, deprecated_values=CLOUD_LEGACY_MODEL_VALUES
        )

        self.remove_parameter_element_by_name("seed")

        # Remove `top_k` parameter as it's not used by Griptape Cloud.
        self.remove_parameter_element_by_name("top_k")

        # Replace `min_p` with `top_p` for Griptape Cloud.
        self._replace_param_by_name(param_name="min_p", new_param_name="top_p", default_value=0.9)

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        if parameter.name == "model":
            # Branch on the provider's own model id to pick family-specific behavior
            # (payload shape, arg presets). Read directly from `value` rather than
            # `_get_selected_model_id` because this fires from `_install_model_access`
            # itself, before `self._model_access` exists.
            provider_model_id = value if isinstance(value, str) else ""
            if "deepseek" in provider_model_id:
                self.hide_parameter_by_name("top_p")
            else:
                self.show_parameter_by_name("top_p")

            # Check and see if max_tokens is defined in the model args
            model_args = next((model["args"] for model in MODEL_CHOICES_ARGS if model["name"] == provider_model_id), {})
            if "max_tokens" in model_args:
                self.parameter_output_values["max_tokens"] = model_args["max_tokens"]
            else:
                self.parameter_output_values["max_tokens"] = -1

        return super().after_value_set(parameter, value)

    def process(self) -> None:
        """Emits the `ModelConfig` for the selected Griptape Cloud model.

        Fails closed if the license denies the selected model. The credential
        (API key or License) is resolved when the model is built, and
        `validate_before_workflow_run` checks that one exists. The catalog's
        per-model argument overrides are applied to the settings: a `None` value
        drops a setting the model rejects, and a value replaces the node's. Overrides
        for `stream` and `structured_output_strategy` have no pydantic-ai equivalent
        and are ignored.
        """
        self._raise_if_model_denied()

        provider_model_id = self._get_selected_model_id()
        settings = self._common_settings()
        if provider_model_id in O_SERIES_MODELS:
            settings.pop("top_p", None)

        model_args = next((model["args"] for model in MODEL_CHOICES_ARGS if model["name"] == provider_model_id), {})
        for arg, value in model_args.items():
            if arg not in GENERATION_SETTINGS:
                continue
            if value is None:
                settings.pop(arg, None)
            else:
                settings[arg] = value

        config = self._build_model_config(ModelProvider.GRIPTAPE_CLOUD, provider_model_id, settings=settings)
        self.parameter_output_values["prompt_model_config"] = config

    def validate_before_workflow_run(self) -> list[Exception] | None:
        """Validates that the Griptape Cloud API key is configured correctly.

        Calls the base class helper `_validate_api_key` with Griptape-specific
        configuration details.
        """
        return self._validate_api_key(
            service_name=SERVICE,
            api_key_env_var=API_KEY_ENV_VAR,
            api_key_url=API_KEY_URL,
            # Griptape Cloud accepts a License as well as an API key; a license-only
            # user has no GT_CLOUD_API_KEY, so resolve both before deciding.
            resolved_credential=resolve_cloud_api_key(),
            missing_credential_msg=missing_credential_message("configure the Griptape Cloud prompt driver"),
        )
