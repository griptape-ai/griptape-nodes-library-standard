"""Defines the OpenAiPrompt node for configuring an OpenAI prompt model.

This module provides the `OpenAiPrompt` class, which allows users
to configure and utilize the OpenAi prompt service within the Griptape
Nodes framework. It inherits common prompt parameters from `BasePrompt`, sets
OpenAi specific model options, requires a OpenAi API key via
node configuration, and emits a `ModelConfig`.
"""

from griptape_nodes.traits.options import Options

from griptape_nodes_library.config.prompt.base_prompt import BasePrompt
from griptape_nodes_library.llm.model_config import ModelProvider

# --- Constants ---

SERVICE = "OpenAI"
API_KEY_URL = "https://platform.openai.com/api-keys"
API_KEY_ENV_VAR = "OPENAI_API_KEY"
MODEL_CHOICES = [
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
    "gpt-5.5",
    "gpt-5.4",
    "gpt-5.2",
    "gpt-5.1",
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-4o",
    "o4-mini",
    "o3",
    "o3-mini",
    "o1",
]
DEFAULT_MODEL = MODEL_CHOICES[0]


class OpenAiPrompt(BasePrompt):
    """Node for configuring an OpenAI prompt model.

    Inherits from `BasePrompt` to leverage common LLM parameters. This node
    customizes the available models to those supported by OpenAi,
    removes parameters not applicable to OpenAi (like 'seed'), and
    requires a OpenAi API key to be set in the node's configuration
    under the 'OpenAi' service.

    The `process` method turns the configured parameters into a `ModelConfig` that
    names the API key secret, and assigns it to the 'prompt_model_config' output parameter.
    """

    def __init__(self, **kwargs) -> None:
        """Initializes the OpenAiPrompt node.

        Calls the superclass initializer, then modifies the inherited 'model'
        parameter to use OpenAi specific models and sets a default.
        It also removes the 'seed' parameter inherited from `BasePrompt` as it's
        not directly supported by the OpenAi implementation.
        """
        super().__init__(**kwargs)

        # --- Customize Inherited Parameters ---

        # Update the 'model' parameter for OpenAi specifics.
        # call openai listmodels to get available models
        import openai

        available_models = [model.id for model in openai.Client().models.list().data]
        # This node offers whatever models the account can see rather than a
        # declared list, so it owns the dropdown trait instead of routing through
        # `_install_model_access`.
        model_param = self.get_parameter_by_name("model")
        if model_param is not None:
            model_param.add_trait(Options(choices=available_models))
        self._update_option_choices(
            param="model", choices=available_models, default=available_models[0] if available_models else ""
        )

        # Remove the 'seed' parameter as it's not directly used by OpenAI.
        self.remove_parameter_element_by_name("seed")

        # Remove `top_k` parameter as it's not used by OpenAi.
        self.remove_parameter_element_by_name("top_k")

        # Replace `min_p` with `top_p` for OpenAi.
        self._replace_param_by_name(param_name="min_p", new_param_name="top_p", default_value=0.9)

    def process(self) -> None:
        """Emits the `ModelConfig` for the selected OpenAI model.

        The API key is referenced by secret name only; it is resolved when the model is built.
        """
        config = self._build_model_config(
            ModelProvider.OPENAI,
            self.get_parameter_value("model"),
            api_key_secret=API_KEY_ENV_VAR,
        )
        self.parameter_output_values["prompt_model_config"] = config

    def validate_before_workflow_run(self) -> list[Exception] | None:
        """Validates that the OpenAi API key is configured correctly.

        Calls the base class helper `_validate_api_key` with OpenAi-specific
        configuration details.
        """
        return self._validate_api_key(
            service_name=SERVICE,
            api_key_env_var=API_KEY_ENV_VAR,
            api_key_url=API_KEY_URL,
        )
