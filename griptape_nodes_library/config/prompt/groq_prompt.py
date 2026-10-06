"""Defines the GroqPrompt node for configuring a Groq prompt model.

This module provides the `GroqPrompt` class, which allows users
to configure and utilize the OpenAi prompt service within the Griptape
Nodes framework. It inherits common prompt parameters from `BasePrompt`, sets
Groq specific model options, requires a Groq API key via
node configuration, and emits a `ModelConfig`.
"""

from griptape_nodes_library.config.prompt.base_prompt import BasePrompt
from griptape_nodes_library.llm.model_config import ModelProvider

# --- Constants ---

SERVICE = "Groq"
BASE_URL = "https://api.groq.com/openai/v1"
API_KEY_URL = "https://console.groq.com/keys"
API_KEY_ENV_VAR = "GROQ_API_KEY"
MODEL_CHOICES = [
    "gemma2-9b-it",
    "meta-llama/llama-guard-4-12b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "llama3-70b-8192",
    "llama3-8b-8192",
    "allam-2-7b",
    "deepseek-r1-distill-llama-70b",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
]
DEFAULT_MODEL = MODEL_CHOICES[0]

# Migrates values saved before the dropdown stored the provider's own model id.
LEGACY_MODEL_VALUES = {
    "Allam 2 7B": "allam-2-7b",
    "DeepSeek R1 Distill Llama 70B": "deepseek-r1-distill-llama-70b",
    "Gemma 2 9B IT": "gemma2-9b-it",
    "Llama 3 70B 8192": "llama3-70b-8192",
    "Llama 3 8B 8192": "llama3-8b-8192",
    "Llama 3.1 8B Instant": "llama-3.1-8b-instant",
    "Llama 3.3 70B Versatile": "llama-3.3-70b-versatile",
    "Llama 4 Maverick 17B 128E Instruct": "meta-llama/llama-4-maverick-17b-128e-instruct",
    "Llama 4 Scout 17B 16E Instruct": "meta-llama/llama-4-scout-17b-16e-instruct",
    "Llama Guard 4 12B": "meta-llama/llama-guard-4-12b",
    "groq_allam_2_7b": "allam-2-7b",
    "groq_deepseek_r1_distill_llama_70b": "deepseek-r1-distill-llama-70b",
    "groq_gemma2_9b_it": "gemma2-9b-it",
    "groq_llama3_70b_8192": "llama3-70b-8192",
    "groq_llama3_8b_8192": "llama3-8b-8192",
    "groq_llama_3_1_8b_instant": "llama-3.1-8b-instant",
    "groq_llama_3_3_70b_versatile": "llama-3.3-70b-versatile",
    "groq_llama_4_maverick_17b_128e_instruct": "meta-llama/llama-4-maverick-17b-128e-instruct",
    "groq_llama_4_scout_17b_16e_instruct": "meta-llama/llama-4-scout-17b-16e-instruct",
    "groq_llama_guard_4_12b": "meta-llama/llama-guard-4-12b",
}


class GroqPrompt(BasePrompt):
    """Node for configuring a Groq prompt model.

    Inherits from `BasePrompt` to leverage common LLM parameters. This node
    customizes the available models to those supported by Groq,
    removes parameters not applicable to Groq (like 'seed'), and
    requires a Groq API key to be set in the node's configuration
    under the 'Groq' service.

    The `process` method turns the configured parameters into a `ModelConfig` that
    names the API key secret, and assigns it to the 'prompt_model_config' output parameter.
    """

    def __init__(self, **kwargs) -> None:
        """Initializes the GroqPrompt node.

        Calls the superclass initializer, then modifies the inherited 'model'
        parameter to use Groq specific models and sets a default.
        It also removes the 'seed' parameter inherited from `BasePrompt` as it's
        not directly supported by the Groq implementation.
        """
        super().__init__(**kwargs)

        # --- Customize Inherited Parameters ---

        # Offer Groq's models as a license-filtered dropdown.
        self._install_model_access(
            model_choices=MODEL_CHOICES, default_model=DEFAULT_MODEL, deprecated_values=LEGACY_MODEL_VALUES
        )

        # Remove the 'seed' parameter as it's not directly used by Groq.
        self.remove_parameter_element_by_name("seed")

        # Remove `top_k` parameter as it's not used by Groq.
        self.remove_parameter_element_by_name("top_k")

        # Replace `min_p` with `top_p` for Groq.
        self._replace_param_by_name(param_name="min_p", new_param_name="top_p", default_value=0.9)

    def process(self) -> None:
        """Emits the `ModelConfig` for the selected Groq model.

        Fails closed if the license denies the selected model. The API key is
        referenced by secret name only.
        """
        self._raise_if_model_denied()

        config = self._build_model_config(
            ModelProvider.GROQ,
            self._get_selected_model_id(),
            api_key_secret=API_KEY_ENV_VAR,
            base_url=BASE_URL,
        )
        self.parameter_output_values["prompt_model_config"] = config

    def validate_before_workflow_run(self) -> list[Exception] | None:
        """Validates that the Groq API key is configured correctly.

        Calls the base class helper `_validate_api_key` with Groq-specific
        configuration details.
        """
        return self._validate_api_key(
            service_name=SERVICE,
            api_key_env_var=API_KEY_ENV_VAR,
            api_key_url=API_KEY_URL,
        )
