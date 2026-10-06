"""Defines the NimPrompt node for configuring a NVIDIA NIM prompt model.

This module provides the `NimPrompt` class, which allows users
to configure and utilize the OpenAi prompt service within the Griptape
Nodes framework. It inherits common prompt parameters from `BasePrompt`, sets
NVIDIA NIM specific model options, requires a NIM API key via
node configuration, and emits a `ModelConfig`.
"""

from griptape_nodes_library.config.prompt.base_prompt import BasePrompt
from griptape_nodes_library.llm.model_config import ModelProvider

# --- Constants ---

SERVICE = "Nvidia"
BASE_URL = "https://integrate.api.nvidia.com/v1"
API_KEY_URL = "https://build.nvidia.com/settings/api-keys"
API_KEY_ENV_VAR = "NVIDIA_API_KEY"
MODEL_CHOICES = [
    "deepseek-ai/deepseek-v3.1",
    "google/gemma-3-1b-it",
    "meta/llama-4-maverick-17b-128e-instruct",
    "meta/llama-4-scout-17b-16e-instruct",
    "meta/llama-3.2-11b-vision-instruct",
    "meta/llama-3.2-90b-vision-instruct",
    "meta/llama3-8b-instruct",
    "nvidia/llama-3.3-nemotron-super-49b-v1.5",
    "nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
    "nvidia/nvidia-nemotron-nano-9b-v2",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "opengpt-x/teuken-7b-instruct-commercial-v0.4",
    "moonshotai/kimi-k2-instruct",
    "mistralai/magistral-small-2506",
]
DEFAULT_MODEL = MODEL_CHOICES[0]

# Migrates values saved before the dropdown stored the provider's own model id.
LEGACY_MODEL_VALUES = {
    "DeepSeek V3.1": "deepseek-ai/deepseek-v3.1",
    "GPT-OSS 120B": "openai/gpt-oss-120b",
    "GPT-OSS 20B": "openai/gpt-oss-20b",
    "Gemma 3 1B IT": "google/gemma-3-1b-it",
    "Kimi K2 Instruct": "moonshotai/kimi-k2-instruct",
    "Llama 3 8B Instruct": "meta/llama3-8b-instruct",
    "Llama 3.1 Nemotron Nano VL 8B v1": "nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
    "Llama 3.2 11B Vision Instruct": "meta/llama-3.2-11b-vision-instruct",
    "Llama 3.2 90B Vision Instruct": "meta/llama-3.2-90b-vision-instruct",
    "Llama 3.3 Nemotron Super 49B v1.5": "nvidia/llama-3.3-nemotron-super-49b-v1.5",
    "Llama 4 Maverick 17B 128E Instruct": "meta/llama-4-maverick-17b-128e-instruct",
    "Llama 4 Scout 17B 16E Instruct": "meta/llama-4-scout-17b-16e-instruct",
    "Magistral Small 2506": "mistralai/magistral-small-2506",
    "Nemotron Nano 9B v2": "nvidia/nvidia-nemotron-nano-9b-v2",
    "Teuken 7B Instruct Commercial v0.4": "opengpt-x/teuken-7b-instruct-commercial-v0.4",
    "nim_deepseek_v3_1": "deepseek-ai/deepseek-v3.1",
    "nim_gemma_3_1b_it": "google/gemma-3-1b-it",
    "nim_gpt_oss_120b": "openai/gpt-oss-120b",
    "nim_gpt_oss_20b": "openai/gpt-oss-20b",
    "nim_kimi_k2_instruct": "moonshotai/kimi-k2-instruct",
    "nim_llama3_8b_instruct": "meta/llama3-8b-instruct",
    "nim_llama_3_1_nemotron_nano_vl_8b_v1": "nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
    "nim_llama_3_2_11b_vision_instruct": "meta/llama-3.2-11b-vision-instruct",
    "nim_llama_3_2_90b_vision_instruct": "meta/llama-3.2-90b-vision-instruct",
    "nim_llama_3_3_nemotron_super_49b_v1_5": "nvidia/llama-3.3-nemotron-super-49b-v1.5",
    "nim_llama_4_maverick_17b_128e_instruct": "meta/llama-4-maverick-17b-128e-instruct",
    "nim_llama_4_scout_17b_16e_instruct": "meta/llama-4-scout-17b-16e-instruct",
    "nim_magistral_small_2506": "mistralai/magistral-small-2506",
    "nim_nemotron_nano_9b_v2": "nvidia/nvidia-nemotron-nano-9b-v2",
    "nim_teuken_7b_instruct_commercial_v0_4": "opengpt-x/teuken-7b-instruct-commercial-v0.4",
}


class NimPrompt(BasePrompt):
    """Node for configuring a NVIDIA NIM prompt model.

    Inherits from `BasePrompt` to leverage common LLM parameters. This node
    customizes the available models to those supported by NVIDIA,
    removes parameters not applicable to NVIDIA (like 'seed'), and
    requires a NVIDIA API key to be set in the node's configuration
    under the 'NVIDIA' service.

    The `process` method turns the configured parameters into a `ModelConfig` that
    names the API key secret, and assigns it to the 'prompt_model_config' output parameter.
    """

    def __init__(self, **kwargs) -> None:
        """Initializes the NimPrompt node.

        Calls the superclass initializer, then modifies the inherited 'model'
        parameter to use NVIDIA specific models and sets a default.
        It also removes the 'seed' parameter inherited from `BasePrompt` as it's
        not directly supported by the NVIDIA implementation.
        """
        super().__init__(**kwargs)

        # --- Customize Inherited Parameters ---

        # Offer NVIDIA's models as a license-filtered dropdown.
        self._install_model_access(
            model_choices=MODEL_CHOICES, default_model=DEFAULT_MODEL, deprecated_values=LEGACY_MODEL_VALUES
        )

        # Remove the 'seed' parameter
        self.remove_parameter_element_by_name("seed")

        # Remove `top_k` parameter as it's not used by NVIDIA.
        self.remove_parameter_element_by_name("top_k")

        # Replace `min_p` with `top_p` for NIM.
        self._replace_param_by_name(param_name="min_p", new_param_name="top_p", default_value=0.9)

    def process(self) -> None:
        """Emits the `ModelConfig` for the selected Nim model.

        Fails closed if the license denies the selected model. The API key is
        referenced by secret name only.
        """
        self._raise_if_model_denied()

        config = self._build_model_config(
            ModelProvider.NIM,
            self._get_selected_model_id(),
            api_key_secret=API_KEY_ENV_VAR,
            base_url=BASE_URL,
        )
        self.parameter_output_values["prompt_model_config"] = config

    def validate_before_workflow_run(self) -> list[Exception] | None:
        """Validates that the Nvidia API key is configured correctly.

        Calls the base class helper `_validate_api_key` with Nvidia-specific
        configuration details.
        """
        return self._validate_api_key(
            service_name=SERVICE,
            api_key_env_var=API_KEY_ENV_VAR,
            api_key_url=API_KEY_URL,
        )
