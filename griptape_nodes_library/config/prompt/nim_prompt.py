"""Defines the NimPrompt node for configuring the OpenAi Prompt Driver.

This module provides the `NimPrompt` class, which allows users
to configure and utilize the OpenAi prompt service within the Griptape
Nodes framework. It inherits common prompt parameters from `BasePrompt`, sets
NVIDIA NIM specific model options, requires a NIM API key via
node configuration, and instantiates the `OpenAiChatPromptDriver`.
"""

from griptape.drivers.prompt.openai import OpenAiChatPromptDriver as GtOpenAiChatPromptDriver
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

from griptape_nodes_library.config.prompt.base_prompt import BasePrompt

# --- Constants ---

SERVICE = "Nvidia"
BASE_URL = "https://integrate.api.nvidia.com/v1"
API_KEY_URL = "https://build.nvidia.com/settings/api-keys"
API_KEY_ENV_VAR = "NVIDIA_API_KEY"
MODEL_CHOICES = [
    "nim_deepseek_v3_1",
    "nim_gemma_3_1b_it",
    "nim_llama_4_maverick_17b_128e_instruct",
    "nim_llama_4_scout_17b_16e_instruct",
    "nim_llama_3_2_11b_vision_instruct",
    "nim_llama_3_2_90b_vision_instruct",
    "nim_llama3_8b_instruct",
    "nim_llama_3_3_nemotron_super_49b_v1_5",
    "nim_llama_3_1_nemotron_nano_vl_8b_v1",
    "nim_nemotron_nano_9b_v2",
    "nim_gpt_oss_20b",
    "nim_gpt_oss_120b",
    "nim_teuken_7b_instruct_commercial_v0_4",
    "nim_kimi_k2_instruct",
    "nim_magistral_small_2506",
]
DEFAULT_MODEL = MODEL_CHOICES[0]

# Migrates values saved before the dropdown stored catalog keys.
LEGACY_MODEL_VALUES = {
    "DeepSeek V3.1": "nim_deepseek_v3_1",
    "GPT-OSS 120B": "nim_gpt_oss_120b",
    "GPT-OSS 20B": "nim_gpt_oss_20b",
    "Gemma 3 1B IT": "nim_gemma_3_1b_it",
    "Kimi K2 Instruct": "nim_kimi_k2_instruct",
    "Llama 3 8B Instruct": "nim_llama3_8b_instruct",
    "Llama 3.1 Nemotron Nano VL 8B v1": "nim_llama_3_1_nemotron_nano_vl_8b_v1",
    "Llama 3.2 11B Vision Instruct": "nim_llama_3_2_11b_vision_instruct",
    "Llama 3.2 90B Vision Instruct": "nim_llama_3_2_90b_vision_instruct",
    "Llama 3.3 Nemotron Super 49B v1.5": "nim_llama_3_3_nemotron_super_49b_v1_5",
    "Llama 4 Maverick 17B 128E Instruct": "nim_llama_4_maverick_17b_128e_instruct",
    "Llama 4 Scout 17B 16E Instruct": "nim_llama_4_scout_17b_16e_instruct",
    "Magistral Small 2506": "nim_magistral_small_2506",
    "Nemotron Nano 9B v2": "nim_nemotron_nano_9b_v2",
    "Teuken 7B Instruct Commercial v0.4": "nim_teuken_7b_instruct_commercial_v0_4",
    "deepseek-ai/deepseek-v3.1": "nim_deepseek_v3_1",
    "google/gemma-3-1b-it": "nim_gemma_3_1b_it",
    "meta/llama-3.2-11b-vision-instruct": "nim_llama_3_2_11b_vision_instruct",
    "meta/llama-3.2-90b-vision-instruct": "nim_llama_3_2_90b_vision_instruct",
    "meta/llama-4-maverick-17b-128e-instruct": "nim_llama_4_maverick_17b_128e_instruct",
    "meta/llama-4-scout-17b-16e-instruct": "nim_llama_4_scout_17b_16e_instruct",
    "meta/llama3-8b-instruct": "nim_llama3_8b_instruct",
    "mistralai/magistral-small-2506": "nim_magistral_small_2506",
    "moonshotai/kimi-k2-instruct": "nim_kimi_k2_instruct",
    "nvidia/llama-3.1-nemotron-nano-vl-8b-v1": "nim_llama_3_1_nemotron_nano_vl_8b_v1",
    "nvidia/llama-3.3-nemotron-super-49b-v1.5": "nim_llama_3_3_nemotron_super_49b_v1_5",
    "nvidia/nvidia-nemotron-nano-9b-v2": "nim_nemotron_nano_9b_v2",
    "openai/gpt-oss-120b": "nim_gpt_oss_120b",
    "openai/gpt-oss-20b": "nim_gpt_oss_20b",
    "opengpt-x/teuken-7b-instruct-commercial-v0.4": "nim_teuken_7b_instruct_commercial_v0_4",
}


class NimPrompt(BasePrompt):
    """Node for configuring and providing a NVIDIA Chat Prompt Driver.

    Inherits from `BasePrompt` to leverage common LLM parameters. This node
    customizes the available models to those supported by NVIDIA,
    removes parameters not applicable to NVIDIA (like 'seed'), and
    requires a NVIDIA API key to be set in the node's configuration
    under the 'NVIDIA' service.

    The `process` method gathers the configured parameters and the API key,
    utilizes the `_get_common_driver_args` helper from `BasePrompt`, adds
    NVIDIA specific configurations, then instantiates a
    `OpenAiChatPromptDriver` with NVIDIA specific configurations and assigns it to the 'prompt_model_config'
    output parameter.
    """

    def __init__(self, **kwargs) -> None:
        """Initializes the NimPrompt node.

        Calls the superclass initializer, then modifies the inherited 'model'
        parameter to use NVIDIA specific models and sets a default.
        It also removes the 'seed' parameter inherited from `BasePrompt` as it's
        not directly supported by the NVIDIA driver implementation.
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
        """Processes the node configuration to create a NIM PromptDriver.

        Retrieves parameter values set on the node and the required API key from
        the node's configuration system. It constructs the arguments dictionary
        for the `OpenAiChatPromptDriver` with NVIDIA specific configurations, handles optional parameters and
        any necessary conversions (like 'min_p' to 'top_p'), instantiates the
        driver, and assigns it to the 'prompt_model_config' output parameter.

        Raises:
            KeyError: If the NVIDIA API key is not found in the node configuration
                      (though `validate_before_workflow_run` should prevent this during execution).
        """
        # Retrieve all parameter values set on the node UI or via input connections.
        params = self.parameter_values

        # A model the license denies must not reach a downstream node as a driver.
        self._raise_if_model_denied()

        # --- Get Common Driver Arguments ---
        # Use the helper method from BasePrompt to get args like temperature, stream, max_attempts, etc.
        common_args = self._get_common_driver_args(params)

        # --- Prepare NVIDIA Specific Arguments ---
        specific_args = {}

        # Retrieve the mandatory API key.
        specific_args["api_key"] = GriptapeNodes.SecretsManager().get_secret(API_KEY_ENV_VAR)

        # Set the base URL for the NVIDIA API.
        specific_args["base_url"] = BASE_URL

        # Get the upstream provider's id for the selected model.
        specific_args["model"] = self._provider_model_id_for_selection()

        # Handle parameters that go into 'extra_params' for NVIDIA.
        extra_params = {}

        extra_params["top_p"] = self.get_parameter_value("top_p")

        # Assign extra_params if not empty
        if extra_params:
            specific_args["extra_params"] = extra_params

        # --- Combine Arguments and Instantiate Driver ---
        # Combine common arguments with Nvidia specific arguments.
        # Specific args take precedence if there's an overlap (though unlikely here).
        all_kwargs = {**common_args, **specific_args}

        # Create the Nvidia prompt driver instance.
        driver = GtOpenAiChatPromptDriver(**all_kwargs)

        # Set the output parameter 'prompt_model_config'.
        self.parameter_output_values["prompt_model_config"] = driver

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
