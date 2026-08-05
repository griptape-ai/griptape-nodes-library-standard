from griptape.drivers.image_generation.openai_image_generation_driver import (
    OpenAiImageGenerationDriver as GtGrokImageGenerationDriver,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

from griptape_nodes_library.config.image.base_image_driver import BaseImageDriver

# --- Constants ---

SERVICE = "Grok"
API_KEY_URL = "https://console.x.ai"
API_KEY_ENV_VAR = "GROK_API_KEY"
MODEL_CHOICES = ["xai_grok_2_image_1212"]
DEFAULT_MODEL = MODEL_CHOICES[0]

# Migrates values saved before the dropdown stored catalog keys.
LEGACY_MODEL_VALUES = {
    "Grok 2 Image": "xai_grok_2_image_1212",
    "grok-2-image-1212": "xai_grok_2_image_1212",
}


class GrokImage(BaseImageDriver):
    """Node for OpenAI Image Generation Driver.

    This node creates an OpenAI image generation driver and outputs its configuration.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # --- Customize Inherited Parameters ---

        # Offer Grok's models as a license-filtered dropdown.
        self._install_model_access(
            model_choices=MODEL_CHOICES, default_model=DEFAULT_MODEL, deprecated_values=LEGACY_MODEL_VALUES
        )

        # remove the 'size' parameter
        self.remove_parameter_element_by_name("image_size")

    def process(self) -> None:
        # Get the parameters from the node
        params = self.parameter_values

        # A model the license denies must not reach a downstream node as a driver.
        self._raise_if_model_denied()

        # --- Get Common Driver Arguments ---
        # Use the helper method from BaseImageDriver to get common driver arguments
        common_args = self._get_common_driver_args(params)

        # --- Prepare Griptape Cloud Specific Arguments ---
        specific_args = {}

        # Set up the grok url
        specific_args["base_url"] = "https://api.x.ai/v1"

        # Retrieve the mandatory API key.
        specific_args["api_key"] = GriptapeNodes.SecretsManager().get_secret(API_KEY_ENV_VAR)

        # Get the upstream provider's id for the selected model; `common_args["model"]`
        # (from `BaseImageDriver._get_common_driver_args`) is the catalog key.
        specific_args["model"] = self._provider_model_id_for_selection()

        all_kwargs = {**common_args, **specific_args}

        self.parameter_output_values["image_model_config"] = GtGrokImageGenerationDriver(**all_kwargs)

    def validate_node(self) -> list[Exception] | None:
        """Validates that the Griptape Cloud API key is configured correctly.

        Calls the base class helper `_validate_api_key` with Griptape-specific
        configuration details.
        """
        return self._validate_api_key(
            service_name=SERVICE,
            api_key_env_var=API_KEY_ENV_VAR,
            api_key_url=API_KEY_URL,
        )
