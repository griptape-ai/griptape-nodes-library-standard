from griptape_nodes_library.config.image.base_image_driver import BaseImageDriver
from griptape_nodes_library.llm.image_generation import GROK_BASE_URL, ImageGenerationConfig, ImageProvider

# --- Constants ---

SERVICE = "Grok"
API_KEY_URL = "https://console.x.ai"
API_KEY_ENV_VAR = "GROK_API_KEY"
MODEL_CHOICES = ["grok-2-image-1212"]
DEFAULT_MODEL = MODEL_CHOICES[0]

# Migrates values saved before the dropdown stored the provider's own model id.
LEGACY_MODEL_VALUES = {
    "Grok 2 Image": "grok-2-image-1212",
    "xai_grok_2_image_1212": "grok-2-image-1212",
}


class GrokImage(BaseImageDriver):
    """Node for Grok Image Generation Driver.

    This node outputs a Grok image generation configuration.
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
        # A model the license denies must not reach a downstream node as a driver.
        self._raise_if_model_denied()

        self.parameter_output_values["image_model_config"] = ImageGenerationConfig(
            provider=ImageProvider.GROK,
            model=self._get_selected_model_id(),
            base_url=GROK_BASE_URL,
            api_key_secret=API_KEY_ENV_VAR,
        )

    def validate_node(self) -> list[Exception] | None:
        """Validates that the Grok API key is configured correctly.

        Calls the base class helper `_validate_api_key` with Grok-specific
        configuration details.
        """
        return self._validate_api_key(
            service_name=SERVICE,
            api_key_env_var=API_KEY_ENV_VAR,
            api_key_url=API_KEY_URL,
        )
