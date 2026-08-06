from griptape.drivers.image_generation.griptape_cloud import (
    GriptapeCloudImageGenerationDriver as GtGriptapeCloudImageGenerationDriver,
)
from griptape_nodes.exe_types.core_types import Parameter
from griptape_nodes.traits.options import Options

from griptape_nodes_library.config.image.base_image_driver import BaseImageDriver
from griptape_nodes_library.utils.cloud_credential_utils import (
    missing_credential_message,
    resolve_cloud_api_key,
)

# --- Constants ---

SERVICE = "Griptape"
API_KEY_URL = "https://cloud.griptape.ai/configuration/api-keys"
API_KEY_ENV_VAR = "GT_CLOUD_API_KEY"
MODEL_CHOICES = ["gpt-image-1-mini", "gpt-image-1.5"]
DEFAULT_MODEL = MODEL_CHOICES[0]
AVAILABLE_SIZES = ["1024x1024", "1536x1024", "1024x1536"]
DEFAULT_SIZE = AVAILABLE_SIZES[0]

# Migrates values saved before the dropdown stored the provider's own model id.
LEGACY_MODEL_VALUES = {
    "GPT Image 1 Mini": "gpt-image-1-mini",
    "GPT Image 1.5": "gpt-image-1.5",
    "gtc_gpt_image_1_5": "gpt-image-1.5",
    "gtc_gpt_image_1_mini": "gpt-image-1-mini",
    # Folded in from this node's own retired DEPRECATED_MODELS dict.
    "dall-e-3": "gpt-image-1-mini",
}


class GriptapeCloudImage(BaseImageDriver):
    """Node for Griptape Cloud Image Generation Driver.

    This node creates an Griptape Cloud image generation driver and outputs its configuration.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # --- Customize Inherited Parameters ---

        # Offer Griptape Cloud's models as a license-filtered dropdown.
        self._install_model_access(
            model_choices=MODEL_CHOICES, default_model=DEFAULT_MODEL, deprecated_values=LEGACY_MODEL_VALUES
        )

        # Update the 'size' parameter for Griptape Cloud specifics.
        self._update_option_choices(param="image_size", choices=AVAILABLE_SIZES, default=str(DEFAULT_SIZE))

        # Add additional parameters specific to Griptape Cloud
        self.add_parameter(
            Parameter(
                name="quality",
                type="str",
                default_value="medium",
                tooltip="Select the quality for image generation.",
                traits={Options(choices=["low", "medium", "high"])},
            )
        )

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

        # Retrieve the mandatory API key.
        specific_args["api_key"] = resolve_cloud_api_key()

        # The provider's own id for the selected model.
        specific_args["model"] = self._get_selected_model_id()

        specific_args["quality"] = self.get_parameter_value("quality")

        all_kwargs = {**common_args, **specific_args}

        self.parameter_output_values["image_model_config"] = GtGriptapeCloudImageGenerationDriver(**all_kwargs)

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
            missing_credential_msg=missing_credential_message("configure the Griptape Cloud image driver"),
        )
