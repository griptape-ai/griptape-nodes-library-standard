from __future__ import annotations

import base64
import json as _json
import logging
from contextlib import suppress
from copy import deepcopy
from io import BytesIO
from typing import Any

from griptape.artifacts import ImageArtifact, ImageUrlArtifact
from griptape_nodes.exe_types.core_types import (
    Parameter,
    ParameterGroup,
    ParameterList,
    ParameterMode,
)
from griptape_nodes.exe_types.param_components.model_access_component import ModelAccessComponent
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_types.parameter_dict import ParameterDict
from griptape_nodes.exe_types.param_types.parameter_image import ParameterImage
from griptape_nodes.exe_types.param_types.parameter_int import ParameterInt
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.files.file import File, FileLoadError
from griptape_nodes.traits.options import Options
from griptape_nodes.utils.artifact_normalization import normalize_artifact_list
from PIL import Image

from griptape_nodes_library.proxy import GriptapeProxyNode

logger = logging.getLogger("griptape_nodes")

__all__ = ["SeedreamImageGeneration"]

# Define constant for prompt truncation length
PROMPT_TRUNCATE_LENGTH = 100

# The models this node offers, as the provider's own ids -- the dropdown stores these and
# the payload sends them as-is. Only Seedream 5.0 Pro carries the "dola-" prefix; the
# provider's ids are inconsistent and the unprefixed ids are what the other models answer to.
MODEL_CHOICES = [
    "dola-seedream-5-0-pro-260628",
    "seedream-5-0-260128",
    "seedream-4-5-251128",
]

# Size options for different models, keyed by the provider's own model id
SIZE_OPTIONS = {
    # A resolution token lets the model choose the aspect ratio from the prompt; an explicit
    # WxH pins it. 1K and 1.5K cost the same, so 1K's explicit dimensions are omitted as
    # strictly lower quality for the price; the token remains for smaller, faster output.
    "dola-seedream-5-0-pro-260628": [
        "1K",
        "1.5K",
        "2K",
        "1536x1536",
        "1792x1344",
        "1344x1792",
        "2048x1152",
        "1152x2048",
        "1872x1248",
        "1248x1872",
        "2352x1008",
        "2048x2048",
        "2368x1776",
        "1776x2368",
        "2816x1584",
        "1584x2816",
        "2496x1664",
        "1664x2496",
        "3136x1344",
    ],
    "seedream-5-0-260128": [
        "2K",
        "3K",
        "4K",
        "2048x2048",
        "2304x1728",
        "1728x2304",
        "2848x1600",
        "1600x2848",
        "2496x1664",
        "1664x2496",
        "3136x1344",
        "3072x3072",
        "3456x2592",
        "2592x3456",
        "4096x2304",
        "2304x4096",
        "2496x3744",
        "3744x2496",
        "4704x2016",
        "4096x4096",
        "3520x4704",
        "4704x3520",
        "5504x3040",
        "3040x5504",
        "3328x4992",
        "4992x3328",
        "6240x2656",
    ],
    "seedream-4-5-251128": [
        "2K",
        "4K",
        "2048x2048",
        "2304x1728",
        "1728x2304",
        "2848x1600",
        "1600x2848",
        "2496x1664",
        "1664x2496",
        "3136x1344",
        "2560x1440",
        "1440x2560",
        "3840x2160",
        "2160x3840",
        "4096x2160",
        "2160x4096",
        "4096x4096",
        "3520x4704",
        "4704x3520",
        "5504x3040",
        "3040x5504",
        "3328x4992",
        "4992x3328",
        "6240x2656",
    ],
}

DEFAULT_MODEL = "seedream-4-5-251128"

# Size selected when the current size isn't offered by the newly selected model.
# Seedream 5.0 Pro deliberately differs from the provider default of 2K: 1.5K is billed at the
# 1K rate while producing better images, so it is the best value rather than the cheapest option.
DEFAULT_SIZE_PER_MODEL = {
    "dola-seedream-5-0-pro-260628": "1.5K",
    "seedream-5-0-260128": "2K",
    "seedream-4-5-251128": "2K",
}

# Maximum number of input images for models that support multiple images,
# keyed by the provider's own model id
MAX_IMAGES_PER_MODEL = {
    "dola-seedream-5-0-pro-260628": 10,
    "seedream-5-0-260128": 14,
    "seedream-4-5-251128": 14,
}

OUTPUT_FORMAT_OPTIONS = ["jpeg", "png"]

# Prompt optimization modes each model accepts. Only Seedream 5.0 Pro takes "fast"; the others
# reject the request with "optimize_prompt_options.mode must be 'standard'". The first entry is
# the fallback when the current selection isn't supported by a newly selected model.
OPTIMIZE_PROMPT_MODES_PER_MODEL = {
    "dola-seedream-5-0-pro-260628": ["standard", "fast"],
    "seedream-5-0-260128": ["standard"],
    "seedream-4-5-251128": ["standard"],
}

# Migrates values saved before this dropdown stored the provider's own model id: friendly
# labels, catalog keys, and the ids of models that have since been retired. Migration is
# single-hop, so every value here must be one of MODEL_CHOICES -- never another retired model.
LEGACY_MODEL_VALUES = {
    "Seedream 5.0 Pro": "dola-seedream-5-0-pro-260628",
    "Seedream 5.0 Lite": "seedream-5-0-260128",
    "Seedream 4.5": "seedream-4-5-251128",
    "gtc_seedream_5_0_pro": "dola-seedream-5-0-pro-260628",
    "gtc_seedream_5_0_lite": "seedream-5-0-260128",
    "gtc_seedream_4_5": "seedream-4-5-251128",
    # Retired models: Seedream 4.0 and the 3.0 pair all migrate to 5.0 Lite.
    "Seedream 4.0": "seedream-5-0-260128",
    "seedream-4-0-250828": "seedream-5-0-260128",
    "gtc_seedream_4_0": "seedream-5-0-260128",
    "Seedream 3.0 T2I": "seedream-5-0-260128",
    "seedream-3-0-t2i-250415": "seedream-5-0-260128",
    "Seedream 3.0 I2I": "seedream-5-0-260128",
    "seededit-3-0-i2i-250628": "seedream-5-0-260128",
    # Never a valid provider id; shipped in a workflow template and rejected by the proxy.
    "seedream-4.5": "seedream-4-5-251128",
}


class SeedreamImageGeneration(GriptapeProxyNode):
    """Generate images using Seedream models via Griptape model proxy.

    Supports three models:
    - Seedream 5.0 Pro: Single-image model with optional multiple image inputs (up to 10).
      Rejects the batch generation fields outright, so max_images does not apply.
      Reference images after the first are billed, unlike the other Seedream models.
      Size options: 1K, 1.5K, 2K (aspect ratio taken from the prompt) or explicit dimensions.
      Total pixels range: [921,600, 4,624,220], Aspect ratio: [1/16, 16]
      1K and 1.5K are billed at the same rate; 2K costs twice as much.
    - Seedream 5.0 Lite: Text-to-image model with optional multiple image inputs (up to 14)
      Size options: 2K, 3K, 4K (aspect ratio taken from the prompt) or explicit dimensions
      Total pixels range: [3,686,400, 16,777,216], Aspect ratio: [1/16, 16]
    - Seedream 4.5: Multi-image model with optional multiple image inputs (up to 14)
      Size options: 2K, 4K (aspect ratio taken from the prompt) or explicit dimensions
      Total pixels range: [3,686,400, 16,777,216], Aspect ratio: [1/16, 16]

    Seedream 4.0 and the Seedream 3.0 models are deprecated; selecting one migrates to a
    supported model and surfaces a dismissible notice asking the user to re-save.

    Inputs:
        - model (str): Model selection (Seedream 5.0 Pro, Seedream 5.0 Lite, Seedream 4.5)
        - prompt (str): Text prompt for image generation
        - images (list): Multiple input images (Seedream 5.0 Lite/4.5 support up to 14, Seedream 5.0 Pro up to 10)
        - size (str): Image size specification (dynamic options based on selected model)
        - max_images (int): Maximum number of images to generate (1-15, not supported by Seedream 5.0 Pro)
        - output_format (str): Output image format - jpeg or png (Seedream 5.0 Pro and 5.0 Lite only, default: jpeg)
        - optimize_prompt_mode (str): Prompt optimization mode - standard or fast (not supported by Seedream 4.5;
          fast is only supported by Seedream 5.0 Pro, default: standard)

    Outputs:
        - generation_id (str): Generation ID from the API
        - provider_response (dict): Verbatim provider response from the model proxy
        - image_url (ImageUrlArtifact): First generated image (always visible, backwards compatible)
        - image_url_2, image_url_3, ..., image_url_N (ImageUrlArtifact): Additional images (shown when API returns multiple images)
        - was_successful (bool): Whether the generation succeeded
        - result_details (str): Details about the generation result or error
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.category = "API Nodes"
        self.description = "Generate images using Seedream models via Griptape model proxy"

        # Model selection
        model_param = ParameterString(
            name="model",
            default_value=DEFAULT_MODEL,
            tooltip="Select the Seedream model to use",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
        )
        self.add_parameter(model_param)
        # License-policy dropdown: the component adds Options + refresh Button traits, marks the
        # models the license denies, and migrates saved values through LEGACY_MODEL_VALUES. The
        # proxy base class refuses to submit a denied selection.
        self._model_access = ModelAccessComponent(
            node=self,
            parameter=model_param,
            model_choices=MODEL_CHOICES,
            default_model=DEFAULT_MODEL,
            deprecated_values=LEGACY_MODEL_VALUES,
        )

        # Core parameters
        self.add_parameter(
            ParameterString(
                name="prompt",
                default_value="",
                tooltip="Text prompt for image generation (max 600 words recommended)",
                multiline=True,
                placeholder_text="Describe the image you want to generate...",
                allow_output=False,
                ui_options={
                    "display_name": "Prompt",
                },
            )
        )

        # Multiple image inputs for multi-image Seedream models (up to 14/10 images, model-dependent)
        self.add_parameter(
            ParameterList(
                name="images",
                input_types=[
                    "ImageArtifact",
                    "ImageUrlArtifact",
                    "str",
                    "list",
                    "list[ImageArtifact]",
                    "list[ImageUrlArtifact]",
                ],
                default_value=[],
                tooltip=(
                    "Input images for Seedream (up to 14 for Seedream 5.0 Lite/4.5, "
                    "10 for Seedream 5.0 Pro). "
                    "Seedream 5.0 Pro bills every reference image after the first; the other models do not."
                ),
                allowed_modes={ParameterMode.INPUT},
                ui_options={"expander": True, "display_name": "Input Images", "hide_property": True},
            )
        )

        # Size parameter - will be updated dynamically based on model selection
        self.add_parameter(
            ParameterString(
                name="size",
                default_value="2K",
                tooltip="Image size specification",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=SIZE_OPTIONS[DEFAULT_MODEL])},
            )
        )

        with ParameterGroup(name="Generation Settings", ui_options={"collapsed": True}) as generation_settings_group:
            ParameterInt(
                name="max_images",
                tooltip="Maximum number of images to generate (1-15)",
                default_value=10,
                slider=True,
                min_val=1,
                max_val=15,
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                hide=False,
            )

            ParameterString(
                name="output_format",
                default_value="jpeg",
                tooltip="Output image format (Seedream 5.0 Pro and 5.0 Lite only)",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=OUTPUT_FORMAT_OPTIONS)},
                ui_options={"hide": True},
            )

            ParameterString(
                name="optimize_prompt_mode",
                default_value="standard",
                tooltip=(
                    "Prompt optimization mode: standard (higher quality) or fast. "
                    "Seedream 4.5 does not support this; only Seedream 5.0 Pro supports fast."
                ),
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=["standard", "fast"])},
                ui_options={"hide": True},
            )

        self.add_node_element(generation_settings_group)

        # OUTPUTS
        self.add_parameter(
            ParameterDict(
                name="provider_response",
                tooltip="Verbatim response from Griptape model proxy",
                allowed_modes={ParameterMode.OUTPUT},
                hide_property=True,
                hide=True,
            )
        )

        # Create all image output parameters upfront (1-15) so they render in one block
        # First parameter is 'image_url' for backwards compatibility, rest are 'image_url_2' through 'image_url_15'
        # Only image_url is visible initially; others are shown when API returns multiple images
        for i in range(1, 16):
            param_name = "image_url" if i == 1 else f"image_url_{i}"
            self.add_parameter(
                ParameterImage(
                    name=param_name,
                    tooltip=f"Generated image {i}",
                    allowed_modes={ParameterMode.OUTPUT, ParameterMode.PROPERTY},
                    settable=False,
                    ui_options={"pulse_on_run": True, "hide": i > 1},
                )
            )

        self._output_file = ProjectFileParameter(
            node=self,
            name="output_file",
            default_filename="seedream_image.jpg",
        )
        self._output_file.add_parameter()

        # Create status parameters for success/failure tracking (at the end)
        self._create_status_parameters(
            result_details_tooltip="Details about the image generation result or any errors",
            result_details_placeholder="Generation status and details will appear here.",
            parameter_group_initially_collapsed=True,
        )

        # Initialize parameter visibility based on default model (Seedream 4.5)
        self._initialize_parameter_visibility()

    def _show_image_output_parameters(self, count: int) -> None:
        """Show image output parameters based on actual result count.

        All 15 image parameters are created during initialization but hidden except image_url.
        This method shows the appropriate number based on the API response.

        Args:
            count: Total number of images returned from API (1-15)
        """
        for i in range(1, 16):
            param_name = "image_url" if i == 1 else f"image_url_{i}"
            if i <= count:
                self.show_parameter_by_name(param_name)
            else:
                self.hide_parameter_by_name(param_name)

    def _initialize_parameter_visibility(self) -> None:
        """Initialize parameter visibility based on default model selection."""
        default_model = self.get_parameter_value("model") or DEFAULT_MODEL
        self._apply_model_parameter_visibility(default_model)

    def _apply_model_parameter_visibility(self, model: str) -> None:
        """Show or hide the model-dependent generation settings for the selected model."""
        match model:
            case "dola-seedream-5-0-pro-260628":
                # Pro rejects the batch fields outright, but does take output_format and
                # prompt optimization (including the "fast" mode that Lite refuses).
                visible = {"output_format", "optimize_prompt_mode"}
            case "seedream-5-0-260128":
                visible = {"max_images", "output_format", "optimize_prompt_mode"}
            case "seedream-4-5-251128":
                visible = {"max_images"}
            case _:
                msg = f"Unknown Seedream model: {model!r}"
                raise ValueError(msg)

        for param_name in ("max_images", "output_format", "optimize_prompt_mode"):
            if param_name in visible:
                self.show_parameter_by_name(param_name)
            else:
                self.hide_parameter_by_name(param_name)

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        """Update size options and parameter visibility based on parameter changes."""
        if parameter.name == "model" and value in SIZE_OPTIONS:
            self._update_model_parameters(value)

        # Convert string paths to ImageUrlArtifact by uploading to static storage
        if parameter.name == "images" and isinstance(value, list):
            updated_list = normalize_artifact_list(value, ImageUrlArtifact, accepted_types=(ImageArtifact,))
            if updated_list != value:
                self.set_parameter_value("images", updated_list)

        return super().after_value_set(parameter, value)

    def _update_model_parameters(self, model: str) -> None:
        """Update parameters and UI based on selected model."""
        new_choices = SIZE_OPTIONS[model]
        current_size = self.get_parameter_value("size")

        self._apply_model_parameter_visibility(model)
        self._update_size_badge(model)
        self._reconcile_optimize_prompt_mode(model)

        # Seedream 5.0 Pro charges twice as much above its pixel threshold, and "2K" is over it.
        # Carrying that size over from a model where it was free of consequence would silently
        # double the cost, so Pro always starts from its cheaper default.
        if current_size in new_choices and model != "dola-seedream-5-0-pro-260628":
            self._update_option_choices("size", new_choices, current_size)
        else:
            default_size = DEFAULT_SIZE_PER_MODEL[model]
            default_size = default_size if default_size in new_choices else new_choices[0]
            self._update_option_choices("size", new_choices, default_size)

    def _reconcile_optimize_prompt_mode(self, model: str) -> None:
        """Restrict the prompt optimization choices to the modes the selected model accepts.

        Only Seedream 5.0 Pro accepts "fast"; the others reject the request outright. Migrating a
        deprecated model that did accept it would otherwise carry "fast" into a model that does not
        and fail at the provider after the UI reported a successful migration.
        """
        supported_modes = OPTIMIZE_PROMPT_MODES_PER_MODEL[model]
        current_mode = self.get_parameter_value("optimize_prompt_mode")
        next_mode = current_mode if current_mode in supported_modes else supported_modes[0]
        self._update_option_choices("optimize_prompt_mode", supported_modes, next_mode)

    def _update_size_badge(self, model: str) -> None:
        """Surface Seedream 5.0 Pro's two price tiers, which the size choices don't convey."""
        size_parameter = self.get_parameter_by_name("size")
        if size_parameter is None:
            return

        if model == "dola-seedream-5-0-pro-260628":
            size_parameter.set_badge(
                variant="note",
                title="2K costs twice as much",
                message=(
                    "`1K` and `1.5K` are billed at the same rate, so `1.5K` is the better value. "
                    "`2K` and any explicit size above 2,360,000 pixels cost twice as much.\n\n"
                    "A resolution level (`1K`, `1.5K`, `2K`) lets the model pick the aspect ratio "
                    "from your prompt; an explicit `WxH` pins it."
                ),
            )
        else:
            size_parameter.clear_badge()

    def _log(self, message: str) -> None:
        with suppress(Exception):
            logger.info(message)

    async def _parse_result(self, result_json: dict[str, Any], generation_id: str) -> None:
        """Parse the result and set output parameters.

        Args:
            result_json: The JSON response from the /result endpoint
            generation_id: The generation ID for this request
        """
        # Extract image data
        data = result_json.get("data", [])
        if not data:
            self._log("No image data in result")
            self._set_safe_defaults()
            self._set_status_results(
                was_successful=False,
                result_details=f"{self.name} generation completed but no image data was found in the response.",
            )
            return

        # Process all images from the response
        image_artifacts = []
        failed_urls = []
        for idx, image_data in enumerate(data):
            image_url = image_data.get("url")
            if not image_url:
                self._log(f"No URL found for image {idx}")
                continue

            artifact = await self._save_single_image_from_url(image_url, generation_id, idx)
            if artifact:
                image_artifacts.append(artifact)
            else:
                failed_urls.append(image_url)

        if not image_artifacts:
            self._log("No images could be saved")
            self._set_safe_defaults()
            if failed_urls:
                details = (
                    f"{self.name} generation completed upstream but the image(s) could not be retrieved. "
                    f"Provider URL(s) (may be temporary): {', '.join(failed_urls)}"
                )
            else:
                details = f"{self.name} generation completed but no image URLs were found in the response."
            self._set_status_results(was_successful=False, result_details=details)
            return

        # Show the appropriate number of image output parameters based on actual image count
        self._show_image_output_parameters(len(image_artifacts))

        # These parameters are PROPERTY|OUTPUT, so the stored value backs what the editor renders
        # while the output value feeds downstream nodes. Setting only the output leaves the stored
        # value empty and the image shows as a placeholder until a reload rehydrates it. This is
        # the same set/publish/output sequence ExecutionStatusComponent uses for its own params.
        for idx, artifact in enumerate(image_artifacts, start=1):
            param_name = "image_url" if idx == 1 else f"image_url_{idx}"
            self.set_parameter_value(param_name, artifact)
            self.publish_update_to_parameter(param_name, artifact)
            self.parameter_output_values[param_name] = artifact

        # Set success status
        count = len(image_artifacts)
        filenames = [artifact.name for artifact in image_artifacts]
        if count == 1:
            details = f"Image generated successfully and saved as {filenames[0]}."
        else:
            details = f"Generated {count} images successfully: {', '.join(filenames)}."
        self._set_status_results(was_successful=True, result_details=details)

    def validate_before_node_run(self) -> list[Exception] | None:
        """Validate parameters before running the node."""
        exceptions = []
        model = self.get_parameter_value("model")

        # Validate image count for models that support multiple images
        if model in MAX_IMAGES_PER_MODEL:
            max_images = MAX_IMAGES_PER_MODEL[model]
            images = self.get_parameter_list_value("images") or []
            if len(images) > max_images:
                exceptions.append(
                    ValueError(f"{self.name}: {model} supports maximum {max_images} images, got {len(images)}")
                )

        return exceptions if exceptions else None

    def _get_parameters(self) -> dict[str, Any]:
        images = self.get_parameter_list_value("images") or []

        # Normalize string paths to ImageUrlArtifact during processing
        # (handles cases where values come from connections and bypass after_value_set)
        images = normalize_artifact_list(images, ImageUrlArtifact, accepted_types=(ImageArtifact,))

        model = self.get_parameter_value("model") or DEFAULT_MODEL

        return {
            "model": model,
            "prompt": self.get_parameter_value("prompt") or "",
            "images": images,
            # Always send an explicit size. The proxy derives Seedream 5.0 Pro's cost tier from
            # the requested size, so falling back to a size the user didn't pick can double the bill.
            "size": self.get_parameter_value("size") or DEFAULT_SIZE_PER_MODEL[model],
            "output_format": self.get_parameter_value("output_format") or "jpeg",
            "optimize_prompt_mode": self.get_parameter_value("optimize_prompt_mode") or "standard",
            "watermark": False,
            "sequential_image_generation": "auto",
            "sequential_image_generation_options": {
                "max_images": self.get_parameter_value("max_images"),
            },
        }

    async def _build_payload(self) -> dict[str, Any]:
        """Build the request payload for Seedream API (without model field)."""
        params = self._get_parameters()
        model = params["model"]

        payload = {
            "model": model,
            "prompt": params["prompt"],
            "size": params["size"],
            "response_format": "url",
            "watermark": params["watermark"],
        }

        await self._add_model_specific_payload_fields(payload, model, params)

        return payload

    async def _build_image_array(self, images: list[Any]) -> list[str]:
        """Build and return a processed image data-URI array."""
        image_array = []
        for img in images:
            image_data = await self._process_input_image(img)
            if image_data:
                image_array.append(image_data)
        return image_array

    async def _add_model_specific_payload_fields(
        self, payload: dict[str, Any], model: str, params: dict[str, Any]
    ) -> None:
        """Add model-dependent fields to Seedream payload."""
        await self._add_multi_image_payload_fields(payload, params)

        match model:
            case "dola-seedream-5-0-pro-260628":
                # Pro generates a single image and errors if the batch fields are present.
                payload["output_format"] = params.get("output_format", "jpeg")
                payload["optimize_prompt_options"] = {"mode": params.get("optimize_prompt_mode", "standard")}
            case "seedream-5-0-260128":
                self._add_batch_payload_fields(payload, params)
                payload["output_format"] = params.get("output_format", "jpeg")
                payload["optimize_prompt_options"] = {"mode": params.get("optimize_prompt_mode", "standard")}
            case "seedream-4-5-251128":
                self._add_batch_payload_fields(payload, params)
            case _:
                msg = f"Unknown Seedream model: {model!r}"
                raise ValueError(msg)

    def _add_batch_payload_fields(self, payload: dict[str, Any], params: dict[str, Any]) -> None:
        """Add the multi-image batch generation fields for models that support batching."""
        payload["sequential_image_generation"] = params["sequential_image_generation"]
        payload["sequential_image_generation_options"] = params["sequential_image_generation_options"]

    async def _add_multi_image_payload_fields(self, payload: dict[str, Any], params: dict[str, Any]) -> None:
        """Add multi-image input field to payload when images are supplied."""
        images = params.get("images", [])
        if not images:
            return

        image_array = await self._build_image_array(images)
        if image_array:
            payload["image"] = image_array

    async def _process_input_image(self, image_input: Any) -> str | None:
        """Process input image and convert to base64 data URI."""
        if not image_input:
            return None

        # Extract string value from input
        image_value = self._extract_image_value(image_input)
        if not image_value:
            return None

        try:
            data_uri = await File(image_value).aread_data_uri(fallback_mime="image/png")
        except FileLoadError:
            logger.debug("%s failed to load image value: %s", self.name, image_value)
            return None
        else:
            return self._coerce_data_uri_to_image_mime(data_uri)

    def _coerce_data_uri_to_image_mime(self, data_uri: str) -> str:
        """Ensure data URIs for input images use an image/* MIME type when detectable.

        Some file sources are read with generic MIME types like text/plain even when
        content is image bytes. This rewrites the data URI header to the detected
        image MIME type while preserving the original base64 payload.
        """
        if not data_uri.startswith("data:") or ";base64," not in data_uri:
            return data_uri

        header, b64_data = data_uri.split(",", 1)
        mime = header[5:].split(";", 1)[0].strip().lower()
        if mime.startswith("image/"):
            return data_uri

        detected_mime = self._detect_image_mime_from_base64(b64_data)
        if not detected_mime:
            return data_uri

        return f"data:{detected_mime};base64,{b64_data}"

    def _detect_image_mime_from_base64(self, b64_data: str) -> str | None:
        """Detect image MIME type from base64-encoded bytes."""
        try:
            decoded = base64.b64decode(b64_data, validate=False)
            image = Image.open(BytesIO(decoded))
            image_format = (image.format or "").lower()
        except Exception:
            return None

        if image_format in ("jpeg", "jpg"):
            return "image/jpeg"
        if image_format:
            return f"image/{image_format}"
        return "image/png"

    def _extract_image_value(self, image_input: Any) -> str | None:
        """Extract string value from various image input types."""
        if isinstance(image_input, str):
            return image_input

        try:
            # ImageUrlArtifact: .value holds URL string
            if hasattr(image_input, "value"):
                value = getattr(image_input, "value", None)
                if isinstance(value, str):
                    return value

            # ImageArtifact: .base64 holds raw or data-URI
            if hasattr(image_input, "base64"):
                b64 = getattr(image_input, "base64", None)
                if isinstance(b64, str) and b64:
                    return b64
        except Exception as e:
            self._log(f"Failed to extract image value: {e}")

        return None

    def _log_request(self, payload: dict[str, Any]) -> None:
        with suppress(Exception):
            sanitized_payload = deepcopy(payload)
            # Truncate long prompts
            prompt = sanitized_payload.get("prompt", "")
            if len(prompt) > PROMPT_TRUNCATE_LENGTH:
                sanitized_payload["prompt"] = prompt[:PROMPT_TRUNCATE_LENGTH] + "..."
            # Redact base64 image data
            if "image" in sanitized_payload:
                image_data = sanitized_payload["image"]
                if isinstance(image_data, list):
                    # Handle array of images
                    redacted_images = []
                    for img in image_data:
                        if isinstance(img, str) and img.startswith("data:image/"):
                            parts = img.split(",", 1)
                            header = parts[0] if parts else "data:image/"
                            b64_len = len(parts[1]) if len(parts) > 1 else 0
                            redacted_images.append(f"{header},<base64 data length={b64_len}>")
                        else:
                            redacted_images.append(img)
                    sanitized_payload["image"] = redacted_images
                elif isinstance(image_data, str) and image_data.startswith("data:image/"):
                    # Handle single image
                    parts = image_data.split(",", 1)
                    header = parts[0] if parts else "data:image/"
                    b64_len = len(parts[1]) if len(parts) > 1 else 0
                    sanitized_payload["image"] = f"{header},<base64 data length={b64_len}>"

            self._log(f"Request payload: {_json.dumps(sanitized_payload, indent=2)}")

    async def _save_single_image_from_url(
        self, image_url: str, generation_id: str | None = None, index: int = 0
    ) -> ImageUrlArtifact | None:
        """Download and save a single image from the provided URL.

        Args:
            image_url: URL of the image to download
            generation_id: Optional generation ID for filename
            index: Index of the image in multi-image response

        Returns:
            ImageUrlArtifact with saved image, or None if download/save fails
        """
        try:
            self._log(f"Downloading image {index} from URL")
            image_bytes = await self._download_bytes_from_url(image_url)

            dest = self._output_file.build_file(_index=index)
            saved = await dest.awrite_bytes(image_bytes)
            self._log(f"Saved image {index} as {saved.name}")
            return ImageUrlArtifact(value=saved.location, name=saved.name)

        except Exception as e:
            # A billed generation whose image cannot be retrieved is a failure, not a
            # silent success. Return None so this image is not counted as saved; the
            # caller reports failure and surfaces the provider URL for manual retrieval.
            self._log(f"Failed to retrieve image {index} from {image_url}: {e}")
            return None

    def _extract_error_message(self, response_json: dict[str, Any]) -> str:
        """Extract error message from failed/errored generation response.

        Tries Seedream-specific error patterns first, then falls back to base implementation.

        Args:
            response_json: The JSON response from the generation status endpoint

        Returns:
            str: A formatted error message to display to the user
        """
        if not response_json:
            return super()._extract_error_message(response_json)

        # Check for v2 API status_detail first (for FAILED/ERROR statuses)
        status_detail = response_json.get("status_detail")
        if status_detail:
            error_msg = self._format_status_detail_error(status_detail)
            if error_msg:
                return f"{self.name} {error_msg}"

        # Try to extract from provider response (legacy pattern)
        parsed_provider_response = self._parse_provider_response(response_json.get("provider_response"))
        if parsed_provider_response:
            provider_error = parsed_provider_response.get("error")
            if provider_error:
                if isinstance(provider_error, dict):
                    error_message = provider_error.get("message", "")
                    details = f"{self.name} {error_message}"
                    if error_code := provider_error.get("code"):
                        details += f"\nError Code: {error_code}"
                    if error_type := provider_error.get("type"):
                        details += f"\nError Type: {error_type}"
                    return details
                return f"{self.name} Provider error: {provider_error}"

        # Fall back to base implementation
        return super()._extract_error_message(response_json)

    def _format_status_detail_error(self, status_detail: dict[str, Any]) -> str | None:
        r"""Format error message from v2 API status_detail field.

        Args:
            status_detail: The status_detail object from a FAILED/ERROR generation response
            Example: {"error": "invalid input", "details": "{\"error\":{\"code\":\"...\",\"message\":\"...\"}}"}

        Returns:
            A formatted error message string, or None if status_detail doesn't contain useful error info
        """
        if not isinstance(status_detail, dict):
            return None

        self._log(f"Parsing status_detail: {status_detail}")

        # Extract top-level error message
        top_error = status_detail.get("error", "")

        # Try to parse the details field (which is a JSON string)
        details_str = status_detail.get("details")
        if details_str and isinstance(details_str, str):
            self._log(f"Found details string, attempting to parse: {details_str[:200]}...")
            try:
                details_obj = _json.loads(details_str)
                self._log(f"Parsed details object: {details_obj}")

                if isinstance(details_obj, dict):
                    error_info = details_obj.get("error", {})
                    if isinstance(error_info, dict):
                        error_code = error_info.get("code", "")
                        error_message = error_info.get("message", "")

                        self._log(f"Extracted error_code={error_code}, error_message length={len(error_message)}")

                        if error_message:
                            # Use the detailed error message as the primary message
                            formatted_msg = error_message
                            if error_code:
                                formatted_msg += f"\nError Code: {error_code}"
                            return formatted_msg
            except Exception as e:
                # If we can't parse details, fall through to simpler format
                self._log(f"Failed to parse status_detail.details JSON: {e}")
        else:
            self._log(f"No details string found or details is not a string: {type(details_str)}")

        # If we have a top-level error but couldn't parse details
        if top_error:
            return f"Generation failed: {top_error}"

        return None

    def _parse_provider_response(self, provider_response: Any) -> dict[str, Any] | None:
        """Parse provider_response if it's a JSON string."""
        if isinstance(provider_response, str):
            try:
                return _json.loads(provider_response)
            except Exception:
                return None
        if isinstance(provider_response, dict):
            return provider_response
        return None

    def _set_safe_defaults(self) -> None:
        self.parameter_output_values["generation_id"] = ""
        self.parameter_output_values["provider_response"] = None

        # Clear all image output parameters (all 15 are created during initialization)
        for i in range(1, 16):
            param_name = "image_url" if i == 1 else f"image_url_{i}"
            self.parameter_output_values[param_name] = None
