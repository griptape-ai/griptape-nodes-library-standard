from __future__ import annotations

import logging
from contextlib import suppress
from typing import Any

from griptape.artifacts import ImageUrlArtifact
from griptape_nodes.exe_types.core_types import ParameterGroup, ParameterMode
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_types.parameter_dict import ParameterDict
from griptape_nodes.exe_types.param_types.parameter_float import ParameterFloat
from griptape_nodes.exe_types.param_types.parameter_image import ParameterImage
from griptape_nodes.exe_types.param_types.parameter_int import ParameterInt
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.files.file import File
from griptape_nodes.traits.options import Options
from griptape_nodes.traits.slider import Slider

from griptape_nodes_library.griptape_proxy_node import GriptapeProxyNode

logger = logging.getLogger("griptape_nodes")

__all__ = ["MuapiImageGeneration"]

# Model mapping from user-friendly names to MuAPI model IDs
MODEL_MAPPING = {
    "Flux Dev": "flux-dev",
    "Flux Schnell": "flux-schnell",
    "Flux Pro": "flux-pro",
    "Flux 2 Pro": "flux-2-pro",
    "Flux 2 Klein 9B Turbo": "flux-2-klein-9b-turbo",
    "Flux 2 Klein 4B Turbo": "flux-2-klein-4b-turbo",
    "Flux Krea Dev": "flux-krea-dev",
    "Midjourney V7": "midjourney-v7-text-to-image",
    "Nano Banana": "nano-banana",
    "HiDream": "hidream",
    "ByteDance Seedream V3": "bytedance-seedream-v3",
    "Qwen 2.0": "qwen-image-2.0",
    "Qwen 2.0 Pro": "qwen-image-2.0-pro",
    "Wan 2.7": "wan2.7-text-to-image",
    "Wan 2.7 Pro": "wan2.7-text-to-image-pro",
}
MODEL_OPTIONS = list(MODEL_MAPPING.keys())
DEFAULT_MODEL = "Flux Dev"

# Image dimension constraints
MIN_IMAGE_DIMENSION = 128
MAX_IMAGE_DIMENSION = 2048
IMAGE_DIMENSION_STEP = 64
DEFAULT_IMAGE_SIZE = 1024

# Inference step constraints
MIN_INFERENCE_STEPS = 1
MAX_INFERENCE_STEPS = 50
DEFAULT_INFERENCE_STEPS = 28

# Guidance scale constraints
MIN_GUIDANCE_SCALE = 1.0
MAX_GUIDANCE_SCALE = 20.0
DEFAULT_GUIDANCE_SCALE = 3.5

# Seed constraints
DEFAULT_SEED = -1
MAX_SEED = 9999999999


class MuapiImageGeneration(GriptapeProxyNode):
    """Generate images using MuAPI models via Griptape model proxy.

    MuAPI is a multi-model aggregator providing access to Flux, Midjourney,
    Nano Banana, HiDream, Seedream, Qwen, and Wan text-to-image models.

    Inputs:
        - model (str): MuAPI model to use for generation
        - prompt (str): Text prompt describing the desired image (2-3000 characters)
        - width (int): Width of the output image in pixels (128-2048, multiples of 64)
        - height (int): Height of the output image in pixels (128-2048, multiples of 64)
        - seed (int): Seed for reproducible results (-1 for random)
        - num_inference_steps (int): Number of denoising steps (1-50)
        - guidance_scale (float): How closely to follow the prompt (1.0-20.0)

    Outputs:
        - generation_id (str): Generation ID from the API
        - provider_response (dict): Verbatim response from Griptape model proxy
        - image_url (ImageUrlArtifact): Generated image as URL artifact
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.category = "API Nodes"
        self.description = "Generate images using MuAPI models via Griptape model proxy"

        # --- INPUT PARAMETERS ---
        self.add_parameter(
            ParameterString(
                name="model",
                default_value=DEFAULT_MODEL,
                tooltip="Select the MuAPI model to use for image generation",
                allow_output=False,
                traits={Options(choices=MODEL_OPTIONS)},
            )
        )

        self.add_parameter(
            ParameterString(
                name="prompt",
                tooltip="Text prompt describing the desired image (2-3000 characters)",
                multiline=True,
                placeholder_text="Describe the image you want to generate...",
                allow_output=False,
                ui_options={"display_name": "Prompt"},
            )
        )

        self.add_parameter(
            ParameterInt(
                name="width",
                default_value=DEFAULT_IMAGE_SIZE,
                tooltip="Width of the output image in pixels. Must be a multiple of 64.",
                allow_output=False,
                min_val=MIN_IMAGE_DIMENSION,
                max_val=MAX_IMAGE_DIMENSION,
                step=IMAGE_DIMENSION_STEP,
            )
        )

        self.add_parameter(
            ParameterInt(
                name="height",
                default_value=DEFAULT_IMAGE_SIZE,
                tooltip="Height of the output image in pixels. Must be a multiple of 64.",
                allow_output=False,
                min_val=MIN_IMAGE_DIMENSION,
                max_val=MAX_IMAGE_DIMENSION,
                step=IMAGE_DIMENSION_STEP,
            )
        )

        with ParameterGroup(name="Generation Settings", ui_options={"collapsed": True}) as gen_settings_group:
            ParameterInt(
                name="seed",
                default_value=DEFAULT_SEED,
                tooltip="Seed for reproducible results (-1 for random)",
                allow_output=False,
                min_val=-1,
                max_val=MAX_SEED,
            )

            ParameterInt(
                name="num_inference_steps",
                default_value=DEFAULT_INFERENCE_STEPS,
                tooltip="Number of denoising steps. Higher values produce more detailed results but take longer.",
                allow_output=False,
                traits={Slider(min_val=MIN_INFERENCE_STEPS, max_val=MAX_INFERENCE_STEPS)},
            )

            ParameterFloat(
                name="guidance_scale",
                default_value=DEFAULT_GUIDANCE_SCALE,
                tooltip="How closely to follow the prompt. Higher values produce results closer to the prompt.",
                allow_output=False,
                traits={Slider(min_val=MIN_GUIDANCE_SCALE, max_val=MAX_GUIDANCE_SCALE)},
            )

        self.add_node_element(gen_settings_group)

        # --- OUTPUT PARAMETERS ---
        self.add_parameter(
            ParameterString(
                name="generation_id",
                tooltip="Generation ID from the API",
                allowed_modes={ParameterMode.OUTPUT},
                hide_property=True,
                hide=True,
            )
        )

        self.add_parameter(
            ParameterDict(
                name="provider_response",
                tooltip="Verbatim response from Griptape model proxy",
                allowed_modes={ParameterMode.OUTPUT},
                hide_property=True,
                hide=True,
            )
        )

        self.add_parameter(
            ParameterImage(
                name="image_url",
                tooltip="Generated image as URL artifact",
                allowed_modes={ParameterMode.OUTPUT, ParameterMode.PROPERTY},
                settable=False,
                pulse_on_run=True,
            )
        )

        self._output_file = ProjectFileParameter(
            node=self,
            name="output_file",
            default_filename="muapi_image.jpg",
        )
        self._output_file.add_parameter()

        # Status parameters must be last
        self._create_status_parameters(
            result_details_tooltip="Details about the image generation result or any errors",
            result_details_placeholder="Generation status will appear here...",
            parameter_group_initially_collapsed=True,
        )

    def _log(self, message: str) -> None:
        with suppress(Exception):
            logger.info(message)

    def _get_api_model_id(self) -> str:
        """Map the user-friendly model name to the MuAPI model ID."""
        model = self.get_parameter_value("model") or DEFAULT_MODEL
        return MODEL_MAPPING.get(str(model), str(model))

    async def _build_payload(self) -> dict[str, Any]:
        """Build the request payload for MuAPI image generation."""
        prompt = self.get_parameter_value("prompt") or ""
        width = self.get_parameter_value("width") or DEFAULT_IMAGE_SIZE
        height = self.get_parameter_value("height") or DEFAULT_IMAGE_SIZE
        seed = self.get_parameter_value("seed")
        num_inference_steps = self.get_parameter_value("num_inference_steps")
        guidance_scale = self.get_parameter_value("guidance_scale")

        payload: dict[str, Any] = {
            "prompt": prompt,
            "width": int(width),
            "height": int(height),
        }

        # Only include optional fields when explicitly set to non-default values
        if seed is not None and int(seed) != DEFAULT_SEED:
            payload["seed"] = int(seed)

        if num_inference_steps is not None:
            payload["num_inference_steps"] = int(num_inference_steps)

        if guidance_scale is not None:
            payload["guidance_scale"] = float(guidance_scale)

        return payload

    async def _parse_result(self, result_json: dict[str, Any], generation_id: str) -> None:
        """Parse the MuAPI result and set output parameters.

        The proxy client's fetch_completed_generation returns the full MuAPI
        poll response, which contains an "outputs" array of CDN URLs.
        """
        outputs = result_json.get("outputs", [])
        if not outputs or not isinstance(outputs, list):
            self._log("No outputs found in result")
            self._set_safe_defaults()
            self._set_status_results(
                was_successful=False,
                result_details="Generation completed but no output URLs were found in the response.",
            )
            return

        # Use the first output URL
        output_url = outputs[0]
        if not output_url or not isinstance(output_url, str):
            self._log("Invalid output URL in result")
            self._set_safe_defaults()
            self._set_status_results(
                was_successful=False,
                result_details="Generation completed but the output URL was invalid.",
            )
            return

        # Download and save the image
        try:
            self._log("Downloading image from URL")
            image_bytes = await File(output_url).aread_bytes()
            if image_bytes:
                dest = self._output_file.build_file()
                saved = await dest.awrite_bytes(image_bytes)
                self.parameter_output_values["image_url"] = ImageUrlArtifact(saved.location)
                self._log(f"Saved image as {saved.name}")
                self._set_status_results(
                    was_successful=True,
                    result_details=f"Image generated successfully and saved as {saved.name}.",
                )
            else:
                self.parameter_output_values["image_url"] = ImageUrlArtifact(value=output_url)
                self._set_status_results(
                    was_successful=True,
                    result_details="Image generated successfully. Using provider URL (could not download image bytes).",
                )
        except Exception as e:
            self._log(f"Failed to save image from URL: {e}")
            self.parameter_output_values["image_url"] = ImageUrlArtifact(value=output_url)
            self._set_status_results(
                was_successful=True,
                result_details=f"Image generated successfully. Using provider URL (could not save to static storage: {e}).",
            )

    def _set_safe_defaults(self) -> None:
        """Set safe default values for outputs."""
        self.parameter_output_values["generation_id"] = ""
        self.parameter_output_values["provider_response"] = None
        self.parameter_output_values["image_url"] = None
