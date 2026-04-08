from __future__ import annotations

import logging
from contextlib import suppress
from typing import Any

from griptape.artifacts import ImageUrlArtifact
from griptape_nodes.exe_types.core_types import ParameterGroup, ParameterMode
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_types.parameter_dict import ParameterDict
from griptape_nodes.exe_types.param_types.parameter_image import ParameterImage
from griptape_nodes.exe_types.param_types.parameter_int import ParameterInt
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.files.file import File
from griptape_nodes.traits.options import Options

from griptape_nodes_library.griptape_proxy_node import GriptapeProxyNode

logger = logging.getLogger("griptape_nodes")

__all__ = ["RecraftImageGeneration"]

# Model mapping from user-friendly names to API model IDs
MODEL_MAPPING = {
    "Recraft V4": "recraftv4",
    "Recraft V4 Pro": "recraftv4_pro",
    "Recraft V4 Vector": "recraftv4_vector",
    "Recraft V4 Pro Vector": "recraftv4_pro_vector",
    "Recraft V3": "recraftv3",
    "Recraft V3 Vector": "recraftv3_vector",
    "Recraft V2": "recraftv2",
    "Recraft V2 Vector": "recraftv2_vector",
}
MODEL_OPTIONS = list(MODEL_MAPPING.keys())
DEFAULT_MODEL = "Recraft V4"

# V2/V3 style presets
STYLE_OPTIONS = [
    "",
    "any",
    "realistic_image",
    "digital_illustration",
    "vector_illustration",
    "icon",
]

# Aspect ratio options and their pixel mappings per model family
ASPECT_RATIO_OPTIONS = [
    "1:1",
    "2:1",
    "1:2",
    "3:2",
    "2:3",
    "4:3",
    "3:4",
    "5:4",
    "4:5",
    "6:10",
    "14:10",
    "10:14",
    "16:9",
    "9:16",
]

SIZE_MAP_V4 = {
    "1:1": "1024x1024",
    "2:1": "1536x768",
    "1:2": "768x1536",
    "3:2": "1280x832",
    "2:3": "832x1280",
    "4:3": "1216x896",
    "3:4": "896x1216",
    "5:4": "1152x896",
    "4:5": "896x1152",
    "6:10": "832x1344",
    "14:10": "1280x896",
    "10:14": "896x1280",
    "16:9": "1344x768",
    "9:16": "768x1344",
}

SIZE_MAP_V4_PRO = {
    "1:1": "2048x2048",
    "2:1": "3072x1536",
    "1:2": "1536x3072",
    "3:2": "2560x1664",
    "2:3": "1664x2560",
    "4:3": "2432x1792",
    "3:4": "1792x2432",
    "5:4": "2304x1792",
    "4:5": "1792x2304",
    "6:10": "1664x2688",
    "14:10": "2560x1792",
    "10:14": "1792x2560",
    "16:9": "2688x1536",
    "9:16": "1536x2688",
}

SIZE_MAP_V2_V3 = {
    "1:1": "1024x1024",
    "2:1": "2048x1024",
    "1:2": "1024x2048",
    "3:2": "1536x1024",
    "2:3": "1024x1536",
    "4:3": "1365x1024",
    "3:4": "1024x1365",
    "5:4": "1280x1024",
    "4:5": "1024x1280",
    "6:10": "1024x1707",
    "14:10": "1434x1024",
    "10:14": "1024x1434",
    "16:9": "1820x1024",
    "9:16": "1024x1820",
}

# Models that support V2/V3-only features (style, negative_prompt)
V2_V3_MODELS = {"recraftv3", "recraftv3_vector", "recraftv2", "recraftv2_vector"}

# Models that use the V4 Pro size map
V4_PRO_MODELS = {"recraftv4_pro", "recraftv4_pro_vector"}

# Vector models use aspect ratios instead of pixel dimensions
VECTOR_MODELS = {"recraftv4_vector", "recraftv4_pro_vector", "recraftv3_vector", "recraftv2_vector"}


class RecraftImageGeneration(GriptapeProxyNode):
    """Generate images using Recraft models via Griptape model proxy.

    Supports raster and vector generation across Recraft V2, V3, and V4 model families.

    Inputs:
        - model (str): Recraft model to use (default: "Recraft V4")
        - prompt (str): Text description of the image to generate
        - aspect_ratio (str): Aspect ratio for the output image
        - n (int): Number of images to generate (1-6)
        - style (str): Style preset (V2/V3 models only)
        - negative_prompt (str): Elements to exclude (V2/V3 models only)

    Outputs:
        - generation_id (str): Generation ID from the API
        - provider_response (dict): Verbatim provider response
        - output_image (ImageUrlArtifact): First generated image
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.category = "API Nodes"
        self.description = "Generate images using Recraft models via Griptape model proxy"

        # Model selection
        self.add_parameter(
            ParameterString(
                name="model",
                default_value=DEFAULT_MODEL,
                tooltip="Select the Recraft model to use",
                allow_output=False,
                traits={Options(choices=MODEL_OPTIONS)},
            )
        )

        # Prompt
        self.add_parameter(
            ParameterString(
                name="prompt",
                tooltip="Text description of the image to generate",
                multiline=True,
                placeholder_text="Describe the image you want to generate...",
                allow_output=False,
            )
        )

        # Aspect ratio
        self.add_parameter(
            ParameterString(
                name="aspect_ratio",
                default_value="1:1",
                tooltip="Aspect ratio for the output image. Pixel dimensions are determined by the model.",
                allow_output=False,
                traits={Options(choices=ASPECT_RATIO_OPTIONS)},
            )
        )

        # Number of images
        self.add_parameter(
            ParameterInt(
                name="n",
                default_value=1,
                tooltip="Number of images to generate (1-6)",
                allow_output=False,
                min_val=1,
                max_val=6,
            )
        )

        # V2/V3 advanced settings
        with ParameterGroup(name="V2/V3 Settings", ui_options={"collapsed": True}) as v2_v3_group:
            ParameterString(
                name="style",
                default_value="",
                tooltip="Style preset for generation (V2/V3 models only, ignored for V4)",
                allow_output=False,
                traits={Options(choices=STYLE_OPTIONS)},
            )

            ParameterString(
                name="negative_prompt",
                default_value="",
                tooltip="Elements to exclude from generation (V2/V3 models only, ignored for V4)",
                multiline=True,
                placeholder_text="Elements to avoid in the image...",
                allow_output=False,
            )

        self.add_node_element(v2_v3_group)

        # OUTPUTS
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
                name="output_image",
                tooltip="Generated image",
                allowed_modes={ParameterMode.OUTPUT, ParameterMode.PROPERTY},
                settable=False,
                ui_options={"pulse_on_run": True},
            )
        )

        self._output_file = ProjectFileParameter(
            node=self,
            name="output_file",
            default_filename="recraft_image.png",
        )
        self._output_file.add_parameter()

        # Status parameters (must be last)
        self._create_status_parameters(
            result_details_tooltip="Details about the image generation result or any errors",
            result_details_placeholder="Generation status will appear here...",
            parameter_group_initially_collapsed=True,
        )

    def _get_api_model_id(self) -> str:
        """Map friendly model name to API model ID."""
        model = self.get_parameter_value("model") or DEFAULT_MODEL
        return MODEL_MAPPING.get(str(model), str(model))

    def _resolve_size(self, aspect_ratio: str, api_model_id: str) -> str:
        """Resolve aspect ratio to the correct size string for the given model.

        Vector models use aspect ratio strings directly. Raster models use
        pixel dimensions that vary by model family.
        """
        if api_model_id in VECTOR_MODELS:
            return aspect_ratio

        if api_model_id in V4_PRO_MODELS:
            size_map = SIZE_MAP_V4_PRO
        elif api_model_id in V2_V3_MODELS:
            size_map = SIZE_MAP_V2_V3
        else:
            size_map = SIZE_MAP_V4

        return size_map.get(aspect_ratio, size_map.get("1:1", "1024x1024"))

    async def _build_payload(self) -> dict[str, Any]:
        """Build the request payload for Recraft image generation."""
        prompt = self.get_parameter_value("prompt") or ""
        aspect_ratio = self.get_parameter_value("aspect_ratio") or "1:1"
        n = self.get_parameter_value("n") or 1

        api_model_id = self._get_api_model_id()
        size = self._resolve_size(aspect_ratio, api_model_id)

        payload: dict[str, Any] = {
            "prompt": prompt,
            "model": api_model_id,
            "n": n,
            "size": size,
            "response_format": "url",
        }

        # Add V2/V3 specific parameters only when applicable
        if api_model_id in V2_V3_MODELS:
            style = self.get_parameter_value("style") or ""
            if style:
                payload["style"] = style

            negative_prompt = self.get_parameter_value("negative_prompt") or ""
            if negative_prompt:
                payload["negative_prompt"] = negative_prompt

        return payload

    async def _parse_result(self, result_json: dict[str, Any], generation_id: str) -> None:
        """Parse the Recraft result and set output parameters.

        The proxy returns the raw Recraft API response:
        {"created": ..., "credits": ..., "data": [{"image_id": "...", "url": "..."}]}
        """
        data = result_json.get("data")
        if not data or not isinstance(data, list) or len(data) == 0:
            self._set_safe_defaults()
            self._set_status_results(
                was_successful=False,
                result_details="Generation completed but no image data was found in the response.",
            )
            return

        first_image = data[0]
        url = first_image.get("url")
        if not url:
            self._set_safe_defaults()
            self._set_status_results(
                was_successful=False,
                result_details="Generation completed but no image URL was found in the response.",
            )
            return

        try:
            self._log("Downloading image from URL")
            image_bytes = await File(url).aread_bytes()
            if image_bytes:
                dest = self._output_file.build_file()
                saved = await dest.awrite_bytes(image_bytes)
                self.parameter_output_values["output_image"] = ImageUrlArtifact(saved.location)
                self._log(f"Saved image as {saved.name}")

                count = len(data)
                details = f"Generated {count} image{'s' if count > 1 else ''} successfully."
                if count > 1:
                    details += " Only the first image is shown in the output."
                self._set_status_results(was_successful=True, result_details=details)
            else:
                self.parameter_output_values["output_image"] = ImageUrlArtifact(value=url)
                self._set_status_results(
                    was_successful=True,
                    result_details="Image generated successfully. Using provider URL (could not download image bytes).",
                )
        except Exception as e:
            self._log(f"Failed to save image from URL: {e}")
            self.parameter_output_values["output_image"] = ImageUrlArtifact(value=url)
            self._set_status_results(
                was_successful=True,
                result_details=f"Image generated successfully. Using provider URL (could not save to static storage: {e}).",
            )

    def _set_safe_defaults(self) -> None:
        """Clear all output parameters on error."""
        self.parameter_output_values["generation_id"] = ""
        self.parameter_output_values["provider_response"] = None
        self.parameter_output_values["output_image"] = None

    def _extract_error_message(self, response_json: dict[str, Any]) -> str:
        """Extract error message from Recraft error response.

        Recraft errors use {"code": "...", "message": "..."} format.
        """
        if not response_json:
            return super()._extract_error_message(response_json)

        # Recraft-specific error format
        message = response_json.get("message")
        if message:
            code = response_json.get("code", "")
            if code:
                return f"{self.name}: {message} (code: {code})"
            return f"{self.name}: {message}"

        return super()._extract_error_message(response_json)

    def _log(self, message: str) -> None:
        with suppress(Exception):
            logger.info(message)
