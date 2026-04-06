from __future__ import annotations

import logging
from typing import Any

from griptape.artifacts import ImageUrlArtifact
from griptape_nodes.exe_types.core_types import ParameterGroup, ParameterMode
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_types.parameter_bool import ParameterBool
from griptape_nodes.exe_types.param_types.parameter_dict import ParameterDict
from griptape_nodes.exe_types.param_types.parameter_image import ParameterImage
from griptape_nodes.exe_types.param_types.parameter_int import ParameterInt
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.files.file import File
from griptape_nodes.traits.options import Options

from griptape_nodes_library.griptape_proxy_node import GriptapeProxyNode

logger = logging.getLogger("griptape_nodes")

__all__ = ["WanImageGeneration"]

# Model mapping from user-friendly names to API model IDs
MODEL_MAPPING = {
    "Wan 2.7 Image Pro": "wan2.7-image-pro",
    "Wan 2.7 Image": "wan2.7-image",
}
MODEL_OPTIONS = list(MODEL_MAPPING.keys())
DEFAULT_MODEL = MODEL_OPTIONS[0]

# Size options
SIZE_OPTIONS = ["1K", "2K", "4K"]
DEFAULT_SIZE = "2K"

# Seed range
MAX_SEED = 2147483647


class WanImageGeneration(GriptapeProxyNode):
    """Generate images using Wan 2.7 models via Griptape model proxy.

    Documentation: https://www.alibabacloud.com/help/en/model-studio/wan-image-generation-and-editing-api-reference

    Inputs:
        - model (str): Wan model to use (default: "Wan 2.7 Image Pro")
        - prompt (str): Text description of the image to generate (max 5000 chars)
        - size (str): Output image resolution shortcut (default: "2K")
        - n (int): Number of images to generate (default: 1, range 1-4)
        - thinking_mode (bool): Enable thinking mode for better quality (default: True)
        - watermark (bool): Add AI-generated watermark (default: False)
        - seed (int): Random seed for reproducibility (default: None)

    Outputs:
        - generation_id (str): Generation ID from the API
        - provider_response (dict): Verbatim provider response from the model proxy
        - image_url_1 through image_url_4 (ImageUrlArtifact): Generated images (shown based on actual count)
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.category = "API Nodes"
        self.description = "Generate images using Wan 2.7 models via Griptape model proxy"

        # Model selection
        self.add_parameter(
            ParameterString(
                name="model",
                default_value=DEFAULT_MODEL,
                tooltip="Select the Wan model to use",
                allow_output=False,
                traits={Options(choices=MODEL_OPTIONS)},
            )
        )

        # Core parameters
        self.add_parameter(
            ParameterString(
                name="prompt",
                tooltip="Text description of the image to generate (max 5000 characters)",
                multiline=True,
                placeholder_text="Describe the image you want to generate...",
                allow_output=False,
                ui_options={
                    "display_name": "Prompt",
                },
            )
        )

        # Size parameter
        self.add_parameter(
            ParameterString(
                name="size",
                default_value=DEFAULT_SIZE,
                tooltip="Output image resolution. 1K=1024x1024, 2K=2048x2048, 4K=4096x4096 (4K only for Pro model, text-to-image only)",
                allow_output=False,
                traits={Options(choices=SIZE_OPTIONS)},
            )
        )

        # Number of images
        self.add_parameter(
            ParameterInt(
                name="n",
                default_value=1,
                tooltip="Number of images to generate (1-4)",
                allow_output=False,
                min_val=1,
                max_val=4,
            )
        )

        with ParameterGroup(name="Generation Settings", ui_options={"collapsed": True}) as generation_settings_group:
            ParameterBool(
                name="thinking_mode",
                default_value=True,
                tooltip="Enable thinking mode for better quality (adds latency). Only works without sequential mode and without image input.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )

            ParameterBool(
                name="watermark",
                default_value=False,
                tooltip="Add 'AI generated' watermark to bottom-right corner",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )

            ParameterInt(
                name="seed",
                default_value=None,
                tooltip="Random seed for reproducibility (leave empty for random). Range: 0-2147483647",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                min_val=0,
                max_val=MAX_SEED,
            )

        self.add_node_element(generation_settings_group)

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

        # Create all image output parameters upfront (1-4)
        # Only the first is visible initially; others are shown when API returns multiple images
        for i in range(1, 5):
            self.add_parameter(
                ParameterImage(
                    name=f"image_url_{i}",
                    tooltip=f"Generated image {i}",
                    allowed_modes={ParameterMode.OUTPUT, ParameterMode.PROPERTY},
                    settable=False,
                    ui_options={"is_full_width": True, "pulse_on_run": True, "hide": i > 1},
                )
            )

        self._output_file = ProjectFileParameter(
            node=self,
            name="output_file",
            default_filename="wan_image.png",
        )
        self._output_file.add_parameter()

        # Create status parameters for success/failure tracking (at the end)
        self._create_status_parameters(
            result_details_tooltip="Details about the image generation result or any errors",
            result_details_placeholder="Generation status and details will appear here.",
            parameter_group_initially_collapsed=True,
        )

    def _get_api_model_id(self) -> str:
        """Map friendly model name to API model ID."""
        model = self.get_parameter_value("model") or DEFAULT_MODEL
        return MODEL_MAPPING.get(str(model), str(model))

    async def _build_payload(self) -> dict[str, Any]:
        """Build the request payload for Wan image generation.

        The payload uses the DashScope messages format with input/parameters structure.
        The model field is not included here as the base class handles routing.
        """
        prompt = self.get_parameter_value("prompt") or ""
        size = self.get_parameter_value("size") or DEFAULT_SIZE
        n = self.get_parameter_value("n") or 1
        thinking_mode = self.get_parameter_value("thinking_mode")
        watermark = self.get_parameter_value("watermark") or False
        seed = self.get_parameter_value("seed")

        # Default thinking_mode to True if not explicitly set
        if thinking_mode is None:
            thinking_mode = True

        # Build the messages structure
        content = [{"text": prompt}]

        payload: dict[str, Any] = {
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": content,
                    }
                ]
            },
            "parameters": {
                "size": size,
                "n": n,
                "thinking_mode": thinking_mode,
                "watermark": watermark,
            },
        }

        # Only include seed if explicitly set
        if seed is not None:
            payload["parameters"]["seed"] = seed

        return payload

    def _show_image_output_parameters(self, count: int) -> None:
        """Show image output parameters based on actual result count.

        All 4 image parameters are created during initialization but hidden except image_url.
        This method shows the appropriate number based on the API response.

        Args:
            count: Total number of images returned from API (1-4)
        """
        for i in range(1, 5):
            param_name = f"image_url_{i}"
            if i <= count:
                self.show_parameter_by_name(param_name)
            else:
                self.hide_parameter_by_name(param_name)

    async def _parse_result(self, result_json: dict[str, Any], generation_id: str) -> None:
        """Parse the Wan image generation result and save images.

        The proxy client returns the full DashScope response. Image URLs are at:
        output.choices[*].message.content[*].image
        """
        # Extract image URLs from all choices
        output = result_json.get("output", {})
        choices = output.get("choices", [])

        if not choices:
            self._set_safe_defaults()
            self._set_status_results(
                was_successful=False,
                result_details="Generation completed but no choices found in the response.",
            )
            return

        # Collect image URLs from all choices
        image_urls: list[str] = []
        for choice in choices:
            message = choice.get("message", {})
            content_items = message.get("content", [])
            for item in content_items:
                if isinstance(item, dict) and item.get("image"):
                    image_urls.append(item["image"])

        if not image_urls:
            self._set_safe_defaults()
            self._set_status_results(
                was_successful=False,
                result_details="Generation completed but no image URLs were found in the response.",
            )
            return

        # Download and save all images
        image_artifacts: list[ImageUrlArtifact] = []
        for index, url in enumerate(image_urls):
            artifact = await self._save_single_image_from_url(url, index)
            if artifact:
                image_artifacts.append(artifact)

        if not image_artifacts:
            self._set_safe_defaults()
            self._set_status_results(
                was_successful=False,
                result_details="Generation completed but no images could be saved.",
            )
            return

        # Show the appropriate number of image output parameters
        self._show_image_output_parameters(len(image_artifacts))

        # Set individual image output parameters
        for idx, artifact in enumerate(image_artifacts, start=1):
            param_name = f"image_url_{idx}"
            self.parameter_output_values[param_name] = artifact

        # Set success status
        count = len(image_artifacts)
        filenames = [artifact.name for artifact in image_artifacts]
        if count == 1:
            details = f"Image generated successfully and saved as {filenames[0]}."
        else:
            details = f"Generated {count} images successfully: {', '.join(filenames)}."
        self._set_status_results(was_successful=True, result_details=details)

    async def _save_single_image_from_url(self, image_url: str, index: int = 0) -> ImageUrlArtifact | None:
        """Download and save a single image from the provided URL.

        Args:
            image_url: URL of the image to download
            index: Index of the image in multi-image response

        Returns:
            ImageUrlArtifact with saved image, or None if download/save fails
        """
        try:
            logger.info("Downloading image %d from URL", index)
            image_bytes = await File(image_url).aread_bytes()
            if not image_bytes:
                logger.warning("Could not download image %d, using provider URL", index)
                return ImageUrlArtifact(value=image_url)

            dest = self._output_file.build_file(_index=index)
            saved = await dest.awrite_bytes(image_bytes)
            logger.info("Saved image %d as %s", index, saved.name)
            return ImageUrlArtifact(value=saved.location, name=saved.name)
        except Exception as e:
            logger.error("Failed to save image %d from URL: %s", index, e)
            return ImageUrlArtifact(value=image_url)

    def _set_safe_defaults(self) -> None:
        """Set safe default values for outputs."""
        self.parameter_output_values["generation_id"] = ""
        self.parameter_output_values["provider_response"] = None

        # Clear all image output parameters
        for i in range(1, 5):
            param_name = f"image_url_{i}"
            self.parameter_output_values[param_name] = None

    def _extract_error_message(self, response_json: dict[str, Any]) -> str:
        """Extract error message from DashScope error responses.

        DashScope errors may include a 'code' and 'message' field at the top level,
        or nested within the response structure.
        """
        if not response_json:
            return f"{self.name} generation failed with no error details provided by API."

        # Try DashScope-specific error format: {"code": "...", "message": "..."}
        code = response_json.get("code")
        message = response_json.get("message")
        if code and message:
            return f"{self.name}: {code}: {message}"

        # Fall back to the base class error extraction
        return super()._extract_error_message(response_json)
