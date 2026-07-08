from __future__ import annotations

import asyncio
import base64
import logging
from enum import StrEnum
from typing import Any

from griptape.artifacts.video_url_artifact import VideoUrlArtifact
from griptape_nodes.exe_types.core_types import ParameterGroup, ParameterMode
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_types.parameter_dict import ParameterDict
from griptape_nodes.exe_types.param_types.parameter_image import ParameterImage
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.exe_types.param_types.parameter_video import ParameterVideo
from griptape_nodes.files.file import File, FileLoadError
from griptape_nodes.traits.options import Options

from griptape_nodes_library.proxy import GriptapeProxyNode

logger = logging.getLogger("griptape_nodes")

__all__ = ["GeminiOmniFlashGeneration"]

# Provider model id served by the Gemini API interactions endpoint via proxy.
API_MODEL_ID = "gemini-omni-flash-preview"


class Task(StrEnum):
    """Generation task inferred from which inputs are provided."""

    TEXT_TO_VIDEO = "text_to_video"
    IMAGE_TO_VIDEO = "image_to_video"


class GeminiOmniFlashGeneration(GriptapeProxyNode):
    """Generate a video using Google's Gemini Omni Flash model via Griptape Cloud model proxy.

    Gemini Omni Flash turns text and images into short, 720p video with audio. It
    is served synchronously by the Gemini API interactions endpoint; the proxy
    bridges that into the standard async generation flow this node uses.

    Inputs:
        - prompt (str): Text prompt for the video
        - image (ImageArtifact|ImageUrlArtifact|str): Optional reference/start image
          (when provided, the task is image_to_video)
        - aspect_ratio (str): Output aspect ratio (default: 16:9, options: 16:9, 9:16)

    Outputs:
        - generation_id (str): Griptape Cloud generation id
        - provider_response (dict): Verbatim response from the API
        - video_url (VideoUrlArtifact): Saved static video URL
        - was_successful (bool): Whether the generation succeeded
        - result_details (str): Details about the generation result or error
    """

    SERVICE_NAME = "Griptape"
    API_KEY_NAME = "GT_CLOUD_API_KEY"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # INPUTS / PROPERTIES
        self.add_parameter(
            ParameterString(
                name="prompt",
                tooltip="Text prompt for the video",
                multiline=True,
                placeholder_text="Describe the video...",
                allow_output=False,
                ui_options={"display_name": "prompt"},
            )
        )

        self.add_parameter(
            ParameterImage(
                name="image",
                default_value=None,
                tooltip="Optional reference image; when provided the task is image_to_video",
                allowed_modes={ParameterMode.INPUT},
                hide_property=True,
                ui_options={"display_name": "image"},
            )
        )

        with ParameterGroup(name="Generation Settings") as generation_settings_group:
            # The model chooses clip length itself (3-10s); duration is not a
            # request parameter, so aspect ratio is the only output control.
            ParameterString(
                name="aspect_ratio",
                default_value="16:9",
                tooltip="Output aspect ratio",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=["16:9", "9:16"])},
            )

        self.add_node_element(generation_settings_group)

        # OUTPUTS
        self.add_parameter(
            ParameterDict(
                name="provider_response",
                tooltip="Verbatim response from API",
                allowed_modes={ParameterMode.OUTPUT},
                hide_property=True,
                hide=True,
            )
        )

        self.add_parameter(
            ParameterVideo(
                name="video_url",
                tooltip="Saved video as URL artifact for downstream display",
                allowed_modes={ParameterMode.OUTPUT, ParameterMode.PROPERTY},
                settable=False,
                ui_options={"pulse_on_run": True},
            )
        )

        self._output_file = ProjectFileParameter(
            node=self,
            name="output_file",
            default_filename="gemini_omni_flash_video.mp4",
        )
        self._output_file.add_parameter()

        # Create status parameters for success/failure tracking
        self._create_status_parameters(
            result_details_tooltip="Details about the video generation result or any errors",
            result_details_placeholder="Generation status and details will appear here.",
            parameter_group_initially_collapsed=True,
        )

    def validate_before_node_run(self) -> list[Exception] | None:
        exceptions = super().validate_before_node_run() or []
        if not self.get_parameter_value("prompt"):
            exceptions.append(ValueError(f"{self.name} prompt must be provided"))
        return exceptions or None

    def _get_api_model_id(self) -> str:
        # Bare provider id; the base class builds POST /api/proxy/v2/models/{id}.
        return API_MODEL_ID

    async def _image_to_base64(self, image_input: Any) -> tuple[str, str] | None:
        """Load an image input and return (mime_type, base64_data), or None."""
        if not image_input:
            return None

        image_value = image_input
        if not isinstance(image_input, str):
            image_value = getattr(image_input, "value", None) or getattr(image_input, "base64", None)
        if not isinstance(image_value, str) or not image_value:
            return None

        try:
            data_uri = await File(image_value).aread_data_uri(fallback_mime="image/png")
        except FileLoadError:
            logger.debug("%s failed to load image value: %s", self.name, image_value)
            return None

        parts = data_uri.split(",", 1)
        if len(parts) != 2:  # noqa: PLR2004
            return None
        header, base64_data = parts
        mime_type = "image/png"
        if ";" in header:
            mime_part = header.split(";")[0].replace("data:", "")
            if mime_part.startswith("image/"):
                mime_type = mime_part
        return mime_type, base64_data

    async def _build_payload(self) -> dict[str, Any]:
        prompt = self.get_parameter_value("prompt") or ""
        aspect_ratio = self.get_parameter_value("aspect_ratio") or "16:9"
        image_input = self.get_parameter_value("image")

        image_b64 = await self._image_to_base64(image_input)

        # The interactions `input` is either a plain prompt string (text_to_video)
        # or a list of content objects mixing text and image (image_to_video).
        model_input: Any
        if image_b64:
            task = Task.IMAGE_TO_VIDEO
            mime_type, base64_data = image_b64
            # Interactions `input` items are flat and type-discriminated:
            # {"type": "image", "data": ..., "mime_type": ...} / {"type": "text", "text": ...}
            model_input = [
                {"type": "image", "data": base64_data, "mime_type": mime_type},
                {"type": "text", "text": prompt},
            ]
        else:
            task = Task.TEXT_TO_VIDEO
            model_input = prompt

        # `delivery` is omitted so the API returns the default inline base64 in
        # the response `data` field (the "uri" mode is for >4MB results and would
        # need an authenticated download; the proxy's /result also returns inline
        # base64 regardless).
        return {
            "input": model_input,
            "response_format": {
                "type": "video",
                "aspect_ratio": aspect_ratio,
            },
            "generation_config": {"video_config": {"task": task.value}},
        }

    def _extract_video_from_steps(self, result_json: dict[str, Any]) -> dict[str, Any] | None:
        """Return the first video content item from the interactions `steps` array."""
        for step in result_json.get("steps", []) or []:
            if not isinstance(step, dict):
                continue
            for item in step.get("content", []) or []:
                if isinstance(item, dict) and item.get("type") == "video":
                    return item
        return None

    async def _parse_result(self, result_json: dict[str, Any], _generation_id: str) -> None:
        video_item = self._extract_video_from_steps(result_json)
        if not video_item:
            logger.warning("%s: No video in result", self.name)
            self._set_safe_defaults()
            self._set_status_results(
                was_successful=False,
                result_details="Generation completed but no video was received",
            )
            return

        base64_data = video_item.get("data")
        if not base64_data:
            logger.warning("%s: Video content missing base64 data", self.name)
            self._set_safe_defaults()
            self._set_status_results(
                was_successful=False,
                result_details="Generation completed but the video contained no data",
            )
            return

        video_bytes = await asyncio.to_thread(base64.b64decode, base64_data)
        dest = self._output_file.build_file()
        saved = await dest.awrite_bytes(video_bytes)
        logger.info("%s: Saved video as %s (%s bytes)", self.name, saved.name, len(video_bytes))

        self.parameter_output_values["video_url"] = VideoUrlArtifact(value=saved.location, name=saved.name)
        self._set_status_results(was_successful=True, result_details="Generated 1 video successfully")

    def _set_safe_defaults(self) -> None:
        """Set safe default values for all outputs on error."""
        self.parameter_output_values["generation_id"] = ""
        self.parameter_output_values["provider_response"] = None
        self.parameter_output_values["video_url"] = None
