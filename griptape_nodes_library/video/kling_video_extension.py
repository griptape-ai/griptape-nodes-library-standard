from __future__ import annotations

import logging
from typing import Any

from griptape.artifacts.video_url_artifact import VideoUrlArtifact
from griptape_nodes.exe_types.core_types import ParameterGroup, ParameterMode
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_types.parameter_dict import ParameterDict
from griptape_nodes.exe_types.param_types.parameter_float import ParameterFloat
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.exe_types.param_types.parameter_video import ParameterVideo

from griptape_nodes_library.proxy import ArtifactKind, GriptapeProxyNode

logger = logging.getLogger("griptape_nodes")

__all__ = ["KlingVideoExtension"]

# Constants
MAX_PROMPT_LENGTH = 2500


class KlingVideoExtension(GriptapeProxyNode):
    """Extend an existing video using Kling AI via Griptape Cloud model proxy.

    Extends videos by 4-5 seconds. Maximum total video length: 3 minutes.

    Inputs:
        - video_id (str): Video ID from previous Kling AI generation (required)
        - prompt (str): Text prompt for video extension (max 2500 chars)
        - negative_prompt (str): Negative text prompt (max 2500 chars)
        - cfg_scale (float): Flexibility in video generation (0-1)
        (Always polls for result: 5s interval, 20 min timeout)

    Outputs:
        - generation_id (str): Griptape Cloud generation id
        - provider_response (dict): Verbatim response from API (latest polling response)
        - video_url (VideoUrlArtifact): Saved extended video URL
        - kling_video_id (str): The Kling AI video ID
        - was_successful (bool): Whether the generation succeeded
        - result_details (str): Details about the generation result or error
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.add_parameter(
            ParameterString(
                name="video_id",
                tooltip="Video ID from previous Kling AI video generation (required)",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                placeholder_text="Enter video ID from previous Kling generation...",
            )
        )
        self.add_parameter(
            ParameterString(
                name="prompt",
                default_value="",
                tooltip="Text prompt for video extension (max 2500 chars)",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                multiline=True,
                placeholder_text="Describe how to continue the video...",
            )
        )
        self.add_parameter(
            ParameterString(
                name="negative_prompt",
                default_value="",
                tooltip="Negative text prompt (max 2500 chars)",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                multiline=True,
                placeholder_text="Describe what you don't want...",
            )
        )
        # Extension Settings Group
        with ParameterGroup(name="Extension Settings") as extension_group:
            ParameterFloat(
                name="cfg_scale",
                default_value=0.5,
                tooltip="Flexibility (0-1). Higher value = lower flexibility, stronger prompt relevance.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        self.add_node_element(extension_group)

        # OUTPUTS
        self.add_parameter(
            ParameterDict(
                name="provider_response",
                tooltip="Verbatim response from API (latest polling response)",
                allowed_modes={ParameterMode.OUTPUT},
                hide_property=True,
            )
        )

        self.add_parameter(
            ParameterVideo(
                name="video_url",
                tooltip="Saved extended video as URL artifact",
                allowed_modes={ParameterMode.OUTPUT, ParameterMode.PROPERTY},
                settable=False,
                ui_options={"pulse_on_run": True},
            )
        )

        self.add_parameter(
            ParameterString(
                name="kling_video_id",
                tooltip="The Kling AI video ID",
                allowed_modes={ParameterMode.OUTPUT},
                placeholder_text="The Kling AI video ID",
            )
        )

        self._output_file = ProjectFileParameter(
            node=self,
            name="output_file",
            default_filename="kling_video_extension.mp4",
        )
        self._output_file.add_parameter()

        # Create status parameters for success/failure tracking
        self._create_status_parameters(
            result_details_tooltip="Details about the video extension result or any errors",
            result_details_placeholder="Extension status and details will appear here.",
            parameter_group_initially_collapsed=True,
        )

    def _get_api_model_id(self) -> str:
        """Get the API model ID for this generation.

        Returns the static model ID for Kling Video Extension.
        """
        return "kling:video-extend"

    async def _build_payload(self) -> dict[str, Any]:
        """Build the request payload for Kling video extension API.

        Returns:
            dict: The request payload (model field excluded, handled by base class)
        """
        video_id = (self.get_parameter_value("video_id") or "").strip()
        prompt = self.get_parameter_value("prompt") or ""
        negative_prompt = self.get_parameter_value("negative_prompt") or ""
        cfg_scale = self.get_parameter_value("cfg_scale") or 0.5

        payload: dict[str, Any] = {
            "video_id": video_id,
            "cfg_scale": float(cfg_scale),
        }

        # Add prompts if provided
        if prompt:
            payload["prompt"] = prompt.strip()

        if negative_prompt:
            payload["negative_prompt"] = negative_prompt.strip()

        return payload

    async def _parse_result(self, result_json: dict[str, Any], generation_id: str) -> None:
        """Save the hosted video and record Kling's own id for it.

        Kling reports the id under {"data": {"task_result": {"videos": [{"id": "..."}]}}}.
        """
        videos = result_json.get("data", {}).get("task_result", {}).get("videos", [])
        video_id = videos[0].get("id") if videos and isinstance(videos[0], dict) else None
        if video_id:
            self.parameter_output_values["kling_video_id"] = video_id
            logger.info("Video ID: %s", video_id)

        await self._save_generated_media(
            generation_id,
            "video_url",
            lambda v, n: VideoUrlArtifact(value=v, name=n),
            kind=ArtifactKind.VIDEO,
            action="extended",
        )

    def _set_safe_defaults(self) -> None:
        """Clear output parameters on error."""
        self.parameter_output_values["video_url"] = None
        self.parameter_output_values["kling_video_id"] = ""

    def validate_before_node_run(self) -> list[Exception] | None:
        """Validate parameters before execution."""
        exceptions = super().validate_before_node_run() or []

        # Get parameter values
        video_id = (self.get_parameter_value("video_id") or "").strip()
        prompt = self.get_parameter_value("prompt") or ""
        negative_prompt = self.get_parameter_value("negative_prompt") or ""
        cfg_scale = self.get_parameter_value("cfg_scale") or 0.5

        # Validate video_id is provided
        if not video_id:
            exceptions.append(ValueError(f"{self.name} requires a video_id from a previous Kling AI generation."))

        # Validate prompt length
        if prompt and len(prompt) > MAX_PROMPT_LENGTH:
            exceptions.append(
                ValueError(
                    f"{self.name} prompt exceeds {MAX_PROMPT_LENGTH} characters (got: {len(prompt)} characters)."
                )
            )

        # Validate negative prompt length
        if negative_prompt and len(negative_prompt) > MAX_PROMPT_LENGTH:
            exceptions.append(
                ValueError(
                    f"{self.name} negative_prompt exceeds {MAX_PROMPT_LENGTH} characters (got: {len(negative_prompt)} characters)."
                )
            )

        # Validate cfg_scale
        if not (0 <= cfg_scale <= 1):
            exceptions.append(ValueError(f"{self.name} cfg_scale must be between 0.0 and 1.0."))

        return exceptions if exceptions else None
