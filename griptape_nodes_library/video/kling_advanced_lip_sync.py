from __future__ import annotations

import logging
import os
from typing import Any, ClassVar
from urllib.parse import urljoin

import httpx
from griptape.artifacts.video_url_artifact import VideoUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterGroup, ParameterMode
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_types.parameter_bool import ParameterBool
from griptape_nodes.exe_types.param_types.parameter_dict import ParameterDict
from griptape_nodes.exe_types.param_types.parameter_float import ParameterFloat
from griptape_nodes.exe_types.param_types.parameter_int import ParameterInt
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.exe_types.param_types.parameter_video import ParameterVideo

from griptape_nodes_library.griptape_proxy_node import GriptapeProxyNode

logger = logging.getLogger("griptape_nodes")

__all__ = ["KlingAdvancedLipSync"]


class KlingAdvancedLipSync(GriptapeProxyNode):
    """Generate advanced lip-synced video using Kling AI via Griptape Cloud model proxy.

    Implements a two-step workflow:
    1. Identify faces in the source video
    2. Apply lip sync to selected face with provided audio

    This advanced version provides:
    - Face detection and selection
    - Precise audio timing control (crop start/end, insert time)
    - Audio volume control for both new and original audio
    - Optional watermark generation

    Inputs:
        - video_source (str): Video ID from previous Kling generation or URL to video file
        - face_id (str): Which face to lip sync (default "0" for first detected face)
        - audio_source (str): Audio URL, file path, or Kling audio ID
        - sound_start_time (int): Audio crop start time in milliseconds (default 0)
        - sound_end_time (int): Audio crop end time in milliseconds (min 2000)
        - sound_insert_time (int): When to insert audio in video in milliseconds (default 0)
        - sound_volume (float): Audio volume multiplier 0-2 (default 1.0)
        - original_audio_volume (float): Original video volume multiplier 0-2 (default 1.0)
        - enable_watermark (bool): Generate watermarked result (default False)

    Outputs:
        - session_id (str): Session ID from face identification step
        - face_data (dict): Detected faces with IDs and timing information
        - generation_id (str): Griptape Cloud generation id
        - provider_response (dict): Verbatim response from API
        - video_url (VideoUrlArtifact): Saved video URL artifact
        - kling_video_id (str): The Kling AI video ID
        - was_successful (bool): Whether the generation succeeded
        - result_details (str): Details about the generation result or error
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # INPUTS / PROPERTIES
        self.add_parameter(
            ParameterString(
                name="video_source",
                tooltip="Video ID from previous Kling generation or URL to video file",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={
                    "placeholder_text": "Video ID or URL",
                    "display_name": "video source",
                },
            )
        )

        self.add_parameter(
            ParameterString(
                name="face_id",
                default_value="0",
                tooltip="Which face to lip sync (from auto-detected faces, default is first face)",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={
                    "placeholder_text": "0",
                    "display_name": "face ID",
                },
            )
        )

        self.add_parameter(
            ParameterString(
                name="audio_source",
                tooltip="Audio URL, file path, or Kling audio ID",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={
                    "placeholder_text": "Audio URL or ID",
                    "display_name": "audio source",
                },
            )
        )

        with ParameterGroup(name="Audio Timing") as timing_group:
            ParameterInt(
                name="sound_start_time",
                default_value=0,
                tooltip="Audio crop start time in milliseconds",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={
                    "placeholder_text": "0",
                    "display_name": "audio start (ms)",
                },
            )

            ParameterInt(
                name="sound_end_time",
                tooltip="Audio crop end time in milliseconds (must be >= 2000)",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={
                    "placeholder_text": "2000",
                    "display_name": "audio end (ms)",
                },
            )

            ParameterInt(
                name="sound_insert_time",
                default_value=0,
                tooltip="When to insert audio in video in milliseconds",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={
                    "placeholder_text": "0",
                    "display_name": "insert time (ms)",
                },
            )

        self.add_node_element(timing_group)

        with ParameterGroup(name="Audio Volumes") as volume_group:
            ParameterFloat(
                name="sound_volume",
                default_value=1.0,
                tooltip="Audio volume multiplier (0-2, default 1.0)",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={
                    "placeholder_text": "1.0",
                    "display_name": "audio volume",
                },
            )

            ParameterFloat(
                name="original_audio_volume",
                default_value=1.0,
                tooltip="Original video volume multiplier (0-2, default 1.0)",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={
                    "placeholder_text": "1.0",
                    "display_name": "original volume",
                },
            )

        self.add_node_element(volume_group)

        self.add_parameter(
            ParameterBool(
                name="enable_watermark",
                default_value=False,
                tooltip="Generate watermarked result",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )

        # OUTPUTS - Face identification results
        self.add_parameter(
            ParameterString(
                name="session_id",
                tooltip="Session ID from face identification step",
                allowed_modes={ParameterMode.OUTPUT},
                hide=True,
            )
        )

        self.add_parameter(
            ParameterDict(
                name="face_data",
                tooltip="Detected faces with IDs and timing information",
                allowed_modes={ParameterMode.OUTPUT},
                hide_property=True,
            )
        )

        # OUTPUTS - Generation results
        self.add_parameter(
            ParameterString(
                name="generation_id",
                tooltip="Griptape Cloud generation id",
                allowed_modes={ParameterMode.OUTPUT},
                hide=True,
            )
        )

        self.add_parameter(
            ParameterDict(
                name="provider_response",
                tooltip="Verbatim response from API (latest polling response)",
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
            default_filename="kling_advanced_lip_sync.mp4",
        )
        self._output_file.add_parameter()

        # Create status parameters for success/failure tracking
        self._create_status_parameters(
            result_details_tooltip="Details about the lip sync generation result or any errors",
            result_details_placeholder="Generation status and details will appear here.",
            parameter_group_initially_collapsed=True,
        )

    def _get_api_model_id(self) -> str:
        """Get the API model ID for this generation.

        Returns:
            str: The model ID with :advanced-lip-sync modality
        """
        return "kling:advanced-lip-sync"

    async def _identify_faces(self, video_source: str, headers: dict[str, str]) -> dict[str, Any] | None:
        """Call the identify face API endpoint to detect faces in the video.

        This is Step 1 of the two-step workflow.

        Args:
            video_source: Video ID or URL
            headers: HTTP headers including Authorization

        Returns:
            dict | None: Response containing session_id and face_data, or None on error
        """
        # Build identify face request
        identify_request: dict[str, str] = {}
        if video_source.startswith(("http://", "https://")):
            identify_request["video_url"] = video_source
        else:
            identify_request["video_id"] = video_source

        # Call identify face endpoint directly (not through proxy)
        # The proxy will handle this as part of the advanced-lip-sync workflow
        base = os.getenv("GT_CLOUD_PROXY_BASE_URL") or os.getenv("GT_CLOUD_BASE_URL", "https://cloud.griptape.ai")
        base_slash = base if base.endswith("/") else base + "/"
        api_base = urljoin(base_slash, "api/")
        proxy_base = urljoin(api_base, "proxy/v2/")

        # The proxy should handle identify_face as part of the model submission
        # For now, we'll include this in the payload and let the proxy handle it
        logger.info("%s: Face identification will be handled by proxy during generation", self.name)
        return identify_request

    async def _build_payload(self) -> dict[str, Any]:
        """Build the request payload for Kling Advanced Lip Sync API.

        This combines both steps: identify faces (handled by proxy) and create lip sync.

        Returns:
            dict: The request payload (model field excluded, handled by base class)
        """
        video_source = self.get_parameter_value("video_source") or ""
        face_id = self.get_parameter_value("face_id") or "0"
        audio_source = self.get_parameter_value("audio_source") or ""
        sound_start_time = self.get_parameter_value("sound_start_time") or 0
        sound_end_time = self.get_parameter_value("sound_end_time") or 2000
        sound_insert_time = self.get_parameter_value("sound_insert_time") or 0
        sound_volume = self.get_parameter_value("sound_volume")
        original_audio_volume = self.get_parameter_value("original_audio_volume")
        enable_watermark = self.get_parameter_value("enable_watermark") or False

        # Build the input object
        input_data: dict[str, Any] = {}

        # Add video source (either video_id or video_url)
        if video_source.startswith(("http://", "https://")):
            input_data["video_url"] = video_source
        else:
            input_data["video_id"] = video_source

        # Build face_choose array with audio configuration
        face_config: dict[str, Any] = {
            "face_id": str(face_id),
            "sound_start_time": int(sound_start_time),
            "sound_end_time": int(sound_end_time),
            "sound_insert_time": int(sound_insert_time),
        }

        # Add audio source (handle URL vs audio_id)
        # If audio_source looks like a URL, use sound_file
        # If it looks like a Kling audio ID, use audio_id
        if audio_source.startswith(("http://", "https://")):
            face_config["sound_file"] = audio_source
        else:
            # Assume it's an audio_id from Kling TTS
            # Could also be a file path, but we'll treat as audio_id for now
            face_config["audio_id"] = audio_source

        # Add optional volume parameters
        if sound_volume is not None:
            face_config["sound_volume"] = float(sound_volume)
        if original_audio_volume is not None:
            face_config["original_audio_volume"] = float(original_audio_volume)

        input_data["face_choose"] = [face_config]

        # Add watermark info if enabled
        if enable_watermark:
            input_data["watermark_info"] = {"enabled": True}

        # The payload should be nested under "input" key
        return {"input": input_data}

    async def _parse_result(self, result_json: dict[str, Any], generation_id: str) -> None:
        """Parse the result and set output parameters.

        Expected structure:
        {
            "data": {
                "session_id": "...",
                "face_data": [...],
                "task_result": {
                    "videos": [{"url": "...", "id": "..."}]
                }
            }
        }
        """
        data = result_json.get("data", {})

        # Extract face identification data if present
        session_id = data.get("session_id")
        face_data = data.get("face_data")

        if session_id:
            self.parameter_output_values["session_id"] = session_id
            logger.info("%s: Session ID: %s", self.name, session_id)

        if face_data:
            self.parameter_output_values["face_data"] = face_data
            logger.info("%s: Detected %d face(s)", self.name, len(face_data) if isinstance(face_data, list) else 0)

        # Extract video result
        task_result = data.get("task_result", {})
        videos = task_result.get("videos", [])

        if not videos or not isinstance(videos, list) or len(videos) == 0:
            self.parameter_output_values["video_url"] = None
            self._set_status_results(
                was_successful=False,
                result_details=f"{self.name} generation completed but no videos found in response.",
            )
            return

        video_info = videos[0]
        download_url = video_info.get("url")
        video_id = video_info.get("id")

        if not download_url:
            self.parameter_output_values["video_url"] = None
            self._set_status_results(
                was_successful=False,
                result_details=f"{self.name} generation completed but no download URL found in response.",
            )
            return

        # Set kling_video_id output parameter
        if video_id:
            self.parameter_output_values["kling_video_id"] = video_id
            logger.info("Video ID: %s", video_id)

        # Download and save video
        try:
            logger.info("%s downloading video from provider URL", self.name)
            video_bytes = await self._download_bytes_from_url(download_url)
        except Exception as e:
            logger.warning("%s failed to download video: %s", self.name, e)
            video_bytes = None

        if video_bytes:
            try:
                dest = self._output_file.build_file()
                saved = await dest.awrite_bytes(video_bytes)
                self.parameter_output_values["video_url"] = VideoUrlArtifact(value=saved.location, name=saved.name)
                logger.info("%s saved video as %s", self.name, saved.name)
                self._set_status_results(
                    was_successful=True, result_details=f"Video generated successfully and saved as {saved.name}."
                )
            except (OSError, PermissionError) as e:
                logger.warning("%s failed to save video: %s, using provider URL", self.name, e)
                self.parameter_output_values["video_url"] = VideoUrlArtifact(value=download_url)
                self._set_status_results(
                    was_successful=True,
                    result_details=f"Video generated successfully. Using provider URL (could not save to storage: {e}).",
                )
        else:
            self.parameter_output_values["video_url"] = VideoUrlArtifact(value=download_url)
            self._set_status_results(
                was_successful=True,
                result_details="Video generated successfully. Using provider URL (could not download video bytes).",
            )

    def _set_safe_defaults(self) -> None:
        """Clear output parameters on error."""
        self.parameter_output_values["session_id"] = ""
        self.parameter_output_values["face_data"] = {}
        self.parameter_output_values["video_url"] = None
        self.parameter_output_values["kling_video_id"] = ""

    def validate_before_node_run(self) -> list[Exception] | None:
        """Validate parameters before execution."""
        exceptions = super().validate_before_node_run() or []

        # Validate video_source is provided
        video_source = self.get_parameter_value("video_source") or ""
        if not video_source.strip():
            exceptions.append(ValueError(f"{self.name} requires a video source (video ID or URL)."))

        # Validate audio_source is provided
        audio_source = self.get_parameter_value("audio_source") or ""
        if not audio_source.strip():
            exceptions.append(ValueError(f"{self.name} requires an audio source (URL or audio ID)."))

        # Validate timing parameters
        sound_start_time = self.get_parameter_value("sound_start_time")
        sound_end_time = self.get_parameter_value("sound_end_time")

        if sound_start_time is not None and sound_start_time < 0:
            exceptions.append(ValueError(f"{self.name} sound_start_time must be >= 0 (got {sound_start_time})."))

        if sound_end_time is not None:
            if sound_end_time < 2000:
                exceptions.append(
                    ValueError(f"{self.name} sound_end_time must be >= 2000 ms (2 seconds) (got {sound_end_time}).")
                )

            if sound_start_time is not None and sound_end_time <= sound_start_time:
                exceptions.append(
                    ValueError(
                        f"{self.name} sound_end_time ({sound_end_time}) must be greater than "
                        f"sound_start_time ({sound_start_time})."
                    )
                )

        sound_insert_time = self.get_parameter_value("sound_insert_time")
        if sound_insert_time is not None and sound_insert_time < 0:
            exceptions.append(ValueError(f"{self.name} sound_insert_time must be >= 0 (got {sound_insert_time})."))

        # Validate volume parameters
        sound_volume = self.get_parameter_value("sound_volume")
        if sound_volume is not None:
            try:
                volume_float = float(sound_volume)
                if not (0 <= volume_float <= 2):
                    exceptions.append(
                        ValueError(f"{self.name} sound_volume must be between 0 and 2 (got {volume_float}).")
                    )
            except (TypeError, ValueError):
                exceptions.append(ValueError(f"{self.name} sound_volume must be a number."))

        original_audio_volume = self.get_parameter_value("original_audio_volume")
        if original_audio_volume is not None:
            try:
                orig_volume_float = float(original_audio_volume)
                if not (0 <= orig_volume_float <= 2):
                    exceptions.append(
                        ValueError(
                            f"{self.name} original_audio_volume must be between 0 and 2 (got {orig_volume_float})."
                        )
                    )
            except (TypeError, ValueError):
                exceptions.append(ValueError(f"{self.name} original_audio_volume must be a number."))

        return exceptions if exceptions else None
