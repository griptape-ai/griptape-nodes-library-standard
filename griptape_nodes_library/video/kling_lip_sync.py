from __future__ import annotations

import logging
from typing import Any, ClassVar

from griptape.artifacts.video_url_artifact import VideoUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterGroup, ParameterMode
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_types.parameter_bool import ParameterBool
from griptape_nodes.exe_types.param_types.parameter_dict import ParameterDict
from griptape_nodes.exe_types.param_types.parameter_float import ParameterFloat
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.exe_types.param_types.parameter_video import ParameterVideo
from griptape_nodes.traits.options import Options

from griptape_nodes_library.griptape_proxy_node import GriptapeProxyNode

logger = logging.getLogger("griptape_nodes")

__all__ = ["KlingLipSync"]

# Voice ID mapping for TTS mode
VOICE_IDS: ClassVar[dict[str, str]] = {
    "Sunny": "genshin_vindi2",
    "Sage": "zhinen_xuesheng",
    "Blossom": "ai_shatang",
    "Peppy": "genshin_klee2",
    "Dove": "genshin_kirara",
    "Shine": "ai_kaiya",
    "Anchor": "oversea_male1",
    "Lyric": "ai_chenjiahao_712",
    "Melody": "girlfriend_4_speech02",
    "Tender": "chat1_female_new-3",
    "Zippy": "cartoon-boy-07",
    "Sprite": "cartoon-girl-01",
    "Rock": "ai_huangyaoshi_712",
    "Grace": "chengshu_jiejie",
    "Helen": "you_pingjing",
    "The Reader": "reader_en_m-v1",
    "Commercial Lady": "commercial_lady_en_f-v1",
}


class KlingLipSync(GriptapeProxyNode):
    """Generate lip-synced video using Kling AI via Griptape Cloud model proxy.

    Supports two modes:
    - text2video: Synthesize speech from text and apply lip sync to video
    - audio2video: Apply lip sync from provided audio to video

    Inputs:
        - video_source (str): Video ID from previous Kling generation or URL to video file
        - mode (str): Generation mode (text2video or audio2video)
        - text (str): Text to convert to speech (required if mode=text2video)
        - voice_id (str): Voice to use for TTS (required if mode=text2video)
        - voice_language (str): Language for text-to-speech (zh or en)
        - voice_speed (float): Speed multiplier for synthesized voice (0.8-2.0)
        - audio_source (str): Audio URL or file path (required if mode=audio2video)
        - audio_type (str): MIME type of audio file (required if mode=audio2video)

    Outputs:
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
                name="mode",
                default_value="text2video",
                tooltip="Generation mode: synthesize speech from text or use provided audio",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=["text2video", "audio2video"])},
            )
        )

        with ParameterGroup(name="Text-to-Speech Settings") as tts_group:
            ParameterString(
                name="text",
                tooltip="Text to convert to speech (required if mode=text2video)",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={
                    "multiline": True,
                    "placeholder_text": "Enter text to synthesize...",
                },
            )

            ParameterString(
                name="voice_id",
                default_value="Sunny",
                tooltip="Voice to use for text-to-speech",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=list(VOICE_IDS.keys()))},
            )

            ParameterString(
                name="voice_language",
                default_value="zh",
                tooltip="Language for text-to-speech synthesis",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=["zh", "en"])},
            )

            ParameterFloat(
                name="voice_speed",
                default_value=1.0,
                tooltip="Speed multiplier for synthesized voice (0.8-2.0)",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )

        self.add_node_element(tts_group)

        with ParameterGroup(name="Audio Settings") as audio_group:
            ParameterString(
                name="audio_source",
                tooltip="Audio URL or file path (required if mode=audio2video)",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={
                    "placeholder_text": "Audio URL",
                    "display_name": "audio source",
                    "hide": True,
                },
            )

            ParameterString(
                name="audio_type",
                default_value="audio/mpeg",
                tooltip="MIME type of audio file",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"hide": True},
            )

        self.add_node_element(audio_group)

        # OUTPUTS
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
            default_filename="kling_lip_sync.mp4",
        )
        self._output_file.add_parameter()

        # Create status parameters for success/failure tracking
        self._create_status_parameters(
            result_details_tooltip="Details about the lip sync generation result or any errors",
            result_details_placeholder="Generation status and details will appear here.",
            parameter_group_initially_collapsed=True,
        )

        # Set initial parameter visibility based on default mode
        self._update_parameter_visibility_for_mode("text2video")

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        """Handle parameter value changes to show/hide dependent parameters."""
        super().after_value_set(parameter, value)

        if parameter.name == "mode":
            self._update_parameter_visibility_for_mode(value)

    def _update_parameter_visibility_for_mode(self, mode: str) -> None:
        """Update parameter visibility based on selected mode."""
        if mode == "text2video":
            # Show text-to-speech parameters
            self.show_parameter_by_name(["text", "voice_id", "voice_language", "voice_speed"])
            # Hide audio parameters
            self.hide_parameter_by_name(["audio_source", "audio_type"])
        elif mode == "audio2video":
            # Hide text-to-speech parameters
            self.hide_parameter_by_name(["text", "voice_id", "voice_language", "voice_speed"])
            # Show audio parameters
            self.show_parameter_by_name(["audio_source", "audio_type"])

    def _get_api_model_id(self) -> str:
        """Get the API model ID for this generation.

        Returns:
            str: The model ID with :lip-sync modality
        """
        return "kling:lip-sync"

    async def _build_payload(self) -> dict[str, Any]:
        """Build the request payload for Kling Lip Sync API.

        Returns:
            dict: The request payload (model field excluded, handled by base class)
        """
        video_source = self.get_parameter_value("video_source") or ""
        mode = self.get_parameter_value("mode") or "text2video"

        # Build the input object
        input_data: dict[str, Any] = {"mode": mode}

        # Add video source (either video_id or video_url)
        # If it looks like a URL, use video_url; otherwise assume it's a video_id
        if video_source.startswith(("http://", "https://")):
            input_data["video_url"] = video_source
        else:
            input_data["video_id"] = video_source

        # Add mode-specific parameters
        if mode == "text2video":
            text = self.get_parameter_value("text") or ""
            voice_id_friendly = self.get_parameter_value("voice_id") or "Sunny"
            voice_id = VOICE_IDS.get(voice_id_friendly, "genshin_vindi2")
            voice_language = self.get_parameter_value("voice_language") or "zh"
            voice_speed = self.get_parameter_value("voice_speed")

            input_data["text"] = text.strip()
            input_data["voice_id"] = voice_id
            input_data["voice_language"] = voice_language
            if voice_speed is not None:
                input_data["voice_speed"] = float(voice_speed)

        elif mode == "audio2video":
            audio_source = self.get_parameter_value("audio_source") or ""
            audio_type = self.get_parameter_value("audio_type") or "audio/mpeg"

            # If audio_source looks like a URL, use audio_url
            # Otherwise, could be a file path that needs to be handled
            if audio_source.startswith(("http://", "https://")):
                input_data["audio_url"] = audio_source
            else:
                # For now, treat as URL. Could be extended to handle file uploads
                input_data["audio_url"] = audio_source

            input_data["audio_type"] = audio_type

        # The payload should be nested under "input" key
        return {"input": input_data}

    async def _parse_result(self, result_json: dict[str, Any], generation_id: str) -> None:
        """Parse the result and set output parameters.

        Expected structure: {"data": {"task_result": {"videos": [{"url": "...", "id": "..."}]}}}
        """
        data = result_json.get("data", {})
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
        self.parameter_output_values["video_url"] = None
        self.parameter_output_values["kling_video_id"] = ""

    def validate_before_node_run(self) -> list[Exception] | None:
        """Validate parameters before execution."""
        exceptions = super().validate_before_node_run() or []

        # Validate video_source is provided
        video_source = self.get_parameter_value("video_source") or ""
        if not video_source.strip():
            exceptions.append(ValueError(f"{self.name} requires a video source (video ID or URL)."))

        # Validate mode
        mode = self.get_parameter_value("mode") or "text2video"
        if mode not in {"text2video", "audio2video"}:
            exceptions.append(ValueError(f"{self.name} mode must be 'text2video' or 'audio2video'."))

        # Mode-specific validation
        if mode == "text2video":
            # Validate text is provided
            text = self.get_parameter_value("text") or ""
            if not text.strip():
                exceptions.append(ValueError(f"{self.name} requires text input when mode is 'text2video'."))

            # Validate voice_speed range
            voice_speed = self.get_parameter_value("voice_speed")
            if voice_speed is not None:
                try:
                    voice_speed_float = float(voice_speed)
                    if not (0.8 <= voice_speed_float <= 2.0):
                        exceptions.append(
                            ValueError(
                                f"{self.name} voice_speed must be between 0.8 and 2.0 (got {voice_speed_float})."
                            )
                        )
                except (TypeError, ValueError):
                    exceptions.append(ValueError(f"{self.name} voice_speed must be a number."))

        elif mode == "audio2video":
            # Validate audio_source is provided
            audio_source = self.get_parameter_value("audio_source") or ""
            if not audio_source.strip():
                exceptions.append(ValueError(f"{self.name} requires an audio source when mode is 'audio2video'."))

            # Validate audio_type is provided
            audio_type = self.get_parameter_value("audio_type") or ""
            if not audio_type.strip():
                exceptions.append(ValueError(f"{self.name} requires an audio type when mode is 'audio2video'."))

        return exceptions if exceptions else None
