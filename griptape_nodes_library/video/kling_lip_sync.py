from __future__ import annotations

import logging
from typing import Any

from griptape.artifacts.video_url_artifact import VideoUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterGroup, ParameterMode
from griptape_nodes.exe_types.param_components.artifact_url.public_artifact_url_parameter import (
    PublicArtifactUrlParameter,
)
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_types.parameter_audio import ParameterAudio
from griptape_nodes.exe_types.param_types.parameter_dict import ParameterDict
from griptape_nodes.exe_types.param_types.parameter_float import ParameterFloat
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.exe_types.param_types.parameter_video import ParameterVideo
from griptape_nodes.traits.options import Options

from griptape_nodes_library.griptape_proxy_node import GriptapeProxyNode

logger = logging.getLogger("griptape_nodes")

__all__ = ["KlingLipSync"]

MAX_TEXT_LENGTH = 120


class KlingLipSync(GriptapeProxyNode):
    """Create a simple Kling lip-sync video via the Griptape Cloud model proxy.

    The provider contract for this endpoint is inferred from local secondary sources.
    """

    MODEL_CHOICES = ["kling-v1-5", "kling-v1-6", "kling-v2", "kling-v2-1"]
    VOICE_LANGUAGE_CHOICES = ["en", "zh"]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.add_parameter(
            ParameterString(
                name="model_name",
                default_value="kling-v2-1",
                tooltip="Inferred simple lip-sync model selection.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=self.MODEL_CHOICES)},
                ui_options={"display_name": "model"},
            )
        )
        self.add_parameter(
            ParameterString(
                name="video_input_type",
                default_value="video_url",
                tooltip="Choose whether to lip-sync a prior Kling video id or a video URL/file.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=["video_url", "video_id"])},
                ui_options={"display_name": "video input type"},
            )
        )
        self.add_parameter(
            ParameterString(
                name="video_id",
                tooltip="Kling video id to lip sync.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                hide=True,
            )
        )
        self._public_video_url_parameter = PublicArtifactUrlParameter(
            node=self,
            artifact_url_parameter=ParameterVideo(
                name="input_video",
                tooltip="Video URL or uploaded video file for simple lip sync.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"display_name": "video"},
            ),
            disclaimer_message="The Kling lip-sync service utilizes this URL to access the video.",
        )
        self._public_video_url_parameter.add_input_parameters()

        self.add_parameter(
            ParameterString(
                name="mode",
                default_value="text2video",
                tooltip="Use text-to-speech or provide an audio file directly.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=["text2video", "audio2video"])},
            )
        )

        with ParameterGroup(name="Text To Speech") as tts_group:
            ParameterString(
                name="text",
                default_value="",
                tooltip="Text to speak for the lip-sync result. Maximum 120 characters.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"multiline": True},
            )
            ParameterString(
                name="voice_id",
                default_value="oversea_male1",
                tooltip="Inferred Kling voice id for text-to-speech mode.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
            ParameterString(
                name="voice_language",
                default_value="en",
                tooltip="Voice language for text-to-speech mode.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=self.VOICE_LANGUAGE_CHOICES)},
                ui_options={"display_name": "voice language"},
            )
            ParameterFloat(
                name="voice_speed",
                default_value=1.0,
                tooltip="Voice speed in text-to-speech mode. Secondary-source range: 0.8-2.0.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        self.add_node_element(tts_group)

        self._public_audio_url_parameter = PublicArtifactUrlParameter(
            node=self,
            artifact_url_parameter=ParameterAudio(
                name="input_audio",
                tooltip="Audio URL or uploaded audio file for simple lip sync.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"display_name": "audio", "hide": True},
            ),
            disclaimer_message="The Kling lip-sync service utilizes this URL to access the audio file.",
        )
        self._public_audio_url_parameter.add_input_parameters()
        self.hide_message_by_name("artifact_url_parameter_message_input_audio")

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
                tooltip="Verbatim response from API",
                allowed_modes={ParameterMode.OUTPUT},
                hide_property=True,
                hide=True,
            )
        )
        self.add_parameter(
            ParameterVideo(
                name="video_url",
                tooltip="Saved lip-synced video as URL artifact",
                allowed_modes={ParameterMode.OUTPUT, ParameterMode.PROPERTY},
                settable=False,
                ui_options={"pulse_on_run": True},
            )
        )
        self.add_parameter(
            ParameterString(
                name="kling_video_id",
                tooltip="The Kling AI video id for the lip-synced output",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

        self._output_file = ProjectFileParameter(
            node=self,
            name="output_file",
            default_filename="kling_lip_sync.mp4",
        )
        self._output_file.add_parameter()

        self._create_status_parameters(
            result_details_tooltip="Details about the simple lip-sync result or any errors",
            result_details_placeholder="Simple lip-sync status and details will appear here.",
            parameter_group_initially_collapsed=True,
        )
        self._update_video_input_visibility(self.get_parameter_value("video_input_type"))
        self._update_mode_visibility(self.get_parameter_value("mode"))

    async def aprocess(self) -> None:
        try:
            await super().aprocess()
        finally:
            self._public_video_url_parameter.delete_uploaded_artifact()
            self._public_audio_url_parameter.delete_uploaded_artifact()

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        super().after_value_set(parameter, value)

        if parameter.name == "video_input_type":
            self._update_video_input_visibility(value)
        if parameter.name == "mode":
            self._update_mode_visibility(value)

    def _update_video_input_visibility(self, value: Any) -> None:
        if value == "video_id":
            self.show_parameter_by_name("video_id")
            self.hide_parameter_by_name("input_video")
            self.hide_message_by_name("artifact_url_parameter_message_input_video")
            return

        self.hide_parameter_by_name("video_id")
        self.show_parameter_by_name("input_video")
        self.show_message_by_name("artifact_url_parameter_message_input_video")

    def _update_mode_visibility(self, value: Any) -> None:
        if value == "audio2video":
            self.hide_parameter_by_name(["text", "voice_id", "voice_language", "voice_speed"])
            self.show_parameter_by_name("input_audio")
            self.show_message_by_name("artifact_url_parameter_message_input_audio")
            return

        self.show_parameter_by_name(["text", "voice_id", "voice_language", "voice_speed"])
        self.hide_parameter_by_name("input_audio")
        self.hide_message_by_name("artifact_url_parameter_message_input_audio")

    def _get_api_model_id(self) -> str:
        return "kling:lip-sync"

    async def _build_payload(self) -> dict[str, Any]:
        input_payload: dict[str, Any] = {
            "model_name": self.get_parameter_value("model_name") or "kling-v2-1",
            "mode": self.get_parameter_value("mode") or "text2video",
        }

        video_input_type = self.get_parameter_value("video_input_type") or "video_url"
        if video_input_type == "video_id":
            input_payload["video_id"] = (self.get_parameter_value("video_id") or "").strip()
        else:
            video_url = self._public_video_url_parameter.get_public_url_for_parameter()
            input_payload["video_url"] = (video_url or "").strip()

        mode = input_payload["mode"]
        if mode == "audio2video":
            audio_url = self._public_audio_url_parameter.get_public_url_for_parameter()
            input_payload["audio_type"] = "url"
            input_payload["audio_url"] = (audio_url or "").strip()
        else:
            input_payload["text"] = (self.get_parameter_value("text") or "").strip()
            input_payload["voice_id"] = (self.get_parameter_value("voice_id") or "").strip()
            input_payload["voice_language"] = self.get_parameter_value("voice_language") or "en"
            input_payload["voice_speed"] = float(self.get_parameter_value("voice_speed") or 1.0)

        return {"input": input_payload}

    async def _parse_result(self, result_json: dict[str, Any], generation_id: str) -> None:
        data = result_json.get("data", {})
        task_result = data.get("task_result", {})
        videos = task_result.get("videos", [])

        if not videos or not isinstance(videos, list):
            self.parameter_output_values["video_url"] = None
            self._set_status_results(
                was_successful=False,
                result_details=f"{self.name} completed but no videos were returned.",
            )
            return

        video_info = videos[0]
        download_url = video_info.get("url")
        video_id = video_info.get("id")
        if not download_url:
            self.parameter_output_values["video_url"] = None
            self._set_status_results(
                was_successful=False,
                result_details=f"{self.name} completed but no download URL was returned.",
            )
            return

        self.parameter_output_values["kling_video_id"] = str(video_id or "")

        try:
            video_bytes = await self._download_bytes_from_url(download_url)
        except Exception as e:  # pragma: no cover - defensive logging path
            logger.warning("%s failed to download video: %s", self.name, e)
            video_bytes = None

        if video_bytes:
            try:
                dest = self._output_file.build_file()
                saved = await dest.awrite_bytes(video_bytes)
                self.parameter_output_values["video_url"] = VideoUrlArtifact(
                    value=saved.location,
                    name=saved.name,
                )
                self._set_status_results(
                    was_successful=True,
                    result_details=f"Simple lip sync completed successfully and saved as {saved.name}.",
                )
                return
            except (OSError, PermissionError) as e:
                logger.warning("%s failed to save video locally: %s", self.name, e)

        self.parameter_output_values["video_url"] = VideoUrlArtifact(value=download_url)
        self._set_status_results(
            was_successful=True,
            result_details=(
                f"Simple lip sync completed successfully for generation {generation_id}. "
                "Using provider URL because local caching was unavailable."
            ),
        )

    def _set_safe_defaults(self) -> None:
        self.parameter_output_values["video_url"] = None
        self.parameter_output_values["kling_video_id"] = ""

    def validate_before_node_run(self) -> list[Exception] | None:
        exceptions = super().validate_before_node_run() or []

        video_input_type = self.get_parameter_value("video_input_type") or "video_url"
        mode = self.get_parameter_value("mode") or "text2video"
        voice_speed = self.get_parameter_value("voice_speed") or 1.0

        if video_input_type == "video_id":
            video_id = (self.get_parameter_value("video_id") or "").strip()
            if not video_id:
                exceptions.append(ValueError(f"{self.name} requires video_id when video input type is video_id."))
        else:
            video_url = self.get_parameter_value("input_video")
            if not video_url:
                exceptions.append(ValueError(f"{self.name} requires a video file or URL when video input type is video_url."))

        if mode == "audio2video":
            input_audio = self.get_parameter_value("input_audio")
            if not input_audio:
                exceptions.append(ValueError(f"{self.name} requires input_audio when mode is audio2video."))
        else:
            text = (self.get_parameter_value("text") or "").strip()
            voice_id = (self.get_parameter_value("voice_id") or "").strip()
            if not text:
                exceptions.append(ValueError(f"{self.name} requires text when mode is text2video."))
            elif len(text) > MAX_TEXT_LENGTH:
                exceptions.append(
                    ValueError(f"{self.name} text exceeds {MAX_TEXT_LENGTH} characters (got {len(text)}).")
                )
            if not voice_id:
                exceptions.append(ValueError(f"{self.name} requires voice_id when mode is text2video."))

        try:
            voice_speed = float(voice_speed)
        except (TypeError, ValueError):
            exceptions.append(ValueError(f"{self.name} voice_speed must be a number between 0.8 and 2.0."))
        else:
            if not 0.8 <= voice_speed <= 2.0:
                exceptions.append(ValueError(f"{self.name} voice_speed must be between 0.8 and 2.0."))

        return exceptions if exceptions else None
