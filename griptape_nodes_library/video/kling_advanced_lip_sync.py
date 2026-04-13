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
from griptape_nodes.exe_types.param_types.parameter_bool import ParameterBool
from griptape_nodes.exe_types.param_types.parameter_dict import ParameterDict
from griptape_nodes.exe_types.param_types.parameter_float import ParameterFloat
from griptape_nodes.exe_types.param_types.parameter_int import ParameterInt
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.exe_types.param_types.parameter_video import ParameterVideo
from griptape_nodes.traits.options import Options

from griptape_nodes_library.proxy import GriptapeProxyNode

logger = logging.getLogger("griptape_nodes")

__all__ = ["KlingAdvancedLipSync"]

MIN_AUDIO_DURATION_MS = 2000


class KlingAdvancedLipSync(GriptapeProxyNode):
    """Create a Kling advanced lip-sync animation from a face-identification session."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.add_parameter(
            ParameterString(
                name="session_id",
                tooltip="Session id returned by Kling Identify Face.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            ParameterDict(
                name="selected_face",
                tooltip="Selected face dictionary from Kling Identify Face.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            ParameterString(
                name="audio_input_type",
                default_value="sound_file",
                tooltip="Choose whether to provide a Kling audio id or a local/URL audio file.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=["sound_file", "audio_id"])},
                ui_options={"display_name": "audio input type"},
            )
        )
        self.add_parameter(
            ParameterString(
                name="audio_id",
                tooltip="Audio id from a prior Kling TTS generation.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                hide=True,
            )
        )

        self._public_audio_url_parameter = PublicArtifactUrlParameter(
            node=self,
            artifact_url_parameter=ParameterAudio(
                name="sound_file",
                tooltip="Audio URL or uploaded audio file for lip sync.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"display_name": "audio"},
            ),
            disclaimer_message="The Kling lip-sync service utilizes this URL to access the audio file.",
        )
        self._public_audio_url_parameter.add_input_parameters()

        with ParameterGroup(name="Timing") as timing_group:
            ParameterInt(
                name="sound_start_time",
                default_value=0,
                tooltip="Start time in the source audio, in milliseconds.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"display_name": "sound start time (ms)"},
            )
            ParameterInt(
                name="sound_end_time",
                default_value=3000,
                tooltip="End time in the source audio, in milliseconds.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"display_name": "sound end time (ms)"},
            )
            ParameterInt(
                name="sound_insert_time",
                default_value=0,
                tooltip="Video insertion start time for the cropped audio, in milliseconds.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"display_name": "sound insert time (ms)"},
            )
        self.add_node_element(timing_group)

        with ParameterGroup(name="Audio Mixing") as audio_group:
            ParameterFloat(
                name="sound_volume",
                default_value=1.0,
                tooltip="Lip-sync audio volume. Kling supports values from 0 to 2.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
            ParameterFloat(
                name="original_audio_volume",
                default_value=1.0,
                tooltip="Original video audio volume. Kling supports values from 0 to 2.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"display_name": "original audio volume"},
            )
            ParameterBool(
                name="watermark",
                default_value=False,
                tooltip="Request a watermarked result in addition to the standard result.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        self.add_node_element(audio_group)

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
            default_filename="kling_advanced_lip_sync.mp4",
        )
        self._output_file.add_parameter()

        self._create_status_parameters(
            result_details_tooltip="Details about the advanced lip-sync result or any errors",
            result_details_placeholder="Advanced lip-sync status and details will appear here.",
            parameter_group_initially_collapsed=True,
        )
        self._update_audio_input_visibility(self.get_parameter_value("audio_input_type"))

    async def aprocess(self) -> None:
        try:
            await super().aprocess()
        finally:
            self._public_audio_url_parameter.delete_uploaded_artifact()

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        super().after_value_set(parameter, value)

        if parameter.name == "audio_input_type":
            self._update_audio_input_visibility(value)

    def _update_audio_input_visibility(self, value: Any) -> None:
        if value == "audio_id":
            self.show_parameter_by_name("audio_id")
            self.hide_parameter_by_name("sound_file")
            self.hide_message_by_name("artifact_url_parameter_message_sound_file")
            return

        self.hide_parameter_by_name("audio_id")
        self.show_parameter_by_name("sound_file")
        self.show_message_by_name("artifact_url_parameter_message_sound_file")

    def _get_api_model_id(self) -> str:
        return "kling:advanced-lip-sync"

    async def _build_payload(self) -> dict[str, Any]:
        session_id = (self.get_parameter_value("session_id") or "").strip()
        selected_face = self.get_parameter_value("selected_face") or {}
        if not isinstance(selected_face, dict):
            raise ValueError("selected_face must be a dictionary from Kling Identify Face")

        face_id = str(selected_face.get("face_id") or "").strip()
        if not face_id:
            raise ValueError("selected_face.face_id is required")

        face_choose_item: dict[str, Any] = {
            "face_id": face_id,
            "sound_start_time": int(self.get_parameter_value("sound_start_time") or 0),
            "sound_end_time": int(self.get_parameter_value("sound_end_time") or 0),
            "sound_insert_time": int(self.get_parameter_value("sound_insert_time") or 0),
            "sound_volume": float(self.get_parameter_value("sound_volume") or 1.0),
            "original_audio_volume": float(self.get_parameter_value("original_audio_volume") or 1.0),
        }

        audio_input_type = self.get_parameter_value("audio_input_type") or "sound_file"
        if audio_input_type == "audio_id":
            face_choose_item["audio_id"] = (self.get_parameter_value("audio_id") or "").strip()
        else:
            sound_file_url = self._public_audio_url_parameter.get_public_url_for_parameter()
            face_choose_item["sound_file"] = (sound_file_url or "").strip()

        payload: dict[str, Any] = {
            "session_id": session_id,
            "face_choose": [face_choose_item],
            "watermark_info": {"enabled": bool(self.get_parameter_value("watermark"))},
        }
        return payload

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
                    result_details=f"Advanced lip sync completed successfully and saved as {saved.name}.",
                )
                return
            except (OSError, PermissionError) as e:
                logger.warning("%s failed to save video locally: %s", self.name, e)

        self.parameter_output_values["video_url"] = VideoUrlArtifact(value=download_url)
        self._set_status_results(
            was_successful=True,
            result_details=(
                f"Advanced lip sync completed successfully for generation {generation_id}. "
                "Using provider URL because local caching was unavailable."
            ),
        )

    def _set_safe_defaults(self) -> None:
        self.parameter_output_values["video_url"] = None
        self.parameter_output_values["kling_video_id"] = ""

    def validate_before_node_run(self) -> list[Exception] | None:
        exceptions = super().validate_before_node_run() or []

        session_id = (self.get_parameter_value("session_id") or "").strip()
        selected_face = self.get_parameter_value("selected_face") or {}
        audio_input_type = self.get_parameter_value("audio_input_type") or "sound_file"
        sound_start_time = self.get_parameter_value("sound_start_time") or 0
        sound_end_time = self.get_parameter_value("sound_end_time") or 0
        sound_insert_time = self.get_parameter_value("sound_insert_time") or 0
        sound_volume = self.get_parameter_value("sound_volume") or 1.0
        original_audio_volume = self.get_parameter_value("original_audio_volume") or 1.0

        if not session_id:
            exceptions.append(ValueError(f"{self.name} requires session_id from Kling Identify Face."))

        if not isinstance(selected_face, dict) or not str(selected_face.get("face_id") or "").strip():
            exceptions.append(ValueError(f"{self.name} requires selected_face with a face_id."))

        if audio_input_type == "audio_id":
            audio_id = (self.get_parameter_value("audio_id") or "").strip()
            if not audio_id:
                exceptions.append(ValueError(f"{self.name} requires audio_id when audio input type is audio_id."))
        else:
            sound_file = self.get_parameter_value("sound_file")
            if not sound_file:
                exceptions.append(
                    ValueError(f"{self.name} requires an audio file or URL when audio input type is sound_file.")
                )

        try:
            sound_start_time = int(sound_start_time)
            sound_end_time = int(sound_end_time)
            sound_insert_time = int(sound_insert_time)
        except (TypeError, ValueError):
            exceptions.append(ValueError(f"{self.name} timing inputs must be integers in milliseconds."))
        else:
            if sound_start_time < 0 or sound_end_time < 0 or sound_insert_time < 0:
                exceptions.append(ValueError(f"{self.name} timing inputs must be 0 or greater."))
            cropped_duration = sound_end_time - sound_start_time
            if cropped_duration < MIN_AUDIO_DURATION_MS:
                exceptions.append(
                    ValueError(
                        f"{self.name} requires at least {MIN_AUDIO_DURATION_MS}ms between sound_start_time and sound_end_time."
                    )
                )

            face_start_time = selected_face.get("start_time") if isinstance(selected_face, dict) else None
            face_end_time = selected_face.get("end_time") if isinstance(selected_face, dict) else None
            try:
                face_start_time = int(face_start_time)
                face_end_time = int(face_end_time)
            except (TypeError, ValueError):
                face_start_time = None
                face_end_time = None

            if face_start_time is not None and face_end_time is not None:
                insert_end_time = sound_insert_time + cropped_duration
                overlap_ms = min(insert_end_time, face_end_time) - max(sound_insert_time, face_start_time)
                if overlap_ms < MIN_AUDIO_DURATION_MS:
                    exceptions.append(
                        ValueError(
                            f"{self.name} requires the inserted audio interval to overlap the selected face interval by at least {MIN_AUDIO_DURATION_MS}ms."
                        )
                    )

        for volume_name, value in {
            "sound_volume": sound_volume,
            "original_audio_volume": original_audio_volume,
        }.items():
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                exceptions.append(ValueError(f"{self.name} {volume_name} must be a number between 0 and 2."))
                continue

            if not 0 <= numeric_value <= 2:
                exceptions.append(ValueError(f"{self.name} {volume_name} must be between 0 and 2."))

        return exceptions if exceptions else None
