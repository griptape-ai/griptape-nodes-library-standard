from __future__ import annotations

import logging
from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.param_components.artifact_url.public_artifact_url_parameter import (
    PublicArtifactUrlParameter,
)
from griptape_nodes.exe_types.param_types.parameter_dict import ParameterDict
from griptape_nodes.exe_types.param_types.parameter_int import ParameterInt
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.exe_types.param_types.parameter_video import ParameterVideo
from griptape_nodes.traits.options import Options

from griptape_nodes_library.griptape_proxy_node import GriptapeProxyNode

logger = logging.getLogger("griptape_nodes")

__all__ = ["KlingIdentifyFace"]


class KlingIdentifyFace(GriptapeProxyNode):
    """Identify lip-sync candidate faces in a video using Kling AI via Griptape Cloud model proxy."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.add_parameter(
            ParameterString(
                name="video_input_type",
                default_value="video_url",
                tooltip="Choose whether to identify faces from a prior Kling video id or from a video URL/file.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=["video_url", "video_id"])},
                ui_options={"display_name": "video input type"},
            )
        )
        self.add_parameter(
            ParameterString(
                name="video_id",
                tooltip="Kling video id to inspect for lip-sync faces.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                placeholder_text="The Kling AI video id",
                hide=True,
            )
        )

        self._public_video_url_parameter = PublicArtifactUrlParameter(
            node=self,
            artifact_url_parameter=ParameterVideo(
                name="video_url",
                tooltip="Video URL or uploaded video file for face identification.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"display_name": "video"},
            ),
            disclaimer_message="The Kling lip-sync service utilizes this URL to access the video for face identification.",
        )
        self._public_video_url_parameter.add_input_parameters()

        self.add_parameter(
            ParameterInt(
                name="selected_face_index",
                default_value=0,
                tooltip="Zero-based index of the face to expose for downstream advanced lip-sync nodes.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"display_name": "selected face index"},
            )
        )

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
            ParameterString(
                name="session_id",
                tooltip="Kling face-identification session id",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )
        self.add_parameter(
            Parameter(
                name="face_data",
                type="list[dict]",
                output_type="list[dict]",
                default_value=[],
                tooltip="Detected face metadata list returned by Kling.",
                allowed_modes={ParameterMode.OUTPUT},
                hide_property=True,
            )
        )
        self.add_parameter(
            ParameterDict(
                name="selected_face",
                tooltip="Selected face metadata for wiring into Kling Advanced Lip Sync.",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )
        self.add_parameter(
            ParameterString(
                name="selected_face_id",
                tooltip="Selected face id",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

        self._create_status_parameters(
            result_details_tooltip="Details about the face-identification result or any errors",
            result_details_placeholder="Face-identification status and details will appear here.",
            parameter_group_initially_collapsed=True,
        )
        self._update_video_input_visibility(self.get_parameter_value("video_input_type"))

    async def aprocess(self) -> None:
        try:
            await super().aprocess()
        finally:
            self._public_video_url_parameter.delete_uploaded_artifact()

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        super().after_value_set(parameter, value)

        if parameter.name == "video_input_type":
            self._update_video_input_visibility(value)
        if parameter.name == "selected_face_index":
            self._update_selected_face_output()

    def _update_video_input_visibility(self, value: Any) -> None:
        if value == "video_id":
            self.show_parameter_by_name("video_id")
            self.hide_parameter_by_name("video_url")
            self.hide_message_by_name("artifact_url_parameter_message_video_url")
            return

        self.hide_parameter_by_name("video_id")
        self.show_parameter_by_name("video_url")
        self.show_message_by_name("artifact_url_parameter_message_video_url")

    def _get_api_model_id(self) -> str:
        return "kling:identify-face"

    async def _build_payload(self) -> dict[str, Any]:
        video_input_type = self.get_parameter_value("video_input_type") or "video_url"

        if video_input_type == "video_id":
            video_id = (self.get_parameter_value("video_id") or "").strip()
            return {"video_id": video_id}

        video_url = self._public_video_url_parameter.get_public_url_for_parameter()
        return {"video_url": (video_url or "").strip()}

    async def _parse_result(self, result_json: dict[str, Any], generation_id: str) -> None:
        data = result_json.get("data", {})
        session_id = data.get("session_id")
        face_data = data.get("face_data")

        if not session_id:
            self._set_safe_defaults()
            self._set_status_results(
                was_successful=False,
                result_details=f"{self.name} completed but did not return a session_id.",
            )
            return

        if not isinstance(face_data, list):
            self._set_safe_defaults()
            self._set_status_results(
                was_successful=False,
                result_details=f"{self.name} completed but did not return face_data.",
            )
            return

        self.parameter_output_values["session_id"] = session_id
        self.parameter_output_values["face_data"] = face_data
        self._update_selected_face_output()

        if not face_data:
            self._set_status_results(
                was_successful=False,
                result_details="Face identification completed successfully, but Kling did not detect any usable faces.",
            )
            return

        selected_face_id = self.parameter_output_values.get("selected_face_id") or ""
        self._set_status_results(
            was_successful=True,
            result_details=(
                f"Face identification completed successfully for generation {generation_id}. "
                f"Detected {len(face_data)} face(s); selected face id: {selected_face_id}."
            ),
        )

    def _update_selected_face_output(self) -> None:
        face_data = self.parameter_output_values.get("face_data") or []
        if not isinstance(face_data, list) or not face_data:
            self.parameter_output_values["selected_face"] = {}
            self.parameter_output_values["selected_face_id"] = ""
            return

        raw_index = self.get_parameter_value("selected_face_index")
        try:
            selected_index = int(raw_index)
        except (TypeError, ValueError):
            selected_index = 0

        if selected_index < 0 or selected_index >= len(face_data):
            self.parameter_output_values["selected_face"] = {}
            self.parameter_output_values["selected_face_id"] = ""
            return

        selected_face = face_data[selected_index]
        if not isinstance(selected_face, dict):
            self.parameter_output_values["selected_face"] = {}
            self.parameter_output_values["selected_face_id"] = ""
            return

        self.parameter_output_values["selected_face"] = selected_face
        self.parameter_output_values["selected_face_id"] = str(selected_face.get("face_id") or "")

    def _set_safe_defaults(self) -> None:
        self.parameter_output_values["session_id"] = ""
        self.parameter_output_values["face_data"] = []
        self.parameter_output_values["selected_face"] = {}
        self.parameter_output_values["selected_face_id"] = ""

    def validate_before_node_run(self) -> list[Exception] | None:
        exceptions = super().validate_before_node_run() or []

        video_input_type = self.get_parameter_value("video_input_type") or "video_url"
        selected_face_index = self.get_parameter_value("selected_face_index") or 0

        if video_input_type == "video_id":
            video_id = (self.get_parameter_value("video_id") or "").strip()
            if not video_id:
                exceptions.append(ValueError(f"{self.name} requires video_id when video input type is video_id."))
        else:
            video_url = self.get_parameter_value("video_url")
            if not video_url:
                exceptions.append(
                    ValueError(f"{self.name} requires a video file or URL when video input type is video_url.")
                )

        try:
            if int(selected_face_index) < 0:
                exceptions.append(ValueError(f"{self.name} selected_face_index must be 0 or greater."))
        except (TypeError, ValueError):
            exceptions.append(ValueError(f"{self.name} selected_face_index must be an integer."))

        return exceptions if exceptions else None
