from __future__ import annotations

import logging
from typing import Any

from griptape.artifacts.video_url_artifact import VideoUrlArtifact
from griptape_nodes.exe_types.core_types import ParameterList, ParameterMode
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_types.parameter_dict import ParameterDict
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.exe_types.param_types.parameter_video import ParameterVideo

from griptape_nodes_library.griptape_proxy_node import GriptapeProxyNode

logger = logging.getLogger("griptape_nodes")

__all__ = ["CartwheelMascotGeneration"]


class CartwheelMascotGeneration(GriptapeProxyNode):
    """Create a Cartwheel mascot job from previously uploaded Cartwheel media IDs."""

    MODEL_ID = "cartwheel-mascot"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.category = "API Nodes"
        self.description = "Create a mascot replacement video with Cartwheel via Griptape Cloud model proxy"

        self.add_parameter(
            ParameterString(
                name="batch_name",
                default_value="",
                tooltip="Optional Cartwheel batch name",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )

        self.add_parameter(
            ParameterString(
                name="input_video_media_id",
                default_value="",
                tooltip="Cartwheel media ID for the uploaded input video",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )

        self.add_parameter(
            ParameterList(
                name="character_reference_media_ids",
                input_types=["str"],
                default_value=[],
                tooltip="One or more Cartwheel media IDs for the reference images",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"expander": True},
            )
        )

        self.add_parameter(
            ParameterString(
                name="prompt",
                default_value="",
                tooltip="Prompt describing the character replacement",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                multiline=True,
                placeholder_text="Describe the mascot replacement...",
            )
        )

        self.add_parameter(
            ParameterString(
                name="environment_image_media_id",
                default_value="",
                tooltip="Optional Cartwheel media ID for the environment image",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )

        self.add_parameter(
            ParameterString(
                name="mascot_name",
                default_value="",
                tooltip="Optional Cartwheel mascot job name",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )

        self.add_parameter(
            ParameterString(
                name="generation_id",
                tooltip="Griptape generation ID",
                allowed_modes={ParameterMode.OUTPUT},
                hide=True,
            )
        )

        self.add_parameter(
            ParameterDict(
                name="provider_response",
                tooltip="Verbatim result returned from the Griptape proxy",
                allowed_modes={ParameterMode.OUTPUT},
                hide=True,
                hide_property=True,
            )
        )

        self.add_parameter(
            ParameterString(
                name="mascot_id",
                tooltip="Cartwheel mascot job ID",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

        self.add_parameter(
            ParameterVideo(
                name="video_url",
                tooltip="Generated mascot video",
                allowed_modes={ParameterMode.OUTPUT, ParameterMode.PROPERTY},
                settable=False,
                ui_options={"pulse_on_run": True},
            )
        )

        self._output_file = ProjectFileParameter(
            node=self,
            name="output_file",
            default_filename="cartwheel_mascot.mp4",
        )
        self._output_file.add_parameter()

        self._create_status_parameters(
            result_details_tooltip="Details about the mascot generation result or any errors",
            result_details_placeholder="Generation status and details will appear here.",
            parameter_group_initially_collapsed=True,
        )

    def _get_api_model_id(self) -> str:
        return self.MODEL_ID

    async def _build_payload(self) -> dict[str, Any]:
        job: dict[str, Any] = {
            "inputVideoMediaID": self._required_string("input_video_media_id"),
            "characterReferenceMediaIDs": self._required_string_list("character_reference_media_ids"),
            "prompt": self._required_string("prompt"),
        }

        environment_image_media_id = self._optional_string("environment_image_media_id")
        if environment_image_media_id:
            job["environmentImageMediaID"] = environment_image_media_id

        mascot_name = self._optional_string("mascot_name")
        if mascot_name:
            job["mascotName"] = mascot_name

        payload: dict[str, Any] = {"jobs": [job]}

        batch_name = self._optional_string("batch_name")
        if batch_name:
            payload["batchName"] = batch_name

        return payload

    async def _parse_result(self, result_json: dict[str, Any], generation_id: str) -> None:
        items = result_json.get("items", [])
        if not isinstance(items, list) or not items or not isinstance(items[0], dict):
            self._set_safe_defaults()
            self._set_status_results(
                was_successful=False,
                result_details=f"{self.name} completed but returned no mascot job items.",
            )
            return

        first_item = items[0]
        mascot_id = first_item.get("mascotID")
        output_video_url = first_item.get("outputVideoURL")

        self.parameter_output_values["mascot_id"] = mascot_id if isinstance(mascot_id, str) else ""

        if not isinstance(output_video_url, str) or not output_video_url:
            self.parameter_output_values["video_url"] = None
            self._set_status_results(
                was_successful=False,
                result_details=f"{self.name} completed but Cartwheel returned no outputVideoURL.",
            )
            return

        video_bytes = await self._download_bytes_from_url(output_video_url)
        if video_bytes:
            try:
                dest = self._output_file.build_file()
                saved = await dest.awrite_bytes(video_bytes)
                self.parameter_output_values["video_url"] = VideoUrlArtifact(value=saved.location, name=saved.name)
                self._set_status_results(
                    was_successful=True,
                    result_details=f"Mascot video generated successfully and saved as {saved.name}.",
                )
                return
            except (OSError, PermissionError) as e:
                logger.warning("%s failed to save mascot video: %s", self.name, e)

        self.parameter_output_values["video_url"] = VideoUrlArtifact(value=output_video_url)
        self._set_status_results(
            was_successful=True,
            result_details="Mascot video generated successfully. Using provider URL.",
        )

    def _set_safe_defaults(self) -> None:
        self.parameter_output_values["provider_response"] = None
        self.parameter_output_values["mascot_id"] = ""
        self.parameter_output_values["video_url"] = None

    def validate_before_node_run(self) -> list[Exception] | None:
        exceptions = super().validate_before_node_run() or []

        if not (self.get_parameter_value("input_video_media_id") or "").strip():
            exceptions.append(ValueError(f"{self.name} requires an input video media ID."))

        if not self._required_string_list("character_reference_media_ids", raise_on_error=False):
            exceptions.append(ValueError(f"{self.name} requires at least one character reference media ID."))

        if not (self.get_parameter_value("prompt") or "").strip():
            exceptions.append(ValueError(f"{self.name} requires a prompt."))

        return exceptions or None

    def _required_string(self, parameter_name: str) -> str:
        value = self._optional_string(parameter_name)
        if not value:
            raise ValueError(f"{self.name} requires {parameter_name}.")
        return value

    def _optional_string(self, parameter_name: str) -> str:
        return str(self.get_parameter_value(parameter_name) or "").strip()

    def _required_string_list(self, parameter_name: str, *, raise_on_error: bool = True) -> list[str]:
        value = self.get_parameter_value(parameter_name) or []
        if not isinstance(value, list):
            if raise_on_error:
                raise ValueError(f"{self.name} parameter {parameter_name} must be a list.")
            return []

        normalized = [str(item).strip() for item in value if str(item).strip()]
        if not normalized and raise_on_error:
            raise ValueError(f"{self.name} requires at least one value for {parameter_name}.")
        return normalized
