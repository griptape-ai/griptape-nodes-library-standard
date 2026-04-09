from __future__ import annotations

import logging
from typing import Any, ClassVar

from griptape.artifacts.image_url_artifact import ImageUrlArtifact
from griptape_nodes.exe_types.core_types import ParameterMode
from griptape_nodes.exe_types.param_components.project_file_parameter import (
    ProjectFileParameter,
)
from griptape_nodes.exe_types.param_types.parameter_dict import ParameterDict
from griptape_nodes.exe_types.param_types.parameter_float import ParameterFloat
from griptape_nodes.exe_types.param_types.parameter_image import ParameterImage
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.traits.options import Options

from griptape_nodes_library.griptape_proxy_node import GriptapeProxyNode

logger = logging.getLogger("griptape_nodes")

__all__ = ["CartwheelCharacterGeneration"]

TEXT_MODE = "Text Prompt"
REFERENCE_IMAGE_MODE = "Reference Image Media ID"


class CartwheelCharacterGeneration(GriptapeProxyNode):
    """Generate a Cartwheel character via Griptape Cloud model proxy."""

    MODEL_ID: ClassVar[str] = "cartwheel-character"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.category = "API Nodes"
        self.description = (
            "Generate a Cartwheel character for downstream motion workflows via Griptape Cloud model proxy"
        )

        self.add_parameter(
            ParameterString(
                name="mode",
                default_value=TEXT_MODE,
                tooltip=(
                    "Choose between prompt-based character generation and using an "
                    "existing Cartwheel reference image media ID"
                ),
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=[TEXT_MODE, REFERENCE_IMAGE_MODE])},
            )
        )

        self.add_parameter(
            ParameterString(
                name="prompt",
                default_value="",
                tooltip="Prompt describing the character to generate",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                multiline=True,
                placeholder_text="Describe the Cartwheel character...",
            )
        )

        self.add_parameter(
            ParameterString(
                name="reference_media_id",
                default_value="",
                tooltip=(
                    "Cartwheel media ID for a previously uploaded reference image. "
                    "This node does not upload images to Cartwheel."
                ),
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )

        self.add_parameter(
            ParameterString(
                name="character_name",
                default_value="",
                tooltip="Optional name for the generated character (max 100 characters)",
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
                name="character_id",
                tooltip="Generated Cartwheel character ID",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

        self.add_parameter(
            ParameterString(
                name="upload_status",
                tooltip="Cartwheel upload pipeline status",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

        self.add_parameter(
            ParameterString(
                name="generated_status",
                tooltip="Cartwheel generated-character pipeline status",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

        self.add_parameter(
            ParameterFloat(
                name="estimated_seconds_wait_time",
                tooltip="Estimated seconds remaining reported by Cartwheel",
                default_value=0.0,
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

        self.add_parameter(
            ParameterImage(
                name="thumbnail_image",
                tooltip="Generated character thumbnail",
                allowed_modes={ParameterMode.OUTPUT, ParameterMode.PROPERTY},
                settable=False,
                ui_options={"pulse_on_run": True},
            )
        )

        self._thumbnail_file = ProjectFileParameter(
            node=self,
            name="thumbnail_file",
            default_filename="cartwheel_character_thumbnail.png",
        )
        self._thumbnail_file.add_parameter()

        self._create_status_parameters(
            result_details_tooltip=("Details about the character generation result or any errors"),
            result_details_placeholder="Generation status and details will appear here.",
            parameter_group_initially_collapsed=True,
        )

        self._sync_visibility(TEXT_MODE)

    def after_value_set(self, parameter: Any, value: Any) -> None:
        if parameter.name == "mode":
            self._sync_visibility(value)
        return super().after_value_set(parameter, value)

    def _sync_visibility(self, mode: str) -> None:
        if mode == REFERENCE_IMAGE_MODE:
            self.hide_parameter_by_name("prompt")
            self.show_parameter_by_name("reference_media_id")
            return

        self.show_parameter_by_name("prompt")
        self.hide_parameter_by_name("reference_media_id")

    def _get_api_model_id(self) -> str:
        return self.MODEL_ID

    async def _build_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}

        character_name = self._optional_string("character_name")
        if character_name:
            payload["characterName"] = character_name

        mode = self.get_parameter_value("mode") or TEXT_MODE
        if mode == REFERENCE_IMAGE_MODE:
            payload["mediaID"] = self._required_string("reference_media_id")
        else:
            payload["prompt"] = self._required_string("prompt")

        return payload

    async def _parse_result(self, result_json: dict[str, Any], _generation_id: str) -> None:
        character = result_json.get("character", {})
        if not isinstance(character, dict):
            self._set_safe_defaults()
            self._set_status_results(
                was_successful=False,
                result_details=(f"{self.name} completed but returned no Cartwheel character payload."),
            )
            return

        character_id = self._string_or_empty(character.get("characterID"))
        if not character_id:
            self._set_safe_defaults()
            self._set_status_results(
                was_successful=False,
                result_details=(f"{self.name} completed but Cartwheel returned no characterID."),
            )
            return

        self.parameter_output_values["character_id"] = character_id
        self.parameter_output_values["upload_status"] = self._string_or_empty(character.get("uploadStatus"))
        self.parameter_output_values["generated_status"] = self._string_or_empty(character.get("generatedStatus"))
        self.parameter_output_values["estimated_seconds_wait_time"] = float(
            character.get("estimatedSecondsWaitTime") or 0
        )

        thumbnail_url = self._string_or_empty(character.get("thumbnailURL"))
        if not thumbnail_url:
            self.parameter_output_values["thumbnail_image"] = None
            self._set_status_results(
                was_successful=True,
                result_details=(
                    f"Character generated successfully. Character ID: {character_id}. Cartwheel returned no thumbnail."
                ),
            )
            return

        thumbnail_bytes = await self._download_bytes_from_url(thumbnail_url)
        if thumbnail_bytes:
            try:
                dest = self._thumbnail_file.build_file()
                saved = await dest.awrite_bytes(thumbnail_bytes)
                self.parameter_output_values["thumbnail_image"] = ImageUrlArtifact(
                    value=saved.location,
                    name=saved.name,
                )
                self._set_status_results(
                    was_successful=True,
                    result_details=(
                        "Character generated successfully. "
                        f"Character ID: {character_id}. Thumbnail saved as {saved.name}."
                    ),
                )
                return
            except (OSError, PermissionError) as e:
                logger.warning("%s failed to save Cartwheel thumbnail: %s", self.name, e)

        self.parameter_output_values["thumbnail_image"] = ImageUrlArtifact(value=thumbnail_url)
        self._set_status_results(
            was_successful=True,
            result_details=(
                f"Character generated successfully. Character ID: {character_id}. Using provider thumbnail URL."
            ),
        )

    def _set_safe_defaults(self) -> None:
        self.parameter_output_values["provider_response"] = None
        self.parameter_output_values["character_id"] = ""
        self.parameter_output_values["upload_status"] = ""
        self.parameter_output_values["generated_status"] = ""
        self.parameter_output_values["estimated_seconds_wait_time"] = 0.0
        self.parameter_output_values["thumbnail_image"] = None

    def validate_before_node_run(self) -> list[Exception] | None:
        exceptions = super().validate_before_node_run() or []

        mode = self.get_parameter_value("mode") or TEXT_MODE
        if mode == REFERENCE_IMAGE_MODE:
            if not self._optional_string("reference_media_id"):
                exceptions.append(
                    ValueError(f"{self.name} requires a Cartwheel reference image media ID in reference-image mode.")
                )
        elif not self._optional_string("prompt"):
            exceptions.append(ValueError(f"{self.name} requires a prompt in text mode."))

        character_name = self._optional_string("character_name")
        if len(character_name) > 100:
            exceptions.append(ValueError(f"{self.name} character_name must be 100 characters or fewer."))

        return exceptions or None

    def _required_string(self, parameter_name: str) -> str:
        value = self._optional_string(parameter_name)
        if not value:
            raise ValueError(f"{self.name} requires {parameter_name}.")
        return value

    def _optional_string(self, parameter_name: str) -> str:
        return str(self.get_parameter_value(parameter_name) or "").strip()

    @staticmethod
    def _string_or_empty(value: Any) -> str:
        return value if isinstance(value, str) else ""
