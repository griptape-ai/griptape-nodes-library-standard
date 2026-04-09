from __future__ import annotations

import logging
from typing import Any, ClassVar

from griptape_nodes.exe_types.core_types import ParameterGroup, ParameterMode
from griptape_nodes.exe_types.param_types.parameter_dict import ParameterDict
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.traits.options import Options

from griptape_nodes_library.griptape_proxy_node import GriptapeProxyNode

logger = logging.getLogger("griptape_nodes")

__all__ = ["CartwheelMotionGeneration"]

TEXT_MODE = "Text Prompt"
VIDEO_MODE = "Reference Video"

MODE_TO_MODEL_ID: dict[str, str] = {
    TEXT_MODE: "cartwheel-motion-text",
    VIDEO_MODE: "cartwheel-motion-video",
}

REQUESTED_MODEL_OPTIONS = ["scoot", "swing"]
EXPORT_TYPE_OPTIONS = [
    "bvh",
    "fbx-blender",
    "fbx-maya",
    "fbx-unreal",
    "fbx-roblox",
    "glb",
    "ma",
    "mb",
]
AXIS_OPTIONS = ["Z", "Y", "X", "-Z", "-Y", "-X"]
FRAME_RATE_OPTIONS = [2, 12, 23.976, 24, 25, 29.97, 30, 48, 50, 59.94, 60, 90, 100, 120, 144, 240]
KEYFRAME_CLEANING_OPTIONS = ["none", "reduce", "simplify"]
HAND_POSE_OPTIONS = [
    "default",
    "claw",
    "curled",
    "fist",
    "fist_clenched",
    "fist_relaxed",
    "gripping",
    "love_you",
    "open_loose",
    "open_tight",
    "peace",
    "pointing",
    "pointing_cool",
    "relaxed",
    "rock",
    "splay",
    "thumbs_up",
]


class CartwheelMotionGeneration(GriptapeProxyNode):
    """Create a Cartwheel motion from text or a previously uploaded reference video.

    This node targets the Griptape proxy contract, which mirrors Cartwheel's request body.
    For reference-video mode, Cartwheel requires a previously uploaded Cartwheel media ID.
    """

    CATEGORY: ClassVar[str] = "API Nodes"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.category = self.CATEGORY
        self.description = "Create a motion with Cartwheel via Griptape Cloud model proxy"

        self.add_parameter(
            ParameterString(
                name="mode",
                default_value=TEXT_MODE,
                tooltip="Choose whether to generate motion from text or a Cartwheel reference video media ID",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=[TEXT_MODE, VIDEO_MODE])},
            )
        )

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
                name="character_id",
                default_value="",
                tooltip="Cartwheel character ID to export the motion on",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )

        self.add_parameter(
            ParameterString(
                name="prompt",
                default_value="",
                tooltip="Text prompt for motion generation",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                multiline=True,
                placeholder_text="Describe the motion...",
            )
        )

        self.add_parameter(
            ParameterString(
                name="reference_media_id",
                default_value="",
                tooltip="Cartwheel media ID for the uploaded reference video",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )

        with ParameterGroup(name="Generation Settings") as generation_settings_group:
            ParameterString(
                name="requested_model",
                default_value="swing",
                tooltip="Cartwheel text-motion model",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=REQUESTED_MODEL_OPTIONS)},
            )

        self.add_node_element(generation_settings_group)

        with ParameterGroup(name="Export Settings", ui_options={"collapsed": True}) as export_settings_group:
            ParameterString(
                name="export_type",
                default_value="bvh",
                tooltip="Cartwheel export type",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=EXPORT_TYPE_OPTIONS)},
            )

            ParameterString(
                name="forward",
                default_value="Z",
                tooltip="Forward axis",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=AXIS_OPTIONS)},
            )

            ParameterString(
                name="up",
                default_value="Y",
                tooltip="Up axis",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=AXIS_OPTIONS)},
            )

            ParameterString(
                name="frame_rate",
                default_value="24",
                tooltip="Output frame rate",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=[str(option) for option in FRAME_RATE_OPTIONS])},
            )

            ParameterString(
                name="frame_step_size",
                default_value="1",
                tooltip="Frame step size",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )

            ParameterString(
                name="move_in_place",
                default_value="false",
                tooltip="Whether to move the character in place",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=["false", "true"])},
            )

            ParameterString(
                name="loop",
                default_value="false",
                tooltip="Whether to loop the motion",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=["false", "true"])},
            )

            ParameterString(
                name="face_expression",
                default_value="",
                tooltip="Optional Cartwheel face expression",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )

            ParameterString(
                name="hand_pose",
                default_value="",
                tooltip="Optional hand pose",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=[""] + HAND_POSE_OPTIONS)},
            )

            ParameterString(
                name="ik_feet",
                default_value="",
                tooltip="Optional inverse kinematics setting for feet",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=["", "true", "false"])},
            )

            ParameterString(
                name="ik_hands",
                default_value="",
                tooltip="Optional inverse kinematics setting for hands",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=["", "true", "false"])},
            )

            ParameterString(
                name="include_mesh",
                default_value="",
                tooltip="Optional include-mesh override",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=["", "true", "false"])},
            )

            ParameterString(
                name="keyframe_cleaning",
                default_value="",
                tooltip="Optional keyframe cleaning method",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=[""] + KEYFRAME_CLEANING_OPTIONS)},
            )

            ParameterString(
                name="skin_hex",
                default_value="",
                tooltip="Optional skin hex override",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )

        self.add_node_element(export_settings_group)

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
                name="batch_id",
                tooltip="Cartwheel batch ID",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

        self.add_parameter(
            ParameterString(
                name="motion_id",
                tooltip="Cartwheel motion ID",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

        self.add_parameter(
            ParameterString(
                name="bvh_url",
                tooltip="Cartwheel BVH download URL",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

        self.add_parameter(
            ParameterString(
                name="cwms_url",
                tooltip="Cartwheel CWMS download URL",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

        self.add_parameter(
            ParameterString(
                name="fbx_url",
                tooltip="Cartwheel FBX download URL",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

        self.add_parameter(
            ParameterString(
                name="gltf_url",
                tooltip="Cartwheel GLTF download URL",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

        self._create_status_parameters(
            result_details_tooltip="Details about the motion generation result or any errors",
            result_details_placeholder="Generation status and details will appear here.",
            parameter_group_initially_collapsed=True,
        )

        self._sync_visibility(TEXT_MODE)

    def after_value_set(self, parameter, value: Any) -> None:
        if parameter.name == "mode":
            self._sync_visibility(value)

    def _sync_visibility(self, mode: str) -> None:
        is_text_mode = mode == TEXT_MODE
        if is_text_mode:
            self.show_parameter_by_name("prompt")
            self.show_parameter_by_name("requested_model")
            self.hide_parameter_by_name("reference_media_id")
        else:
            self.hide_parameter_by_name("prompt")
            self.hide_parameter_by_name("requested_model")
            self.show_parameter_by_name("reference_media_id")

    def _get_api_model_id(self) -> str:
        mode = self.get_parameter_value("mode") or TEXT_MODE
        return MODE_TO_MODEL_ID.get(mode, MODE_TO_MODEL_ID[TEXT_MODE])

    async def _build_payload(self) -> dict[str, Any]:
        export_settings: dict[str, Any] = {
            "characterID": self._required_string("character_id"),
            "exportType": self.get_parameter_value("export_type"),
            "forward": self.get_parameter_value("forward"),
            "up": self.get_parameter_value("up"),
            "frameRate": self._parse_float_string("frame_rate", as_int_if_possible=False),
            "frameStepSize": self._parse_float_string("frame_step_size", as_int_if_possible=True),
            "moveInPlace": self._parse_bool_string("move_in_place"),
        }

        self._add_optional_string(export_settings, "faceExpression", "face_expression")
        self._add_optional_string(export_settings, "handPose", "hand_pose")
        self._add_optional_bool(export_settings, "ikFeet", "ik_feet")
        self._add_optional_bool(export_settings, "ikHands", "ik_hands")
        self._add_optional_bool(export_settings, "includeMesh", "include_mesh")
        self._add_optional_string(export_settings, "keyframeCleaning", "keyframe_cleaning")
        self._add_optional_string(export_settings, "skinHex", "skin_hex")
        loop = self.get_parameter_value("loop")
        if loop in {"true", "false"}:
            export_settings["loop"] = self._parse_bool_string("loop")

        payload: dict[str, Any] = {
            "exportSettings": export_settings,
        }

        batch_name = self.get_parameter_value("batch_name")
        if batch_name:
            payload["batchName"] = batch_name

        mode = self.get_parameter_value("mode") or TEXT_MODE
        if mode == TEXT_MODE:
            payload["prompts"] = [self._required_string("prompt")]
            payload["requestedModel"] = self.get_parameter_value("requested_model")
        else:
            payload["mediaIDs"] = [self._required_string("reference_media_id")]

        return payload

    async def _parse_result(self, result_json: dict[str, Any], generation_id: str) -> None:
        batch = result_json.get("batch", {})
        motions = result_json.get("motions", [])

        if not isinstance(batch, dict) or not isinstance(motions, list) or not motions:
            self._set_safe_defaults()
            self._set_status_results(
                was_successful=False,
                result_details=f"{self.name} completed but returned no motion items.",
            )
            return

        first_motion = motions[0] if isinstance(motions[0], dict) else {}
        batch_id = batch.get("batchID") if isinstance(batch.get("batchID"), str) else ""
        motion_id = first_motion.get("motionID") if isinstance(first_motion.get("motionID"), str) else ""

        self.parameter_output_values["batch_id"] = batch_id
        self.parameter_output_values["motion_id"] = motion_id
        self.parameter_output_values["bvh_url"] = self._string_or_empty(first_motion.get("bvhURL"))
        self.parameter_output_values["cwms_url"] = self._string_or_empty(first_motion.get("cwmsURL"))
        self.parameter_output_values["fbx_url"] = self._string_or_empty(first_motion.get("fbxURL"))
        self.parameter_output_values["gltf_url"] = self._string_or_empty(first_motion.get("gltfURL"))

        download_count = sum(
            1
            for key in ("bvh_url", "cwms_url", "fbx_url", "gltf_url")
            if self.parameter_output_values[key]
        )
        if download_count == 0:
            self._set_status_results(
                was_successful=False,
                result_details=f"{self.name} completed but Cartwheel returned no downloadable motion URLs.",
            )
            return

        self._set_status_results(
            was_successful=True,
            result_details=(
                f"Motion generated successfully. "
                f"Batch ID: {batch_id or 'unknown'}. Motion ID: {motion_id or 'unknown'}."
            ),
        )

    def _set_safe_defaults(self) -> None:
        self.parameter_output_values["provider_response"] = None
        self.parameter_output_values["batch_id"] = ""
        self.parameter_output_values["motion_id"] = ""
        self.parameter_output_values["bvh_url"] = ""
        self.parameter_output_values["cwms_url"] = ""
        self.parameter_output_values["fbx_url"] = ""
        self.parameter_output_values["gltf_url"] = ""

    def validate_before_node_run(self) -> list[Exception] | None:
        exceptions = super().validate_before_node_run() or []

        character_id = self.get_parameter_value("character_id") or ""
        if not character_id.strip():
            exceptions.append(ValueError(f"{self.name} requires a Cartwheel character ID."))

        mode = self.get_parameter_value("mode") or TEXT_MODE
        if mode == TEXT_MODE:
            prompt = self.get_parameter_value("prompt") or ""
            if not prompt.strip():
                exceptions.append(ValueError(f"{self.name} requires a prompt in text mode."))
        else:
            reference_media_id = self.get_parameter_value("reference_media_id") or ""
            if not reference_media_id.strip():
                exceptions.append(
                    ValueError(
                        f"{self.name} requires a Cartwheel reference video media ID in reference-video mode."
                    )
                )

        return exceptions or None

    def _required_string(self, parameter_name: str) -> str:
        value = self.get_parameter_value(parameter_name) or ""
        value = str(value).strip()
        if not value:
            raise ValueError(f"{self.name} requires {parameter_name}.")
        return value

    def _parse_bool_string(self, parameter_name: str) -> bool:
        value = str(self.get_parameter_value(parameter_name) or "").strip().lower()
        if value not in {"true", "false"}:
            raise ValueError(f"{self.name} parameter {parameter_name} must be 'true' or 'false'.")
        return value == "true"

    def _parse_float_string(self, parameter_name: str, *, as_int_if_possible: bool) -> int | float:
        raw_value = str(self.get_parameter_value(parameter_name) or "").strip()
        try:
            parsed = float(raw_value)
        except ValueError as e:
            raise ValueError(f"{self.name} parameter {parameter_name} must be numeric.") from e

        if as_int_if_possible and parsed.is_integer():
            return int(parsed)
        return parsed

    def _add_optional_string(self, payload: dict[str, Any], key: str, parameter_name: str) -> None:
        value = str(self.get_parameter_value(parameter_name) or "").strip()
        if value:
            payload[key] = value

    def _add_optional_bool(self, payload: dict[str, Any], key: str, parameter_name: str) -> None:
        raw_value = str(self.get_parameter_value(parameter_name) or "").strip().lower()
        if raw_value in {"true", "false"}:
            payload[key] = raw_value == "true"

    @staticmethod
    def _string_or_empty(value: Any) -> str:
        return value if isinstance(value, str) else ""
