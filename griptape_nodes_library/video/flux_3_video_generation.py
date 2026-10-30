from __future__ import annotations

import logging
from contextlib import suppress
from enum import StrEnum
from typing import Any, ClassVar

from griptape.artifacts import ImageArtifact, ImageUrlArtifact
from griptape.artifacts.video_url_artifact import VideoUrlArtifact
from griptape_nodes.exe_types.core_types import NodeError, Parameter, ParameterGroup, ParameterList, ParameterMode
from griptape_nodes.exe_types.param_components.model_access_component import ModelAccessComponent
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_types.parameter_bool import ParameterBool
from griptape_nodes.exe_types.param_types.parameter_dict import ParameterDict
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.exe_types.param_types.parameter_video import ParameterVideo
from griptape_nodes.files.file import File, FileLoadError
from griptape_nodes.traits.options import Options
from griptape_nodes.utils.artifact_normalization import normalize_artifact_list

from griptape_nodes_library.media import coerce_media_url_or_data_uri, prepare_media_data_uri
from griptape_nodes_library.proxy import ArtifactKind, GriptapeProxyNode

logger = logging.getLogger("griptape_nodes")

__all__ = ["Flux3VideoGeneration"]

DEFAULT_MODEL = "flux-3-video"
MODEL_OPTIONS = [DEFAULT_MODEL]
MAX_KEYFRAMES = 10
MIN_DURATION_SEC = 5
MAX_T2V_DURATION_SEC = 20
MAX_V2V_DURATION_SEC = 15


class Flux3VideoMode(StrEnum):
    T2V = "t2v"
    I2V = "i2v"
    V2V = "v2v"


class Flux3VideoResolution(StrEnum):
    HD = "hd"
    FHD = "fhd"
    QHD = "qhd"
    UHD = "uhd"


class Flux3SafetyTolerance(StrEnum):
    LEAST_RESTRICTIVE = "least restrictive"
    MODERATE = "moderate"
    MOST_RESTRICTIVE = "most restrictive"


LEGACY_MODEL_VALUES: dict[str, str] = {
    "FLUX 3 Video": DEFAULT_MODEL,
    "gtc_flux_3_video": DEFAULT_MODEL,
}

MODE_OPTIONS = [
    Flux3VideoMode.T2V.value,
    Flux3VideoMode.I2V.value,
    Flux3VideoMode.V2V.value,
]
ASPECT_RATIO_OPTIONS = [
    "auto",
    "21:9",
    "2:1",
    "16:9",
    "4:3",
    "1:1",
    "3:4",
    "9:16",
    "9:21",
]
RESOLUTION_OPTIONS = [
    Flux3VideoResolution.HD.value,
    Flux3VideoResolution.FHD.value,
    Flux3VideoResolution.QHD.value,
    Flux3VideoResolution.UHD.value,
]
DURATION_OPTIONS_T2V = ["auto", *[str(s) for s in range(MIN_DURATION_SEC, MAX_T2V_DURATION_SEC + 1)]]
DURATION_OPTIONS_V2V = ["auto", *[str(s) for s in range(MIN_DURATION_SEC, MAX_V2V_DURATION_SEC + 1)]]
SAFETY_TOLERANCE_OPTIONS = [
    Flux3SafetyTolerance.LEAST_RESTRICTIVE.value,
    Flux3SafetyTolerance.MODERATE.value,
    Flux3SafetyTolerance.MOST_RESTRICTIVE.value,
]


class Flux3VideoGeneration(GriptapeProxyNode):
    """Generate videos with synchronized audio using Black Forest Labs FLUX 3 Video via Griptape model proxy.

    Inputs:
        - model (str): FLUX 3 video model ("flux-3-video")
        - mode (str): Generation mode: "t2v" (text-to-video), "i2v" (image-to-video keyframes), or "v2v" (video continuation)
        - prompt (str): Text description of the desired video
        - keyframes (list): [i2v only] 1 to 10 keyframe images (first starts the video, second ends it, additional frames are spaced evenly)
        - start_video (VideoUrlArtifact): [v2v only] Source video clip to continue from
        - aspect_ratio (str): Output aspect ratio ("auto" or preset ratio)
        - duration (str): Video duration in seconds ("auto", or 5-20 for t2v/i2v and 5-15 for v2v)
        - resolution (str): Output resolution class ("hd", "fhd", "qhd", or "uhd")
        - generate_audio (bool): Generate synchronized audio alongside the video (default: True)
        - draft (bool): Render a fast HD draft preview (default: False)
        - safety_tolerance (str): Content moderation preset ("least restrictive", "moderate", or "most restrictive")

    Outputs:
        - generation_id (str): Generation ID from the API
        - provider_response (dict): Verbatim response from the model proxy
        - video_url (VideoUrlArtifact): Generated video
        - was_successful (bool): Whether the generation succeeded
        - result_details (str): Details about the generation result or error
    """

    LEGACY_MODEL_VALUES: ClassVar[dict[str, str]] = LEGACY_MODEL_VALUES

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.category = "API Nodes"
        self.description = "Generate videos with synchronized audio using FLUX 3 Video via Griptape model proxy"

        model_param = ParameterString(
            name="model",
            default_value=DEFAULT_MODEL,
            tooltip="Select the FLUX 3 video model to use",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
        )
        self.add_parameter(model_param)
        self._model_access = ModelAccessComponent(
            node=self,
            parameter=model_param,
            model_choices=MODEL_OPTIONS,
            default_model=DEFAULT_MODEL,
            deprecated_values=LEGACY_MODEL_VALUES,
        )

        self.add_parameter(
            ParameterString(
                name="mode",
                default_value=Flux3VideoMode.T2V.value,
                tooltip="Generation mode: 't2v' (Text to Video), 'i2v' (Image to Video keyframes), or 'v2v' (Video Continuation)",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=MODE_OPTIONS)},
                ui_options={"display_name": "Mode"},
            )
        )

        self.add_parameter(
            ParameterString(
                name="prompt",
                tooltip="Text description of the desired video and audio",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                multiline=True,
                placeholder_text="Describe the video scene, motion, and sound...",
                allow_output=False,
                ui_options={"display_name": "Prompt"},
            )
        )

        self.add_parameter(
            ParameterList(
                name="keyframes",
                input_types=[
                    "ImageArtifact",
                    "ImageUrlArtifact",
                    "str",
                    "list",
                    "list[ImageArtifact]",
                    "list[ImageUrlArtifact]",
                ],
                default_value=[],
                tooltip="1 to 10 keyframe images for Image-to-Video ('i2v') mode. 1 image pins the start frame; 2 pin start and end; 3+ space evenly across the duration.",
                allowed_modes={ParameterMode.INPUT},
                hide=True,
                ui_options={"expander": True, "display_name": "Keyframes", "hide_property": True},
            )
        )

        self.add_parameter(
            ParameterVideo(
                name="start_video",
                default_value="",
                tooltip="Video clip to continue from in Video Continuation ('v2v') mode",
                allowed_modes={ParameterMode.INPUT},
                hide_property=True,
                hide=True,
                ui_options={"display_name": "Start Video"},
            )
        )

        self.add_parameter(
            ParameterString(
                name="aspect_ratio",
                default_value="auto",
                tooltip="Output aspect ratio. 'auto' lets FLUX 3 choose from the prompt and any reference media.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=ASPECT_RATIO_OPTIONS)},
                ui_options={"display_name": "Aspect Ratio"},
            )
        )

        self.add_parameter(
            ParameterString(
                name="duration",
                default_value="auto",
                tooltip="Video duration in seconds ('auto', or 5-20s for t2v/i2v and 5-15s for v2v)",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=DURATION_OPTIONS_T2V)},
                ui_options={"display_name": "Duration"},
            )
        )

        self.add_parameter(
            ParameterString(
                name="resolution",
                default_value=Flux3VideoResolution.HD.value,
                tooltip="Output video resolution class ('hd', 'fhd', 'qhd', or 'uhd')",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=RESOLUTION_OPTIONS)},
                ui_options={"display_name": "Resolution"},
            )
        )

        with ParameterGroup(name="Generation Settings", ui_options={"collapsed": True}) as generation_settings_group:
            ParameterBool(
                name="generate_audio",
                default_value=True,
                tooltip="Generate synchronized audio alongside the video",
                allow_output=False,
            )

            ParameterBool(
                name="draft",
                default_value=False,
                tooltip="Render a fast preview at 'hd' resolution (~40% of standard HD cost)",
                allow_output=False,
            )

            ParameterString(
                name="safety_tolerance",
                default_value=Flux3SafetyTolerance.MODERATE.value,
                tooltip="Content moderation strictness level",
                allow_output=False,
                traits={Options(choices=SAFETY_TOLERANCE_OPTIONS)},
            )

        self.add_node_element(generation_settings_group)

        # OUTPUTS
        self.add_parameter(
            ParameterDict(
                name="provider_response",
                tooltip="Verbatim response from Griptape model proxy",
                allowed_modes={ParameterMode.OUTPUT},
                hide_property=True,
                hide=True,
            )
        )

        self.add_parameter(
            ParameterVideo(
                name="video_url",
                tooltip="Generated video as URL artifact",
                allowed_modes={ParameterMode.OUTPUT, ParameterMode.PROPERTY},
                settable=False,
                ui_options={"pulse_on_run": True},
            )
        )

        self._output_file = ProjectFileParameter(
            node=self,
            name="output_file",
            default_filename="flux_3_video.mp4",
        )
        self._output_file.add_parameter()

        self._create_status_parameters(
            result_details_tooltip="Details about the video generation result or any errors",
            result_details_placeholder="Generation status and details will appear here.",
            parameter_group_initially_collapsed=True,
        )

    def _log(self, message: str) -> None:
        with suppress(Exception):
            logger.info(message)

    def _sync_mode_visibility(self) -> None:
        mode = self.get_parameter_value("mode") or Flux3VideoMode.T2V.value
        match mode:
            case Flux3VideoMode.T2V:
                self.hide_parameter_by_name("keyframes")
                self.hide_parameter_by_name("start_video")
            case Flux3VideoMode.I2V:
                self.show_parameter_by_name("keyframes")
                self.hide_parameter_by_name("start_video")
            case Flux3VideoMode.V2V:
                self.hide_parameter_by_name("keyframes")
                self.show_parameter_by_name("start_video")
            case _:
                msg = f"Unknown FLUX 3 video mode: {mode!r}"
                raise ValueError(msg)

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        super().after_value_set(parameter, value)

        if parameter.name == "keyframes" and isinstance(value, list):
            updated_list = normalize_artifact_list(value, ImageUrlArtifact, accepted_types=(ImageArtifact,))
            if updated_list != value:
                self.set_parameter_value("keyframes", updated_list)

        if parameter.name == "mode" and value in MODE_OPTIONS:
            self._sync_mode_visibility()

    @staticmethod
    def _parse_safety_tolerance(value: str | None) -> int:
        if not value:
            msg = "Pick a 'safety_tolerance' value."
            raise NodeError(msg)

        match value:
            case Flux3SafetyTolerance.MOST_RESTRICTIVE:
                return 0
            case Flux3SafetyTolerance.MODERATE:
                return 2
            case Flux3SafetyTolerance.LEAST_RESTRICTIVE:
                return 4
            case _:
                msg = f"'safety_tolerance' must be one of {SAFETY_TOLERANCE_OPTIONS}. It's set to {value!r}."
                raise NodeError(msg)

    @staticmethod
    def _parse_duration(value: Any, *, mode: str) -> int | str:
        max_sec = MAX_V2V_DURATION_SEC if mode == Flux3VideoMode.V2V else MAX_T2V_DURATION_SEC
        if value is None or str(value).strip().lower() == "auto":
            return "auto"
        try:
            seconds = int(str(value).strip())
        except (TypeError, ValueError) as e:
            msg = f"'Duration' must be 'auto' or an integer between {MIN_DURATION_SEC} and {max_sec}. It's set to {value!r}."
            raise NodeError(msg) from e
        if not (MIN_DURATION_SEC <= seconds <= max_sec):
            msg = f"'Duration' for mode '{mode}' must be between {MIN_DURATION_SEC} and {max_sec} seconds, or 'auto'. It's set to {seconds}."
            raise NodeError(msg)
        return seconds

    def _raw_keyframes(self) -> list[Any]:
        raw = self.get_parameter_list_value("keyframes") or self.parameter_values.get("keyframes") or []
        if not isinstance(raw, list):
            raw = [raw] if raw else []
        return [item for item in raw if item is not None and item != ""]

    @staticmethod
    def _has_media_value(value: Any) -> bool:
        if value is None:
            return False
        if hasattr(value, "value"):
            return bool(value.value)
        return bool(value)

    async def _prepare_video_data_uri(self, video_input: Any) -> str:
        video_value = coerce_media_url_or_data_uri(video_input, kind="video")
        if not video_value:
            msg = "Connect a video to 'Start Video'."
            raise NodeError(msg)
        if video_value.startswith("data:"):
            return video_value
        try:
            return await File(video_value).aread_data_uri(fallback_mime="video/mp4")
        except FileLoadError as e:
            msg = f"Could not read the video connected to 'Start Video': {e}"
            raise NodeError(msg) from e

    def validate_before_node_run(self) -> list[Exception] | None:
        exceptions = super().validate_before_node_run() or []

        prompt = (self.get_parameter_value("prompt") or "").strip()
        if not prompt:
            exceptions.append(NodeError("Enter a prompt in 'Prompt' to generate a video."))

        mode = self.get_parameter_value("mode") or Flux3VideoMode.T2V.value
        if mode not in MODE_OPTIONS:
            exceptions.append(NodeError(f"'Mode' must be one of {MODE_OPTIONS}. It's set to {mode!r}."))
        else:
            try:
                parsed_duration = self._parse_duration(self.get_parameter_value("duration"), mode=mode)
            except NodeError as e:
                exceptions.append(e)
                parsed_duration = "auto"

            match mode:
                case Flux3VideoMode.T2V:
                    pass
                case Flux3VideoMode.I2V:
                    keyframes = self._raw_keyframes()
                    if not keyframes:
                        exceptions.append(NodeError("Connect at least one image to 'Keyframes' when 'Mode' is 'i2v'."))
                    elif len(keyframes) > MAX_KEYFRAMES:
                        exceptions.append(
                            NodeError(
                                f"'Keyframes' accepts at most {MAX_KEYFRAMES} images. {len(keyframes)} are connected."
                            )
                        )
                    elif len(keyframes) >= 3 and parsed_duration == "auto":
                        exceptions.append(
                            NodeError(
                                "When 3 or more images are connected to 'Keyframes', choose an explicit number of seconds in 'Duration' instead of 'auto'."
                            )
                        )
                case Flux3VideoMode.V2V:
                    if not self._has_media_value(self.get_parameter_value("start_video")):
                        exceptions.append(NodeError("Connect a video to 'Start Video' when 'Mode' is 'v2v'."))
                case _:
                    msg = f"Unknown FLUX 3 video mode: {mode!r}"
                    raise ValueError(msg)

        draft = bool(self.get_parameter_value("draft"))
        resolution = self.get_parameter_value("resolution") or Flux3VideoResolution.HD.value
        if draft and resolution != Flux3VideoResolution.HD:
            exceptions.append(
                NodeError(
                    "'draft' mode only supports 'hd' in 'Resolution'. Switch 'Resolution' to 'hd' or disable 'draft'."
                )
            )

        return exceptions or None

    async def _build_payload(self) -> dict[str, Any]:
        mode = self.get_parameter_value("mode") or Flux3VideoMode.T2V.value
        prompt = (self.get_parameter_value("prompt") or "").strip()
        aspect_ratio = self.get_parameter_value("aspect_ratio") or "auto"
        duration = self._parse_duration(self.get_parameter_value("duration"), mode=mode)
        resolution = self.get_parameter_value("resolution") or Flux3VideoResolution.HD.value
        generate_audio = self.get_parameter_value("generate_audio")
        draft = bool(self.get_parameter_value("draft"))
        safety_tolerance = self._parse_safety_tolerance(self.get_parameter_value("safety_tolerance"))

        payload: dict[str, Any] = {
            "mode": mode,
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "duration": duration,
            "resolution": resolution,
            "generate_audio": True if generate_audio is None else bool(generate_audio),
            "draft": draft,
            "safety_tolerance": safety_tolerance,
        }

        match mode:
            case Flux3VideoMode.T2V:
                pass
            case Flux3VideoMode.I2V:
                keyframes_list = normalize_artifact_list(
                    self._raw_keyframes(), ImageUrlArtifact, accepted_types=(ImageArtifact,)
                )
                encoded_keyframes: list[str] = []
                for image_input in keyframes_list[:MAX_KEYFRAMES]:
                    data_uri = await prepare_media_data_uri(image_input, kind="image", node_name=self.name)
                    if data_uri:
                        encoded_keyframes.append(data_uri)
                if not encoded_keyframes:
                    msg = "Connect at least one readable image to 'Keyframes'."
                    raise NodeError(msg)
                payload["keyframes"] = encoded_keyframes[0] if len(encoded_keyframes) == 1 else encoded_keyframes
            case Flux3VideoMode.V2V:
                payload["start_video"] = await self._prepare_video_data_uri(self.get_parameter_value("start_video"))
            case _:
                msg = f"Unknown FLUX 3 video mode: {mode!r}"
                raise ValueError(msg)

        return payload

    async def _parse_result(self, _result_json: dict[str, Any], generation_id: str) -> None:
        await self._save_generated_media(
            generation_id,
            "video_url",
            lambda v, n: VideoUrlArtifact(value=v, name=n),
            kind=ArtifactKind.VIDEO,
            media_kind="video",
        )

    def _set_safe_defaults(self) -> None:
        self.parameter_output_values["generation_id"] = ""
        self.parameter_output_values["provider_response"] = None
        self.parameter_output_values["video_url"] = None
