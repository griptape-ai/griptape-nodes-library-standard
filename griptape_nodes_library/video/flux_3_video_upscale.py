from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import urlsplit

from griptape.artifacts.video_url_artifact import VideoUrlArtifact
from griptape_nodes.exe_types.core_types import NodeError, ParameterGroup, ParameterMode
from griptape_nodes.exe_types.param_components.model_access_component import ModelAccessComponent
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_types.parameter_dict import ParameterDict
from griptape_nodes.exe_types.param_types.parameter_float import ParameterFloat
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.exe_types.param_types.parameter_video import ParameterVideo
from griptape_nodes.files.file import File, FileLoadError
from griptape_nodes.traits.options import Options

from griptape_nodes_library.media import coerce_media_url_or_data_uri
from griptape_nodes_library.proxy import ArtifactKind, GriptapeProxyNode
from griptape_nodes_library.utils.ffmpeg_utils import extract_video_metadata_structured

logger = logging.getLogger("griptape_nodes")

__all__ = ["Flux3VideoUpscale"]

DEFAULT_MODEL = "flux-video-upscale-v1"
MODEL_OPTIONS = [DEFAULT_MODEL]
MAX_INPUT_DURATION_SECONDS = 20.0
MIN_UPSCALE_FACTOR = 1.5
MAX_UPSCALE_FACTOR = 3.0
DEFAULT_UPSCALE_FACTOR = 2.0
SUPPORTED_CONTAINER = "mp4"


class Flux3UpscaleCreativity(StrEnum):
    PRECISE = "precise"
    CREATIVE = "creative"


class Flux3SafetyTolerance(StrEnum):
    LEAST_RESTRICTIVE = "least restrictive"
    MODERATE = "moderate"
    MOST_RESTRICTIVE = "most restrictive"


LEGACY_MODEL_VALUES: dict[str, str] = {
    "FLUX 3 Upscale": DEFAULT_MODEL,
    "gtc_flux_video_upscale_v1": DEFAULT_MODEL,
}

CREATIVITY_OPTIONS = [
    Flux3UpscaleCreativity.PRECISE.value,
    Flux3UpscaleCreativity.CREATIVE.value,
]
SAFETY_TOLERANCE_OPTIONS = [
    Flux3SafetyTolerance.LEAST_RESTRICTIVE.value,
    Flux3SafetyTolerance.MODERATE.value,
    Flux3SafetyTolerance.MOST_RESTRICTIVE.value,
]


class Flux3VideoUpscale(GriptapeProxyNode):
    """Upscale videos 1.5x-3x using Black Forest Labs FLUX 3 Video Upscale via Griptape model proxy.

    Inputs:
        - model (str): FLUX video upscale model ("flux-video-upscale-v1")
        - input_video (VideoUrlArtifact): Source MP4 video clip to upscale (up to 20 seconds, max 2560x1440)
        - upscale_factor (float): Output scaling relative to source resolution (1.5 to 3.0, default: 2.0)
        - creativity (str): "precise" (0, preserves source faithfully) or "creative" (1, enhances detail)
        - prompt (str): Optional description steering enhanced detail
        - safety_tolerance (str): Content moderation preset ("least restrictive", "moderate", or "most restrictive")

    Outputs:
        - generation_id (str): Generation ID from the API
        - provider_response (dict): Verbatim response from the model proxy
        - video_url (VideoUrlArtifact): Upscaled video
        - was_successful (bool): Whether the upscale succeeded
        - result_details (str): Details about the upscale result or error
    """

    LEGACY_MODEL_VALUES: ClassVar[dict[str, str]] = LEGACY_MODEL_VALUES

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.category = "API Nodes"
        self.description = "Upscale videos 1.5x-3x using FLUX 3 Video Upscale via Griptape model proxy"

        model_param = ParameterString(
            name="model",
            default_value=DEFAULT_MODEL,
            tooltip="Select the FLUX video upscale model to use",
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

        video_param = ParameterVideo(
            name="input_video",
            default_value="",
            tooltip="Source MP4 video clip to upscale (up to 20 seconds, max 2560x1440, max 50MB)",
            allowed_modes={ParameterMode.INPUT},
            hide_property=True,
            ui_options={"display_name": "Input Video"},
        )
        video_param.set_badge(
            variant="info",
            title="Video requirements",
            message=(
                f"- .mp4 file\n- {MAX_INPUT_DURATION_SECONDS:g} seconds or less\n- Up to 2560x1440 (3.7 MP per frame)\n\n"
                "Use a Trim Video or Resize Video node to prepare the clip."
            ),
        )
        self.add_parameter(video_param)

        self.add_parameter(
            ParameterFloat(
                name="upscale_factor",
                default_value=DEFAULT_UPSCALE_FACTOR,
                tooltip="Output scaling relative to the source resolution (1.5 to 3.0, capped at 13.75 MP frame)",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                slider=True,
                min_val=MIN_UPSCALE_FACTOR,
                max_val=MAX_UPSCALE_FACTOR,
                ui_options={"display_name": "Upscale Factor"},
            )
        )

        self.add_parameter(
            ParameterString(
                name="creativity",
                default_value=Flux3UpscaleCreativity.CREATIVE.value,
                tooltip="'precise' preserves the source faithfully; 'creative' allows detail enhancement",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=CREATIVITY_OPTIONS)},
                ui_options={"display_name": "Creativity"},
            )
        )

        self.add_parameter(
            ParameterString(
                name="prompt",
                default_value="",
                tooltip="Optional description of the clip's content to steer detail enhancement (leave empty for neutral upscale)",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                multiline=True,
                placeholder_text="Optional description to guide enhanced detail...",
                allow_output=False,
                ui_options={"display_name": "Prompt"},
            )
        )

        with ParameterGroup(name="Generation Settings", ui_options={"collapsed": True}) as generation_settings_group:
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
                tooltip="Upscaled video",
                allowed_modes={ParameterMode.OUTPUT, ParameterMode.PROPERTY},
                settable=False,
                ui_options={"pulse_on_run": True},
            )
        )

        self._output_file = ProjectFileParameter(
            node=self,
            name="output_file",
            default_filename="flux_3_video_upscale.mp4",
        )
        self._output_file.add_parameter()

        self._create_status_parameters(
            result_details_tooltip="Details about the video upscaling result or any errors",
            result_details_placeholder="Upscaling status and details will appear here.",
            parameter_group_initially_collapsed=True,
        )

    def _log(self, message: str) -> None:
        with suppress(Exception):
            logger.info(message)

    @staticmethod
    def _has_media_value(value: Any) -> bool:
        if value is None:
            return False
        if hasattr(value, "value"):
            return bool(value.value)
        return bool(value)

    @staticmethod
    def _container_token(video_value: str) -> str:
        if video_value.startswith("data:"):
            header = video_value.removeprefix("data:").split(",", 1)[0]
            return header.split(";", 1)[0].split("/", 1)[-1].lower()
        return Path(urlsplit(video_value).path).suffix.lstrip(".").lower()

    def _probe_duration(self, video_value: str) -> float | None:
        try:
            location = File(video_value).resolve()
            metadata = extract_video_metadata_structured(location)
        except (FileLoadError, ValueError, RuntimeError) as e:
            logger.warning("%s: could not probe input video duration, skipping the check: %s", self.name, e)
            return None
        return metadata.file_details.optional_duration

    def _find_input_problems(self, video_value: str) -> list[str]:
        problems: list[str] = []

        token = self._container_token(video_value)
        if token and token != SUPPORTED_CONTAINER:
            problems.append(f"It's a .{token} file. FLUX Video Upscale requires .mp4.")

        if not video_value.startswith("data:"):
            duration = self._probe_duration(video_value)
            if duration is not None and duration > MAX_INPUT_DURATION_SECONDS:
                problems.append(
                    f"It's {duration:.2f} seconds long. FLUX Video Upscale's limit is {MAX_INPUT_DURATION_SECONDS:g} seconds."
                )

        return problems

    async def _check_input_limits(self, video_value: str) -> None:
        problems = await asyncio.to_thread(self._find_input_problems, video_value)
        if problems:
            msg = f"Cannot upscale this video connected to 'Input Video'. {' '.join(problems)} Use a Trim Video node to fix this."
            raise NodeError(msg)

    async def _prepare_video_data_uri(self, video_input: Any) -> str:
        video_value = coerce_media_url_or_data_uri(video_input, kind="video")
        if not video_value:
            msg = "Connect a video to 'Input Video'."
            raise NodeError(msg)

        await self._check_input_limits(video_value)

        if video_value.startswith("data:"):
            return video_value

        try:
            return await File(video_value).aread_data_uri(fallback_mime="video/mp4")
        except FileLoadError as e:
            msg = f"Could not load video from '{video_value}': {e}"
            raise NodeError(msg) from e

    @staticmethod
    def _parse_creativity(value: str | None) -> int:
        if value == Flux3UpscaleCreativity.PRECISE:
            return 0
        if value == Flux3UpscaleCreativity.CREATIVE:
            return 1
        msg = f"'Creativity' must be one of {CREATIVITY_OPTIONS}. It's set to {value!r}."
        raise NodeError(msg)

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

    def validate_before_node_run(self) -> list[Exception] | None:
        exceptions = super().validate_before_node_run() or []

        video_value = self.get_parameter_value("input_video")
        if not self._has_media_value(video_value):
            exceptions.append(NodeError("Connect a video to 'Input Video'."))

        upscale_factor = self.get_parameter_value("upscale_factor")
        if upscale_factor is None or not (MIN_UPSCALE_FACTOR <= float(upscale_factor) <= MAX_UPSCALE_FACTOR):
            exceptions.append(
                NodeError(
                    f"'Upscale Factor' must be between {MIN_UPSCALE_FACTOR} and {MAX_UPSCALE_FACTOR}. It's set to {upscale_factor!r}."
                )
            )

        creativity = self.get_parameter_value("creativity")
        if creativity not in CREATIVITY_OPTIONS:
            exceptions.append(
                NodeError(f"'Creativity' must be one of {CREATIVITY_OPTIONS}. It's set to {creativity!r}.")
            )

        return exceptions or None

    async def _build_payload(self) -> dict[str, Any]:
        input_video_uri = await self._prepare_video_data_uri(self.get_parameter_value("input_video"))
        upscale_factor = float(self.get_parameter_value("upscale_factor") or DEFAULT_UPSCALE_FACTOR)
        creativity = self._parse_creativity(self.get_parameter_value("creativity"))
        prompt = (self.get_parameter_value("prompt") or "").strip()
        safety_tolerance = self._parse_safety_tolerance(self.get_parameter_value("safety_tolerance"))

        return {
            "input_video": input_video_uri,
            "upscale_factor": upscale_factor,
            "creativity": creativity,
            "prompt": prompt,
            "safety_tolerance": safety_tolerance,
        }

    async def _parse_result(self, _result_json: dict[str, Any], generation_id: str) -> None:
        await self._save_generated_media(
            generation_id,
            "video_url",
            lambda v, n: VideoUrlArtifact(value=v, name=n),
            kind=ArtifactKind.VIDEO,
            action="upscaled",
        )

    def _set_safe_defaults(self) -> None:
        self.parameter_output_values["generation_id"] = ""
        self.parameter_output_values["provider_response"] = None
        self.parameter_output_values["video_url"] = None
