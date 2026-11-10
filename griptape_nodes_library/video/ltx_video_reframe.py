from __future__ import annotations

import asyncio
import json
import logging
import subprocess
from enum import StrEnum
from typing import Any

from griptape.artifacts.video_url_artifact import VideoUrlArtifact
from griptape_nodes.exe_types.core_types import NodeError, Parameter, ParameterMode
from griptape_nodes.exe_types.param_components.model_access_component import ModelAccessComponent
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_types.parameter_dict import ParameterDict
from griptape_nodes.exe_types.param_types.parameter_float import ParameterFloat
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.exe_types.param_types.parameter_video import ParameterVideo
from griptape_nodes.files.file import FileLoadError
from griptape_nodes.traits.options import Options

# static_ffmpeg is dynamically installed by the library loader at runtime
from static_ffmpeg import run  # type: ignore[import-untyped]

from griptape_nodes_library.media import coerce_media_url_or_data_uri, prepare_media_data_uri, resolve_media_location
from griptape_nodes_library.proxy import ArtifactKind, GriptapeProxyNode
from griptape_nodes_library.utils.ffmpeg_utils import describe_ffmpeg_failure, is_quarter_turned
from griptape_nodes_library.video.public_video_url_mixin import PublicVideoUrlMixin

logger = logging.getLogger("griptape_nodes")

__all__ = ["LTXVideoReframe", "PlacementMode"]

# The dropdown stores the provider's own model id. The reframe endpoint is served under the
# bare `ltx` model name, so the proxy id is `ltx:video-to-video-reframe`.
DEFAULT_MODEL = "ltx"
API_OPERATION = "video-to-video-reframe"

# Input limits per the LTX docs:
# https://docs.ltx.io/api-documentation/api-reference/async-video-generation/submit-video-to-video-reframe
MAX_VIDEO_DURATION_SECONDS = 60
# Container durations carry rounding (60 s at 30 fps can read 60.02), and the frame count is the
# authoritative limit, so the duration check allows a small margin.
DURATION_TOLERANCE_SECONDS = 0.1
PROBE_TIMEOUT_SECONDS = 30
MAX_VIDEO_FRAMES = 1800
MIN_VIDEO_SIDE_PIXELS = 64
MAX_VIDEO_SIDE_PIXELS = 4096

# Resolution -> aspect-ratio label shown under each dropdown row.
RESOLUTION_ASPECT_RATIOS: dict[str, str] = {
    "720x720": "1:1 (square)",
    "1080x1080": "1:1 (square)",
    "720x900": "4:5 (portrait)",
    "1080x1350": "4:5 (portrait)",
    "900x720": "5:4 (landscape)",
    "1350x1080": "5:4 (landscape)",
    "720x1280": "9:16 (portrait)",
    "1080x1920": "9:16 (portrait)",
    "1280x720": "16:9 (landscape)",
    "1920x1080": "16:9 (landscape)",
}
SUPPORTED_RESOLUTIONS = list(RESOLUTION_ASPECT_RATIOS)
DEFAULT_RESOLUTION = "1080x1920"


class PlacementMode(StrEnum):
    CENTER = "center"
    ANCHOR = "anchor"
    RECTANGLE = "rectangle"


class Anchor(StrEnum):
    CENTER = "center"
    TOP = "top"
    BOTTOM = "bottom"
    LEFT = "left"
    RIGHT = "right"
    TOP_LEFT = "top_left"
    TOP_RIGHT = "top_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_RIGHT = "bottom_right"


ANCHOR_PARAMETER_NAMES = ["anchor"]
RECTANGLE_PARAMETER_NAMES = ["x", "y", "width", "height"]

_PLACEMENT_TOOLTIP_NOTE = (
    "Placements other than 'center' can't enlarge the video more than 2x its original size, "
    "so a small input video may be rejected."
)


def _optional_float(value: Any, *, label: str) -> float | None:
    """Return `value` as a float, or None when it is blank."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as e:
        msg = f"'{label}' must be a number between 0 and 1. It's set to {value!r}."
        raise NodeError(msg) from e


def build_placement(  # noqa: PLR0913
    mode: str,
    *,
    anchor: str | None = None,
    x: Any = None,
    y: Any = None,
    width: Any = None,
    height: Any = None,
) -> dict[str, Any] | None:
    """Validate the placement settings and return the API `placement` object.

    Returns None for `center`, because omitting `placement` centres the video at the largest
    size that fits. Raises NodeError naming the offending parameter otherwise.
    """
    match mode:
        case PlacementMode.CENTER:
            return None
        case PlacementMode.ANCHOR:
            valid_anchors = [member.value for member in Anchor]
            if anchor not in valid_anchors:
                msg = f"'Anchor' must be one of {', '.join(valid_anchors)}. It's set to {anchor!r}."
                raise NodeError(msg)
            return {"anchor": str(anchor)}
        case PlacementMode.RECTANGLE:
            return _build_rectangle(x=x, y=y, width=width, height=height)
        case _:
            valid_modes = ", ".join(member.value for member in PlacementMode)
            msg = f"'Placement Mode' must be one of {valid_modes}. It's set to {mode!r}."
            raise NodeError(msg)


def _build_rectangle(*, x: Any, y: Any, width: Any, height: Any) -> dict[str, Any]:
    x_value = _optional_float(x, label="X")
    y_value = _optional_float(y, label="Y")
    width_value = _optional_float(width, label="Width")
    height_value = _optional_float(height, label="Height")

    # A blank X or Y is the frame's top-left edge, matching the parameters' defaults.
    x_value = 0.0 if x_value is None else x_value
    y_value = 0.0 if y_value is None else y_value

    if not 0 <= x_value < 1:
        msg = f"'X' must be at least 0 and less than 1. It's set to {x_value}."
        raise NodeError(msg)
    if not 0 <= y_value < 1:
        msg = f"'Y' must be at least 0 and less than 1. It's set to {y_value}."
        raise NodeError(msg)
    if width_value is None and height_value is None:
        msg = "Set 'Width', 'Height' or both. The other follows the video's aspect ratio when left blank."
        raise NodeError(msg)
    if width_value is not None and not 0 < width_value <= 1:
        msg = f"'Width' must be greater than 0 and at most 1. It's set to {width_value}."
        raise NodeError(msg)
    if height_value is not None and not 0 < height_value <= 1:
        msg = f"'Height' must be greater than 0 and at most 1. It's set to {height_value}."
        raise NodeError(msg)

    placement: dict[str, Any] = {"x": x_value, "y": y_value}
    if width_value is not None:
        placement["width"] = width_value
    if height_value is not None:
        placement["height"] = height_value
    return placement


class LTXVideoReframe(PublicVideoUrlMixin, GriptapeProxyNode):
    """Reframe a video to a new resolution using LTX AI via Griptape Cloud model proxy.

    The model fills the areas around the source video with generated content that matches it.
    Billed per second of input video, by output resolution (720p or 1080p tier).

    Inputs:
        - video (VideoUrlArtifact): Source video (required, max 60s and 1800 frames, 64-4096 px per side)
        - resolution (str): Target output resolution
        - placement_mode (str): center (default), anchor or rectangle
        - anchor (str): Where the video lands in the frame (anchor mode)
        - x, y, width, height (float): Position and size as fractions of the frame (rectangle mode)
        - model (str): Model to use (ltx)

    Outputs:
        - generation_id (str): Griptape Cloud generation id
        - provider_response (dict): Response from API (latest polling response)
        - video_url (VideoUrlArtifact): Saved reframed video
        - was_successful (bool): Whether the generation succeeded
        - result_details (str): Details about the generation result or error
    """

    SERVICE_NAME = "Griptape"
    API_KEY_NAME = "GT_CLOUD_API_KEY"
    DEFAULT_MAX_ATTEMPTS = 240

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # INPUTS / PROPERTIES

        model_param = ParameterString(
            name="model",
            default_value=DEFAULT_MODEL,
            tooltip="Model to use for video reframing",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
        )
        self.add_parameter(model_param)
        # License-policy dropdown: the component adds Options + refresh Button traits and
        # marks the models the license denies; the proxy base refuses a denied selection.
        self._model_access = ModelAccessComponent(
            node=self,
            parameter=model_param,
            model_choices=[DEFAULT_MODEL],
            default_model=DEFAULT_MODEL,
        )

        self.add_parameter(
            ParameterVideo(
                name="video",
                tooltip=(
                    f"Source video to reframe (max {MAX_VIDEO_DURATION_SECONDS} seconds and "
                    f"{MAX_VIDEO_FRAMES} frames, {MIN_VIDEO_SIDE_PIXELS}-{MAX_VIDEO_SIDE_PIXELS} px per side)"
                ),
                allowed_modes={ParameterMode.INPUT},
                hide_property=True,
                ui_options={"display_name": "input video"},
            )
        )

        resolution_param = ParameterString(
            name="resolution",
            default_value=DEFAULT_RESOLUTION,
            tooltip=(
                "Target output resolution. The source video is extended to fill this frame. "
                "Billed per second of input video by output resolution tier (720p or 1080p)."
            ),
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            traits={Options(choices=SUPPORTED_RESOLUTIONS)},
        )
        self.add_parameter(resolution_param)
        resolution_param.update_ui_options(
            {
                "data": [{"name": name, "subtitle": ratio} for name, ratio in RESOLUTION_ASPECT_RATIOS.items()],
                "dropdown_row_subtitles": True,
            }
        )

        self._placement_mode_param = ParameterString(
            name="placement_mode",
            default_value=PlacementMode.CENTER.value,
            tooltip=(
                "Where the source video lands on the output frame. 'center' fits it at the largest "
                "size that fits, 'anchor' aligns it to an edge or corner, and 'rectangle' sets its "
                f"position and size exactly. {_PLACEMENT_TOOLTIP_NOTE}"
            ),
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            traits={Options(choices=[mode.value for mode in PlacementMode])},
        )
        self.add_parameter(self._placement_mode_param)

        self.add_parameter(
            ParameterString(
                name="anchor",
                default_value=Anchor.CENTER.value,
                tooltip=f"Edge or corner the video is aligned to. {_PLACEMENT_TOOLTIP_NOTE}",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=[anchor.value for anchor in Anchor])},
            )
        )
        self.add_parameter(
            ParameterFloat(
                name="x",
                default_value=0.0,
                tooltip="Left edge of the placed video as a fraction of the output width (0 to less than 1).",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                step=0.01,
            )
        )
        self.add_parameter(
            ParameterFloat(
                name="y",
                default_value=0.0,
                tooltip="Top edge of the placed video as a fraction of the output height (0 to less than 1).",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                step=0.01,
            )
        )
        self.add_parameter(
            Parameter(
                name="width",
                type="float",
                input_types=["float", "int", "str"],
                # Defaults to None like 'Height': the engine does not save None values, so a
                # default of 1.0 would bring a cleared width back after a save and reload.
                default_value=None,
                tooltip=(
                    "Width of the placed video as a fraction of the output width (greater than 0, at most 1). "
                    "Leave blank to follow the height and the video's aspect ratio. Set at least one of "
                    f"'Width' and 'Height'. {_PLACEMENT_TOOLTIP_NOTE}"
                ),
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"step": 0.01},
            )
        )
        self.add_parameter(
            Parameter(
                name="height",
                type="float",
                input_types=["float", "int", "str"],
                default_value=None,
                tooltip=(
                    "Height of the placed video as a fraction of the output height (greater than 0, at most 1). "
                    "Leave blank to follow the width and the video's aspect ratio. Set at least one of "
                    f"'Width' and 'Height'. {_PLACEMENT_TOOLTIP_NOTE}"
                ),
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"step": 0.01},
            )
        )
        # Centre placement is the default, so neither set of controls is relevant until chosen.
        self.hide_parameter_by_name(ANCHOR_PARAMETER_NAMES + RECTANGLE_PARAMETER_NAMES)

        # OUTPUTS
        self.add_parameter(
            ParameterDict(
                name="provider_response",
                tooltip="Response from API (latest polling response)",
                allowed_modes={ParameterMode.OUTPUT},
                hide_property=True,
                hide=True,
            )
        )

        self.add_parameter(
            ParameterVideo(
                name="video_url",
                tooltip="Saved reframed video",
                allowed_modes={ParameterMode.OUTPUT, ParameterMode.PROPERTY},
                settable=False,
                ui_options={"pulse_on_run": True},
            )
        )

        self._output_file = ProjectFileParameter(
            node=self,
            name="output_file",
            default_filename="ltx_video_reframe.mp4",
        )
        self._output_file.add_parameter()

        self._create_status_parameters(
            result_details_tooltip="Details about the video reframe result or any errors",
            result_details_placeholder="Reframe status and details will appear here.",
            parameter_group_initially_collapsed=True,
        )

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        if parameter is self._placement_mode_param:
            self._update_placement_visibility(value)
        super().after_value_set(parameter, value)

    def _update_placement_visibility(self, mode: Any) -> None:
        match mode:
            case PlacementMode.ANCHOR:
                self.show_parameter_by_name(ANCHOR_PARAMETER_NAMES)
                self.hide_parameter_by_name(RECTANGLE_PARAMETER_NAMES)
            case PlacementMode.RECTANGLE:
                self.hide_parameter_by_name(ANCHOR_PARAMETER_NAMES)
                self.show_parameter_by_name(RECTANGLE_PARAMETER_NAMES)
            case _:
                self.hide_parameter_by_name(ANCHOR_PARAMETER_NAMES + RECTANGLE_PARAMETER_NAMES)

    def _get_api_model_id(self) -> str:
        return f"{self._get_selected_model_id()}:{API_OPERATION}"

    @staticmethod
    def _extract_input_video_url(video_input: Any) -> str | None:
        return coerce_media_url_or_data_uri(video_input, kind="video")

    def _probe_video(self, video_url: str) -> tuple[int, int, int, float] | None:
        """Probe a video with ffprobe. Returns (width, height, frames, duration seconds) or None."""
        try:
            _, ffprobe_path = run.get_or_fetch_platform_executables_else_raise()

            # Project-saved inputs carry macro paths like {outputs}/clip.mp4 that ffprobe can't open
            location = resolve_media_location(video_url)

            cmd = [
                ffprobe_path,
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_streams",
                "-show_format",
                "-select_streams",
                "v:0",
                location,
            ]
            result = subprocess.run(  # noqa: S603
                cmd, capture_output=True, text=True, check=True, timeout=PROBE_TIMEOUT_SECONDS
            )
            return self._parse_probe_output(json.loads(result.stdout))
        except (
            subprocess.TimeoutExpired,
            subprocess.CalledProcessError,
            json.JSONDecodeError,
            FileLoadError,
            ValueError,
            KeyError,
        ) as e:
            logger.debug("%s ffprobe failed to probe video: %s", self.name, describe_ffmpeg_failure(e))
            return None

    @staticmethod
    def _parse_probe_output(data: dict[str, Any]) -> tuple[int, int, int, float] | None:
        """Turn ffprobe JSON into (width, height, frames, duration seconds), or None if unusable.

        Width and height are swapped for quarter-turned videos, so they describe the video as
        players show it. Frames come from `nb_frames`, falling back to duration x frame rate.
        """
        streams = data.get("streams", [])
        if not streams:
            return None

        stream = streams[0]
        width = int(stream.get("width") or 0)
        height = int(stream.get("height") or 0)
        if width <= 0 or height <= 0:
            return None
        if is_quarter_turned(stream):
            width, height = height, width

        duration = _first_float(stream.get("duration"), (data.get("format") or {}).get("duration")) or 0.0
        frames_str = stream.get("nb_frames")
        if frames_str and str(frames_str).isdigit():
            frames = int(frames_str)
        else:
            frames = round(duration * _frame_rate(stream.get("avg_frame_rate")))

        return width, height, frames, duration

    async def _validate_video_input(self, video: Any) -> None:
        """Check a video is connected and, when it can be probed, that it fits LTX's input limits."""
        if not video:
            msg = "Connect a video to 'input video' to reframe."
            raise NodeError(msg)

        video_url = self._extract_input_video_url(video)
        if not video_url:
            return

        # ffprobe is a blocking subprocess, so keep it off the event loop.
        probed = await asyncio.to_thread(self._probe_video, video_url)
        if probed is None:
            # If we can't probe, let the proxy return the error.
            return

        width, height, frames, duration = probed
        if duration > MAX_VIDEO_DURATION_SECONDS + DURATION_TOLERANCE_SECONDS:
            msg = (
                f"The input video is {duration:.1f}s long, over the {MAX_VIDEO_DURATION_SECONDS}s limit. "
                "Trim it and try again."
            )
            raise NodeError(msg)
        if frames > MAX_VIDEO_FRAMES:
            msg = (
                f"The input video has {frames} frames, over the {MAX_VIDEO_FRAMES} frame limit. "
                "Trim it or lower its frame rate and try again."
            )
            raise NodeError(msg)
        if not (
            MIN_VIDEO_SIDE_PIXELS <= width <= MAX_VIDEO_SIDE_PIXELS
            and MIN_VIDEO_SIDE_PIXELS <= height <= MAX_VIDEO_SIDE_PIXELS
        ):
            msg = (
                f"The input video is {width}x{height}. Each side must be between "
                f"{MIN_VIDEO_SIDE_PIXELS} and {MAX_VIDEO_SIDE_PIXELS} pixels. Resize it and try again."
            )
            raise NodeError(msg)

    async def _prepare_video_data_uri_async(self, video_input: Any) -> str | None:
        """Convert video input to a base64 data URI."""
        return await prepare_media_data_uri(video_input, kind="video", node_name=self.name)

    async def _process_generation(self) -> None:
        self._reset_video_uploads()
        try:
            await super()._process_generation()
        finally:
            await self._cleanup_video_uploads()

    async def _build_payload(self) -> dict[str, Any]:
        video = self.get_parameter_value("video")
        await self._validate_video_input(video)

        resolution = self.get_parameter_value("resolution") or DEFAULT_RESOLUTION
        if resolution not in SUPPORTED_RESOLUTIONS:
            msg = f"'Resolution' must be one of {', '.join(SUPPORTED_RESOLUTIONS)}. It's set to {resolution!r}."
            raise NodeError(msg)

        placement = build_placement(
            self.get_parameter_value("placement_mode") or PlacementMode.CENTER.value,
            anchor=self.get_parameter_value("anchor"),
            x=self.get_parameter_value("x"),
            y=self.get_parameter_value("y"),
            width=self.get_parameter_value("width"),
            height=self.get_parameter_value("height"),
        )

        # Tier 1: upload to Griptape Cloud and send LTX a public URL it fetches server-side
        # (avoids the 413 from base64-inflating the body). Tier 2: base64 data URI + size guard.
        video_uri = await self._resolve_video_uri(video)

        payload: dict[str, Any] = {"video_uri": video_uri, "resolution": resolution}
        if placement is not None:
            payload["placement"] = placement
        return payload

    async def _parse_result(self, _result_json: dict[str, Any], generation_id: str) -> None:
        """Save the hosted video. LTX returns the media as the response body itself."""
        await self._save_generated_media(
            generation_id,
            "video_url",
            lambda v, n: VideoUrlArtifact(value=v, name=n),
            kind=ArtifactKind.VIDEO,
            action="reframed",
        )

    def _set_safe_defaults(self) -> None:
        self.parameter_output_values["generation_id"] = ""
        self.parameter_output_values["provider_response"] = None
        self.parameter_output_values["video_url"] = None


def _first_float(*values: Any) -> float | None:
    for value in values:
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _frame_rate(rate: Any) -> float:
    """Parse an ffprobe rational such as "30000/1001". Returns 0.0 when unusable."""
    try:
        numerator, _, denominator = str(rate).partition("/")
        return float(numerator) / float(denominator or 1)
    except (ValueError, ZeroDivisionError):
        return 0.0
