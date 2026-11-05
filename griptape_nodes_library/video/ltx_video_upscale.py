from __future__ import annotations

import asyncio
import json
import logging
import subprocess
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from griptape.artifacts.video_url_artifact import VideoUrlArtifact
from griptape_nodes.exe_types.core_types import NodeError, ParameterMode
from griptape_nodes.exe_types.param_components.model_access_component import ModelAccessComponent
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_types.parameter_dict import ParameterDict
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.exe_types.param_types.parameter_video import ParameterVideo
from griptape_nodes.files.file import FileLoadError
from griptape_nodes.traits.options import Options

# static_ffmpeg is dynamically installed by the library loader at runtime
from static_ffmpeg import run  # type: ignore[import-untyped]

from griptape_nodes_library.media import (
    coerce_media_url_or_data_uri,
    is_public_https_domain_url,
    prepare_media_data_uri,
    resolve_media_location,
)
from griptape_nodes_library.proxy import ArtifactKind, GriptapeProxyNode
from griptape_nodes_library.utils.ffmpeg_utils import describe_ffmpeg_failure
from griptape_nodes_library.video.public_video_url_mixin import PublicVideoUrlMixin

logger = logging.getLogger("griptape_nodes")

__all__ = ["LTXVideoUpscale"]

MODEL_ID = "ltx"
API_MODEL_ID = "ltx:video-upscale"


class UpscaleResolution(StrEnum):
    """Output resolutions the video upscale endpoint accepts."""

    HD_1080P = "1080p"
    UHD_4K = "4k"
    UHD_8K = "8k"


DEFAULT_RESOLUTION = UpscaleResolution.UHD_4K

# Input limits per LTX video upscale docs.
# https://lightricks-preview-feature-alpha-gen-docs.docs.buildwithfern.com/api-documentation/api-reference/async-video-generation/submit-video-upscale
MIN_SHORT_SIDE = 480
MAX_LONG_SIDE = 7680
MAX_SHORT_SIDE = 4320
MIN_FRAMES = 2
MIN_FPS = 1.0
ACCEPTED_CODECS = ("h264", "hevc")


def _max_frames_for(resolution: str) -> int:
    """Maximum input frame count for an output resolution."""
    match resolution:
        case UpscaleResolution.HD_1080P:
            return 2401
        case UpscaleResolution.UHD_4K:
            return 601
        case UpscaleResolution.UHD_8K:
            return 241
        case _:
            msg = f"Unsupported resolution '{resolution}'. Valid resolutions: {', '.join(UpscaleResolution)}."
            raise ValueError(msg)


@dataclass(frozen=True)
class VideoProbe:
    """The stream properties the upscale checks need. Frames and fps are None when unreadable."""

    width: int
    height: int
    frames: int | None
    codec: str | None
    fps: float | None


def _frame_count(stream: dict[str, Any], fps: float | None) -> int | None:
    """Frame count from ``nb_frames``, or duration x fps when the container doesn't record it."""
    nb_frames = stream.get("nb_frames")
    if nb_frames and str(nb_frames).isdigit():
        return int(nb_frames)
    duration_text = stream.get("duration")
    if duration_text and fps is not None:
        try:
            duration = float(duration_text)
        except (TypeError, ValueError):
            return None
        if duration > 0:
            return round(duration * fps)
    return None


def _parse_frame_rate(value: str | None) -> float | None:
    """Parse an ffprobe rate such as ``25/1`` or ``30000/1001``. Returns None when unknown."""
    if not value:
        return None
    numerator_text, separator, denominator_text = value.partition("/")
    try:
        numerator = float(numerator_text)
        denominator = float(denominator_text) if separator else 1.0
    except ValueError:
        return None
    if numerator <= 0 or denominator <= 0:
        return None
    return numerator / denominator


class LTXVideoUpscale(PublicVideoUrlMixin, GriptapeProxyNode):
    """Upscale a video to 1080p, 4K or 8K using LTX AI via Griptape Cloud model proxy.

    Inputs:
        - video (VideoUrlArtifact): Input video to upscale (required, H.264 or HEVC, short side at least 480 px,
          at most 7680x4320, sent as a public URL or base64)
        - resolution (str): Output resolution (1080p, 4k or 8k, default 4k)
        - model (str): Model to use (LTX)
        (Always polls for result: 5s interval, 20 min timeout)

    Outputs:
        - generation_id (str): Griptape Cloud generation id
        - provider_response (dict): Response from API (latest polling response)
        - video_url (VideoUrlArtifact): Saved upscaled video
        - was_successful (bool): Whether the generation succeeded
        - result_details (str): Details about the generation result or error
    """

    SERVICE_NAME = "Griptape"
    API_KEY_NAME = "GT_CLOUD_API_KEY"
    # Griptape Cloud downloads a URL input (up to 200 MB) to check and bill it while handling the
    # create call, allowing up to 150 s for that download, so wait longer than that for the response.
    SUBMIT_TIMEOUT_SECONDS = 200
    DEFAULT_MAX_ATTEMPTS = 240

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # INPUTS / PROPERTIES

        # Model parameter (the upscale endpoint is addressed by a fixed API model id)
        model_param = ParameterString(
            name="model",
            default_value=MODEL_ID,
            tooltip="Model to use for video upscale",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
        )
        self.add_parameter(model_param)
        # License-policy dropdown: the component adds Options + refresh Button traits and
        # marks the models the license denies; the proxy base refuses a denied selection.
        self._model_access = ModelAccessComponent(
            node=self,
            parameter=model_param,
            model_choices=[MODEL_ID],
            default_model=MODEL_ID,
        )

        # Video input parameter
        self.add_parameter(
            ParameterVideo(
                name="video",
                tooltip="Input video to upscale (H.264 or HEVC, short side at least 480 px, at most 7680x4320)",
                allowed_modes={ParameterMode.INPUT},
                hide_property=True,
                ui_options={"display_name": "input video"},
            )
        )

        self.add_parameter(
            ParameterString(
                name="resolution",
                default_value=DEFAULT_RESOLUTION.value,
                tooltip=(
                    "Output resolution. Higher resolutions allow fewer input frames. "
                    "Billed per second of input video, by output resolution."
                ),
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=[resolution.value for resolution in UpscaleResolution])},
            )
        )

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
                tooltip="Saved upscaled video",
                allowed_modes={ParameterMode.OUTPUT, ParameterMode.PROPERTY},
                settable=False,
                ui_options={"pulse_on_run": True},
            )
        )

        self._output_file = ProjectFileParameter(
            node=self,
            name="output_file",
            default_filename="ltx_video_upscale.mp4",
        )
        self._output_file.add_parameter()

        # Create status parameters for success/failure tracking
        self._create_status_parameters(
            result_details_tooltip="Details about the video upscale result or any errors",
            result_details_placeholder="Upscale status and details will appear here.",
            parameter_group_initially_collapsed=True,
        )

    @staticmethod
    def _extract_input_video_url(video_input: Any) -> str | None:
        return coerce_media_url_or_data_uri(video_input, kind="video")

    def _probe_video(self, video_url: str) -> VideoProbe | None:
        """Read width, height, frame count, codec and fps in one ffprobe call.

        Returns None when the probe fails, so validation is skipped and the proxy reports the problem.
        ``-count_frames`` is left out because it decodes the whole stream, which is slow on large inputs.
        Frame count comes from ``nb_frames`` when the container records it, and otherwise from
        duration x fps.
        """
        try:
            _, ffprobe_path = run.get_or_fetch_platform_executables_else_raise()

            # Project-saved inputs carry macro paths like {outputs}/clip.mp4 that ffprobe can't open
            location = resolve_media_location(video_url)

            cmd = [
                ffprobe_path,
                # "error" rather than "quiet" so the log below can report why the probe failed.
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_streams",
                "-select_streams",
                "v:0",
                location,
            ]

            result = subprocess.run(  # noqa: S603
                cmd, capture_output=True, text=True, check=True, timeout=30
            )

            streams = json.loads(result.stdout).get("streams", [])
            if not streams:
                return None

            stream = streams[0]
            width = int(stream.get("width") or 0)
            height = int(stream.get("height") or 0)
            if width <= 0 or height <= 0:
                return None

            fps = _parse_frame_rate(stream.get("avg_frame_rate")) or _parse_frame_rate(stream.get("r_frame_rate"))
            frames = _frame_count(stream, fps)
            codec = stream.get("codec_name")

            return VideoProbe(
                width=width,
                height=height,
                frames=frames,
                codec=str(codec) if codec else None,
                fps=fps,
            )

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

    def _validate_video_input(self, video: Any, resolution: str) -> str | None:
        """Check the input video against the upscale endpoint's limits before submitting.

        Each check is skipped when its property can't be read, so an unreadable probe lets
        the proxy report the problem instead of blocking the run.
        """
        if not video:
            return "Connect a video to 'input video' to upscale."

        video_url = self._extract_input_video_url(video)
        if not video_url:
            return None

        probe = self._probe_video(video_url)
        if probe is None:
            return None

        if probe.codec is not None and probe.codec not in ACCEPTED_CODECS:
            return (
                f"'input video' uses {probe.codec}, which LTX video upscale doesn't accept. "
                "Convert it to H.264 or HEVC and try again."
            )

        short_side = min(probe.width, probe.height)
        long_side = max(probe.width, probe.height)
        if short_side < MIN_SHORT_SIDE:
            return (
                f"'input video' is {probe.width}x{probe.height}, with a short side under the "
                f"{MIN_SHORT_SIDE} px minimum. Use a larger video and try again."
            )
        if long_side > MAX_LONG_SIDE or short_side > MAX_SHORT_SIDE:
            return (
                f"'input video' is {probe.width}x{probe.height}, larger than the "
                f"{MAX_LONG_SIDE}x{MAX_SHORT_SIDE} limit. Resize it and try again."
            )
        if probe.frames is not None and probe.frames < MIN_FRAMES:
            return (
                f"'input video' has {probe.frames} frame, but upscale needs at least {MIN_FRAMES}. "
                "Use a longer video and try again."
            )
        max_frames = _max_frames_for(resolution)
        if probe.frames is not None and probe.frames > max_frames:
            return (
                f"'input video' has {probe.frames} frames, over the {max_frames} frame limit for "
                f"'resolution' {resolution}. Trim the video or choose a lower 'resolution' and try again."
            )
        if probe.fps is not None and probe.fps < MIN_FPS:
            return (
                f"'input video' runs at {probe.fps:.2f} fps, below the {MIN_FPS:g} fps minimum. "
                "Use a video with a higher frame rate and try again."
            )

        return None

    def _validate_resolution(self, resolution: str) -> str | None:
        if resolution not in {member.value for member in UpscaleResolution}:
            return (
                f"'resolution' '{resolution}' is not supported. "
                f"Choose one of: {', '.join(member.value for member in UpscaleResolution)}."
            )
        return None

    async def _process_generation(self) -> None:
        self._reset_video_uploads()
        try:
            await super()._process_generation()
        finally:
            await self._cleanup_video_uploads()

    def _get_api_model_id(self) -> str:
        return API_MODEL_ID

    async def _prepare_video_data_uri_async(self, video_input: Any) -> str | None:
        """Convert video input to a base64 data URI."""
        return await prepare_media_data_uri(video_input, kind="video", node_name=self.name)

    async def _build_payload(self) -> dict[str, Any]:
        """Build the request payload for LTX video upscale."""
        video = self.get_parameter_value("video")
        resolution = self.get_parameter_value("resolution") or DEFAULT_RESOLUTION.value

        if error := self._validate_resolution(resolution):
            raise NodeError(error)
        # ffprobe runs in a worker thread so a slow remote input doesn't stall the event loop.
        if error := await asyncio.to_thread(self._validate_video_input, video, resolution):
            raise NodeError(error)

        # Tier 1: upload to Griptape Cloud and send LTX a public URL it fetches server-side.
        # Tier 2: base64 data URI with a size guard.
        video_uri = await self._resolve_video_uri(video)
        self._warn_if_proxy_downloads_external_url(video, video_uri)

        # The model is fixed by the endpoint, so no `model` field is sent.
        payload: dict[str, Any] = {
            "video_uri": video_uri,
            "resolution": resolution,
        }

        return payload

    def _warn_if_proxy_downloads_external_url(self, video: Any, video_uri: str) -> None:
        """Tell the user submission may be slow when the proxy must fetch a third-party URL first.

        Griptape Cloud downloads a URL input to check and bill it before submitting. Local files are
        uploaded to Griptape storage, which serves quickly, but an already-public URL is passed through
        and fetched from its own host, which can take minutes for a large file on a slow server.
        """
        # The mixin passes an already-public https URL through unchanged and uploads anything else,
        # so the proxy fetches from a third-party host exactly when the URI sent is the user's own URL.
        connected = coerce_media_url_or_data_uri(video, kind="video")
        if not connected or video_uri != connected or not is_public_https_domain_url(video_uri):
            return
        message = (
            "The input video is a link to another website. Griptape Cloud downloads it to check its "
            "length and size before starting, which can take up to about 2.5 minutes for a large file "
            "on a slow server. Connect a local file instead for a faster start."
        )
        logger.warning("%s: %s", self.name, message)
        # An in-progress message: leaves the success flag that routes the node untouched.
        self.status_component.clear_execution_status(initial_message=message)

    async def _parse_result(self, _result_json: dict[str, Any], generation_id: str) -> None:
        """Save the hosted video. LTX returns the media as the response body itself."""
        await self._save_generated_media(
            generation_id,
            "video_url",
            lambda v, n: VideoUrlArtifact(value=v, name=n),
            kind=ArtifactKind.VIDEO,
            action="upscaled",
        )

    def _extract_error_message(self, response_json: dict[str, Any]) -> str:
        if not response_json:
            return ""

        status_detail = response_json.get("status_detail")
        if isinstance(status_detail, dict):
            error = status_detail.get("error", "")
            details = status_detail.get("details", "")

            if details and isinstance(details, str):
                try:
                    details_obj = json.loads(details)
                    if isinstance(details_obj, dict):
                        error_obj = details_obj.get("error")
                        if isinstance(error_obj, dict):
                            clean_message = error_obj.get("message")
                            if clean_message:
                                details = clean_message
                except (ValueError, json.JSONDecodeError):
                    pass

            if error and details:
                return f"{error}: {details}"
            return str(error or details or "")

        error = response_json.get("error")
        if isinstance(error, dict):
            return str(error.get("message") or error.get("type") or "")
        if isinstance(error, str):
            return error

        return ""

    def _handle_payload_build_error(self, e: Exception) -> None:
        # NodeError and ValueError carry messages the user should read unchanged, so both skip
        # the base class's "Failed to build the request:" prefix.
        if isinstance(e, NodeError | ValueError):
            self._set_safe_defaults()
            self._set_status_results(was_successful=False, result_details=str(e))
            self._handle_failure_exception(e)
            return

        super()._handle_payload_build_error(e)

    def _set_safe_defaults(self) -> None:
        self.parameter_output_values["generation_id"] = ""
        self.parameter_output_values["provider_response"] = None
        self.parameter_output_values["video_url"] = None
