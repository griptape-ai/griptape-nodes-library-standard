from __future__ import annotations

import asyncio
import json
import logging
import subprocess
from typing import Any, NamedTuple

from griptape.artifacts import VideoUrlArtifact
from griptape_nodes.exe_types.core_types import NodeError, Parameter, ParameterMode
from griptape_nodes.exe_types.param_components.model_access_component import ModelAccessComponent
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_types.parameter_bool import ParameterBool
from griptape_nodes.exe_types.param_types.parameter_dict import ParameterDict
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.exe_types.param_types.parameter_video import ParameterVideo
from griptape_nodes.files.file import FileLoadError

# static_ffmpeg is dynamically installed by the library loader at runtime
from static_ffmpeg import run  # type: ignore[import-untyped]

from griptape_nodes_library.media import (
    coerce_media_url_or_data_uri,
    is_public_https_domain_url,
    prepare_media_data_uri,
    resolve_media_location,
)
from griptape_nodes_library.proxy import ArtifactKind, GriptapeProxyNode
from griptape_nodes_library.proxy.hosted_artifacts import HostedArtifactError
from griptape_nodes_library.proxy.output_size import estimate_prores_4444_bytes
from griptape_nodes_library.utils.ffmpeg_utils import describe_ffmpeg_failure, is_quarter_turned
from griptape_nodes_library.video.public_video_url_mixin import PublicVideoUrlMixin

logger = logging.getLogger("griptape_nodes")

__all__ = ["LTXVideoAlphaGen"]

MODEL = "ltx"
DEFAULT_MODEL = MODEL
MODEL_CHOICES = [MODEL]

PRORES_PARAMETER = "output_rgba_prores"
RGBA_OUTPUT_PARAMETER = "rgba_video"
RGBA_FILE_PARAMETER = "rgba_video_file"
INPUT_PARAMETER_LABEL = "input video"

# Input limits per the LTX alpha generation docs.
MAX_FRAMES = 3072
MAX_LONG_SIDE = 3840
MAX_SHORT_SIDE = 2160
# Longest allowed long:short aspect ratio, 32:9.
MAX_ASPECT_LONG = 32
MAX_ASPECT_SHORT = 9


class _ProbeResult(NamedTuple):
    """Probed input properties.

    For a quarter-turned video, width and height are swapped to the dimensions the viewer sees.
    Every limit check uses the long side, the short side or their ratio, so the swap changes no
    decision. It only orders the two numbers shown in error messages.
    """

    width: int
    height: int
    frames: int
    duration_seconds: float


def _first_url(result: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    """Return the first non-empty string URL in ``result`` among ``keys``."""
    for key in keys:
        url = result.get(key)
        if isinstance(url, str) and url:
            return url
    return None


class LTXVideoAlphaGen(PublicVideoUrlMixin, GriptapeProxyNode):
    """Generate an alpha matte for a video using LTX AI via Griptape Cloud model proxy.

    Inputs:
        - video (VideoUrlArtifact): Source video (required, sent as a public URL, or a base64 data URI as a fallback)
        - output_rgba_prores (bool): Also save the input as a ProRes 4444 video with the matte as its alpha channel

    Outputs:
        - generation_id (str): Griptape Cloud generation id
        - provider_response (dict): Response from API (latest polling response)
        - mask_video (VideoUrlArtifact): Greyscale alpha matte MP4 (white is foreground, black is transparent)
        - rgba_video (VideoUrlArtifact): ProRes 4444 `.mov` with straight alpha, when requested
        - was_successful (bool): Whether the generation succeeded
        - result_details (str): Details about the generation result or error
    """

    SERVICE_NAME = "Griptape"
    API_KEY_NAME = "GT_CLOUD_API_KEY"
    SUPPORTS_DIRECT_DOWNLOAD = True
    # Griptape Cloud downloads a URL input (up to 200 MB) to check and bill it while handling the
    # create call, allowing up to 150 s for that download, so wait longer than that for the response.
    SUBMIT_TIMEOUT_SECONDS = 200

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # INPUTS / PROPERTIES

        model_param = ParameterString(
            name="model",
            default_value=DEFAULT_MODEL,
            tooltip="Model to use for alpha generation",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
        )
        self.add_parameter(model_param)
        # License-policy dropdown: the component adds Options + refresh Button traits and
        # marks the models the license denies; the proxy base refuses a denied selection.
        self._model_access = ModelAccessComponent(
            node=self,
            parameter=model_param,
            model_choices=MODEL_CHOICES,
            default_model=DEFAULT_MODEL,
        )

        self.add_parameter(
            ParameterVideo(
                name="video",
                tooltip=(
                    "Video to generate an alpha matte for. Up to 3,072 frames, 3840 pixels on the long side "
                    "and 2160 on the short side, an aspect ratio of at most 32:9, and 200 MB. "
                    "Billed per second of video by its short side: up to 1080 pixels $0.25/s, "
                    "up to 1440 pixels $0.50/s, up to 2160 pixels $1.00/s."
                ),
                allowed_modes={ParameterMode.INPUT},
                hide_property=True,
                ui_options={"display_name": INPUT_PARAMETER_LABEL},
            )
        )

        self.add_parameter(
            ParameterBool(
                name=PRORES_PARAMETER,
                default_value=False,
                tooltip=(
                    "Also save the input video as a ProRes 4444 .mov with the matte as its alpha channel. "
                    "ProRes files are very large: about 42 MB per second at 1080p and 172 MB per second at 4K. "
                    "When off, the ProRes file is never downloaded, by this machine or by Griptape Cloud: the "
                    "matte is fetched straight from LTX."
                ),
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"display_name": "output RGBA ProRes"},
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
                name="mask_video",
                tooltip="Greyscale alpha matte MP4. White is foreground, black is transparent, greys are partly transparent.",
                allowed_modes={ParameterMode.OUTPUT, ParameterMode.PROPERTY},
                settable=False,
                ui_options={"pulse_on_run": True},
            )
        )

        self._output_file = ProjectFileParameter(
            node=self,
            name="output_file",
            default_filename="ltx_alpha_gen_mask.mp4",
        )
        self._output_file.add_parameter()

        self.add_parameter(
            ParameterVideo(
                name=RGBA_OUTPUT_PARAMETER,
                tooltip="ProRes 4444 .mov of the input with the matte as its straight alpha channel. Empty unless requested.",
                allowed_modes={ParameterMode.OUTPUT, ParameterMode.PROPERTY},
                settable=False,
                ui_options={"pulse_on_run": True},
            )
        )

        self._rgba_file = ProjectFileParameter(
            node=self,
            name=RGBA_FILE_PARAMETER,
            default_filename="ltx_alpha_gen_rgba.mov",
        )
        self._rgba_file.add_parameter()

        self._create_status_parameters(
            result_details_tooltip="Details about the alpha generation result or any errors",
            result_details_placeholder="Alpha generation status and details will appear here.",
            parameter_group_initially_collapsed=True,
        )

        # Probe results cached for the duration of one run, so the size estimate and the input
        # validation share one ffprobe. None outside a run: the cache never outlives it.
        self._probe_cache: dict[str, _ProbeResult | None] | None = None

        self._update_rgba_visibility()

        self.set_initial_node_size(height=400)

    def set_parameter_value(self, param_name: str, value: Any, **kwargs: Any) -> None:
        super().set_parameter_value(param_name, value, **kwargs)
        # The engine skips after_value_set when restoring a saved workflow (initial_setup), so the
        # ProRes outputs' visibility must be re-derived from the loaded toggle here.
        if kwargs.get("initial_setup", False) and param_name == PRORES_PARAMETER:
            self._update_rgba_visibility()

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        super().after_value_set(parameter, value)
        if parameter.name == PRORES_PARAMETER:
            self._update_rgba_visibility()

    def _update_rgba_visibility(self) -> None:
        names = [RGBA_OUTPUT_PARAMETER, RGBA_FILE_PARAMETER]
        if self._prores_requested():
            self.show_parameter_by_name(names)
        else:
            self.hide_parameter_by_name(names)

    def _prores_requested(self) -> bool:
        return bool(self.get_parameter_value(PRORES_PARAMETER))

    def _direct_download_enabled(self) -> bool:
        # LTX produces the ProRes cutout whether or not it is wanted, and hosting it would make
        # Griptape Cloud download and store gigabytes nobody reads. Without the ProRes the only
        # output is the small matte, so the run fetches it from LTX directly.
        return super()._direct_download_enabled() or not self._prores_requested()

    async def _process_generation(self) -> None:
        self._reset_video_uploads()
        self._probe_cache = {}
        try:
            await super()._process_generation()
        finally:
            self._probe_cache = None
            await self._cleanup_video_uploads()

    def _get_api_model_id(self) -> str:
        return "ltx:video-alpha-gen"

    @staticmethod
    def _extract_input_video_url(video_input: Any) -> str | None:
        return coerce_media_url_or_data_uri(video_input, kind="video")

    @staticmethod
    def _parse_frame_rate(stream: dict[str, Any]) -> float:
        for key in ("avg_frame_rate", "r_frame_rate"):
            raw = stream.get(key)
            if not isinstance(raw, str) or "/" not in raw:
                continue
            numerator, _, denominator = raw.partition("/")
            try:
                rate = float(numerator) / float(denominator)
            except (ValueError, ZeroDivisionError):
                continue
            if rate > 0:
                return rate
        return 0.0

    @staticmethod
    def _positive_float(value: Any) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0.0
        return number if number > 0 else 0.0

    def _probe_video(self, video_url: str) -> _ProbeResult | None:
        """Probe a video URL/data-URI with one ffprobe. Returns the dimensions, frames and duration, or None."""
        try:
            _, ffprobe_path = run.get_or_fetch_platform_executables_else_raise()

            # Project-saved inputs carry macro paths like {outputs}/clip.mp4 that ffprobe can't open
            location = resolve_media_location(video_url)

            cmd = [
                ffprobe_path,
                # "error" rather than "quiet" so the log below can report why the probe
                # failed; the JSON stays on stdout, diagnostics go to stderr.
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
                cmd, capture_output=True, text=True, check=True, timeout=60
            )

            data = json.loads(result.stdout)
            streams = data.get("streams", [])
            if not streams:
                return None

            stream = streams[0]
            width = int(stream.get("width") or 0)
            height = int(stream.get("height") or 0)
            if width <= 0 or height <= 0:
                return None
            # A quarter-turned clip is stored on its side. The swap decides nothing (every check is
            # orientation-agnostic); it only keeps the dimensions in error messages as the viewer sees them.
            if is_quarter_turned(stream):
                width, height = height, width

            fps = self._parse_frame_rate(stream)
            duration = self._positive_float(stream.get("duration")) or self._positive_float(
                (data.get("format") or {}).get("duration")
            )
            frames_str = stream.get("nb_frames")
            frames = int(frames_str) if frames_str and str(frames_str).isdigit() else 0
            if frames <= 0 and duration > 0 and fps > 0:
                frames = round(duration * fps)
            if duration <= 0 and frames > 0 and fps > 0:
                duration = frames / fps

            return _ProbeResult(width, height, frames, duration)
        except (
            subprocess.TimeoutExpired,
            subprocess.CalledProcessError,
            json.JSONDecodeError,
            FileLoadError,
            ValueError,
            KeyError,
            AttributeError,
            TypeError,
        ) as e:
            logger.debug("%s ffprobe failed to probe video: %s", self.name, describe_ffmpeg_failure(e))
            return None

    def _probe_video_cached(self, video_url: str) -> _ProbeResult | None:
        """`_probe_video`, memoised for the current run so a run probes each input once."""
        cache = self._probe_cache
        if cache is None:
            return self._probe_video(video_url)
        if video_url not in cache:
            cache[video_url] = self._probe_video(video_url)
        return cache[video_url]

    def _estimate_output_bytes(self) -> int | None:
        """Estimate the ProRes size, or None when it is not requested (the matte MP4 is small)."""
        if not self._prores_requested():
            return None
        video_url = self._extract_input_video_url(self.get_parameter_value("video"))
        if not video_url:
            return None
        probed = self._probe_video_cached(video_url)
        if probed is None:
            return None
        estimate = estimate_prores_4444_bytes(probed.width, probed.height, probed.duration_seconds)
        return estimate or None

    def _validate_video_input(self, video: Any) -> None:
        """Raise a NodeError when the video is missing or outside the LTX alpha generation limits.

        A video that cannot be probed is left for the proxy to judge.
        """
        label = INPUT_PARAMETER_LABEL
        if not video:
            msg = f"Connect a video to '{label}' to generate an alpha matte."
            raise NodeError(msg)

        video_url = self._extract_input_video_url(video)
        if not video_url:
            return

        probed = self._probe_video_cached(video_url)
        if probed is None:
            return

        width, height, frames, _ = probed
        long_side = max(width, height)
        short_side = min(width, height)

        if frames > MAX_FRAMES:
            msg = (
                f"The video in '{label}' has {frames} frames, but alpha generation accepts up to "
                f"{MAX_FRAMES}. Trim the video and try again."
            )
            raise NodeError(msg)
        if long_side > MAX_LONG_SIDE:
            msg = (
                f"The video in '{label}' is {width}x{height}. Its long side is {long_side} pixels, "
                f"but alpha generation accepts up to {MAX_LONG_SIDE}. Resize it and try again."
            )
            raise NodeError(msg)
        if short_side > MAX_SHORT_SIDE:
            msg = (
                f"The video in '{label}' is {width}x{height}. Its short side is {short_side} pixels, "
                f"but alpha generation accepts up to {MAX_SHORT_SIDE}. Resize it and try again."
            )
            raise NodeError(msg)
        if long_side * MAX_ASPECT_SHORT > short_side * MAX_ASPECT_LONG:
            msg = (
                f"The video in '{label}' is {width}x{height}, a wider aspect ratio than the "
                f"{MAX_ASPECT_LONG}:{MAX_ASPECT_SHORT} that alpha generation accepts. Crop it and try again."
            )
            raise NodeError(msg)

    async def _prepare_video_data_uri_async(self, video_input: Any) -> str | None:
        """Convert video input to a base64 data URI."""
        return await prepare_media_data_uri(video_input, kind="video", node_name=self.name)

    async def _build_payload(self) -> dict[str, Any]:
        video = self.get_parameter_value("video")
        # ffprobe runs in a worker thread so a slow remote input doesn't stall the event loop. With
        # the ProRes off, no pre-submit estimate has probed yet, so this is the run's first probe.
        await asyncio.to_thread(self._validate_video_input, video)

        # Tier 1: upload to Griptape Cloud and send LTX a public URL it fetches server-side.
        # Tier 2: base64 data URI with a pre-flight size guard.
        # The mixin raises its own user-facing messages (unreadable or too large).
        video_uri = await self._resolve_video_uri(video)
        self._warn_if_proxy_downloads_external_url(video, video_uri)

        # The proxy derives duration and the billing tier from the decoded video, so the
        # request only carries video_uri.
        return {"video_uri": video_uri}

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

    def _direct_download_result_urls(self, result_json: dict[str, Any]) -> list[tuple[str, str]]:
        """Map the LTX job result to the hosted-artifact order: the matte MP4 first, then the ProRes."""
        result = result_json.get("result") if isinstance(result_json, dict) else None
        if not isinstance(result, dict):
            return []
        # The live API returns matte_video_url / cutout_video_url; the preview docs (as of 2026-11-05)
        # still say video_url / rgba_video_url. Accept both, live spelling first.
        mask_url = _first_url(result, ("matte_video_url", "video_url"))
        if not mask_url:
            # Both outputs are videos told apart by position. Without the matte, the ProRes would land
            # at position 0 and be saved as the matte, so report no outputs and let the matte save fail.
            return []
        urls: list[tuple[str, str]] = [(ArtifactKind.VIDEO, mask_url)]
        rgba_url = _first_url(result, ("cutout_video_url", "rgba_video_url"))
        if rgba_url:
            urls.append((ArtifactKind.VIDEO, rgba_url))
        return urls

    async def _parse_result(self, _result_json: dict[str, Any], generation_id: str) -> None:
        """Save the matte MP4 and, when requested, the ProRes .mov.

        Both artifacts are videos, so they are told apart by position within the video kind: the
        matte is video 0 and the ProRes is video 1. A failed matte save fails the run. A missing or
        failed ProRes is reported in the status, but the matte is the primary output, so the run succeeds.
        """
        self.parameter_output_values[RGBA_OUTPUT_PARAMETER] = None
        if not await self._save_generated_media(
            generation_id,
            "mask_video",
            lambda v, n: VideoUrlArtifact(value=v, name=n),
            kind=ArtifactKind.VIDEO,
            position=0,
            media_kind="matte video",
        ):
            return

        rgba_note = await self._save_rgba_video(generation_id)
        self._set_status_results(
            was_successful=True,
            result_details=f"Alpha generation finished. The matte video was saved. {rgba_note}",
        )

    async def _save_rgba_video(self, generation_id: str) -> str:
        """Save the ProRes .mov when requested and return a sentence describing the outcome.

        The artifact is resolved once, so a missing ProRes costs no extra lookups.
        """
        if not self._prores_requested():
            return "The ProRes video was not requested."
        try:
            artifact = await self._hosted_artifact(generation_id, kind=ArtifactKind.VIDEO, position=1)
        except HostedArtifactError as e:
            logger.warning("%s: generation %s has no ProRes video: %s", self.name, generation_id, e)
            return "The ProRes video was not produced."
        try:
            saved = await self._save_artifact_to_project(artifact, generation_id, output_file=self._rgba_file)
        except Exception as e:
            logger.warning("%s could not save the ProRes video: %s", self.name, e)
            return f"The ProRes video could not be saved: {e}"
        self.parameter_output_values[RGBA_OUTPUT_PARAMETER] = VideoUrlArtifact(value=saved.location, name=saved.name)
        return "The ProRes video was saved."

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
        if isinstance(e, (NodeError, ValueError)):
            self._set_safe_defaults()
            self._set_status_results(was_successful=False, result_details=str(e))
            self._handle_failure_exception(e)
            return

        super()._handle_payload_build_error(e)

    def _set_safe_defaults(self) -> None:
        self.parameter_output_values["generation_id"] = ""
        self.parameter_output_values["provider_response"] = None
        self.parameter_output_values["mask_video"] = None
        self.parameter_output_values[RGBA_OUTPUT_PARAMETER] = None
