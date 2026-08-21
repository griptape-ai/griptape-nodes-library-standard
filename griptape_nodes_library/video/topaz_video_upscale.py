from __future__ import annotations

import asyncio
import logging
import math
from enum import StrEnum
from typing import Any

from griptape.artifacts.video_url_artifact import VideoUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.param_components.artifact_url.public_artifact_url_parameter import (
    PublicArtifactUrlParameter,
)
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_types.parameter_dict import ParameterDict
from griptape_nodes.exe_types.param_types.parameter_int import ParameterInt
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.exe_types.param_types.parameter_video import ParameterVideo
from griptape_nodes.files.file import File, FileLoadError
from griptape_nodes.traits.options import Options
from griptape_nodes.traits.slider import Slider

from griptape_nodes_library.media import coerce_media_url_or_data_uri
from griptape_nodes_library.proxy import GriptapeProxyNode
from griptape_nodes_library.utils.ffmpeg_utils import VideoMetadata, extract_video_metadata_structured

logger = logging.getLogger("griptape_nodes")

__all__ = ["TopazVideoUpscale"]

MODEL_MAPPING = {
    "Starlight Precise 2.6": "topaz-video-slp-2.6",
    "Starlight Precise 2.5": "topaz-video-slp-2.5",
}

DEFAULT_MODEL = "Starlight Precise 2.6"

# Topaz caps a Starlight job at 9000 frames. The proxy clamps the *billed* volume to
# match, so exceeding it does not overcharge -- but Topaz still rejects the job, and
# failing here is cheaper than discovering it after the upload.
MAX_STARLIGHT_FRAMES = 9000

# Starlight has two rate tiers, chosen by output pixel *area*, not height. A portrait
# 1080x1920 output bills as 1080p; 1921x1080 already bills as 4K.
# https://developer.topazlabs.com/getting-started/model-pricing
STARLIGHT_1080P_MAX_PIXELS = 1920 * 1080

# Topaz's documented hard output ceiling for Starlight (distinct from the 1080p/4K
# billing-tier boundary above): https://docs.topazlabs.com/video-ai/project-starlight
STARLIGHT_MAX_OUTPUT_PIXELS = 3840 * 2160

COST_BADGE_MESSAGE = (
    "Starlight is metered **per frame**, not per second, and costs far more than "
    "Topaz's non-generative video models.\n\n"
    "Two rate tiers, picked by output pixel area:\n"
    "- **1080p** (up to 1920x1080) — the cheaper tier\n"
    "- **4K** (anything larger) — roughly **2.2x** the 1080p rate\n\n"
    "A 10-second 30fps clip is 300 frames whichever tier it lands in.\n\n"
    "[Topaz model pricing](https://developer.topazlabs.com/getting-started/model-pricing)"
)


class ResizeMode(StrEnum):
    """How the output resolution is derived from the source."""

    WIDTH = "width"
    HEIGHT = "height"
    WIDTH_HEIGHT = "width and height"
    PERCENTAGE = "percentage"


class TopazVideoUpscale(GriptapeProxyNode):
    """Upscale a video with Topaz Starlight Precise via the Griptape Cloud model proxy.

    Inputs:
        - video (VideoUrlArtifact): source video to upscale (sent as a base64 data URI)

    Outputs:
        - video_output (VideoUrlArtifact): the upscaled video, saved to project storage
        - generation_id (str): Griptape Cloud generation id
        - provider_response (dict): the provider's result payload
        - was_successful (bool) / result_details (str): execution status
    """

    SERVICE_NAME = "Griptape"
    API_KEY_NAME = "GT_CLOUD_API_KEY"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.category = "video"
        self.description = (
            "Upscale a video using Topaz Starlight Precise via the Griptape model proxy. "
            "Billed per frame -- see the cost note on the model parameter."
        )

        # INPUTS / PROPERTIES

        model_param = ParameterString(
            name="model",
            default_value=DEFAULT_MODEL,
            tooltip="Starlight Precise model to upscale with",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            traits={Options(choices=list(MODEL_MAPPING.keys()))},
        )
        model_param.set_badge(
            variant="warning",
            title="Billed per frame",
            message=COST_BADGE_MESSAGE,
        )
        self.add_parameter(model_param)

        # Topaz downloads the source itself rather than receiving it in the request
        # body -- a video is far too large to base64 into JSON. This uploads the
        # input to Griptape Cloud storage and hands back a public URL, which
        # `_build_payload` passes through as `source.external`. An input that is
        # already a public http(s) URL is passed through without re-uploading.
        self._public_video_url_parameter = PublicArtifactUrlParameter(
            node=self,
            artifact_url_parameter=ParameterVideo(
                name="video",
                tooltip="Source video to upscale",
                allowed_modes={ParameterMode.INPUT},
                hide_property=True,
                ui_options={"display_name": "input video"},
            ),
            disclaimer_message="Topaz Labs fetches the video from this URL to perform the upscale.",
        )
        self._public_video_url_parameter.add_input_parameters()

        resize_mode_param = ParameterString(
            name="resize_mode",
            default_value=ResizeMode.PERCENTAGE,
            tooltip="How to derive the output resolution from the source. Output size selects the billing tier.",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            traits={Options(choices=[m.value for m in ResizeMode])},
        )
        self.add_parameter(resize_mode_param)

        target_size_param = ParameterInt(
            name="target_size",
            default_value=1920,
            tooltip="Target size in pixels for the width or height mode. The other dimension scales to preserve aspect ratio.",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            hide=True,
        )
        target_size_param.add_trait(Slider(min_val=2, max_val=3840))
        self.add_parameter(target_size_param)

        target_width_param = ParameterInt(
            name="target_width",
            default_value=3840,
            tooltip="Output width in pixels. Rounded down to an even number.",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            hide=True,
        )
        target_width_param.add_trait(Slider(min_val=2, max_val=3840))
        self.add_parameter(target_width_param)

        target_height_param = ParameterInt(
            name="target_height",
            default_value=2160,
            tooltip="Output height in pixels. Rounded down to an even number.",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            hide=True,
        )
        target_height_param.add_trait(Slider(min_val=2, max_val=2160))
        self.add_parameter(target_height_param)

        percentage_param = ParameterInt(
            name="percentage",
            default_value=200,
            tooltip="Upscale the source resolution by this percentage (e.g. 200 doubles it).",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
        )
        percentage_param.add_trait(Slider(min_val=100, max_val=400))
        self.add_parameter(percentage_param)

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
                name="video_output",
                tooltip="The upscaled video",
                allowed_modes={ParameterMode.OUTPUT, ParameterMode.PROPERTY},
                settable=False,
                ui_options={"pulse_on_run": True},
            )
        )

        self._output_file = ProjectFileParameter(
            node=self,
            name="output_file",
            default_filename="topaz_video_upscale.mp4",
        )
        self._output_file.add_parameter()

        self._create_status_parameters(
            result_details_tooltip="Details about the upscale result or any errors",
            result_details_placeholder="Upscale status and details will appear here.",
            parameter_group_initially_collapsed=True,
        )

        self.set_initial_node_size(height=400)
        self._update_tier_badge()

    # -- lifecycle ---------------------------------------------------------

    async def aprocess(self) -> None:
        try:
            await super().aprocess()
        finally:
            # Only once polling has finished: Topaz fetches the source from this
            # URL when the job starts, so deleting it any earlier would pull the
            # video out from under a queued job.
            self._public_video_url_parameter.delete_uploaded_artifact()

    # -- model routing -----------------------------------------------------

    def _get_api_model_id(self) -> str:
        model_name = self.get_parameter_value("model") or DEFAULT_MODEL
        return MODEL_MAPPING.get(model_name, MODEL_MAPPING[DEFAULT_MODEL])

    # -- UI reactions ------------------------------------------------------

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        super().after_value_set(parameter, value)

        if parameter.name == "resize_mode":
            match value:
                case ResizeMode.WIDTH | ResizeMode.HEIGHT:
                    self.show_parameter_by_name("target_size")
                    self.hide_parameter_by_name(["target_width", "target_height", "percentage"])
                case ResizeMode.WIDTH_HEIGHT:
                    self.hide_parameter_by_name(["target_size", "percentage"])
                    self.show_parameter_by_name(["target_width", "target_height"])
                case ResizeMode.PERCENTAGE:
                    self.hide_parameter_by_name(["target_size", "target_width", "target_height"])
                    self.show_parameter_by_name("percentage")
                case _:
                    msg = f"Unknown resize mode: {value!r}"
                    raise ValueError(msg)

        if parameter.name in ("resize_mode", "target_size", "target_width", "target_height", "percentage"):
            self._update_tier_badge()

    def _update_tier_badge(self) -> None:
        """Show which billing tier the current settings land in.

        For width-and-height and percentage modes the tier is exact -- it depends only
        on numbers already on the node. For width/height-only modes it depends on the
        source's aspect ratio, which is not known until the node runs, so say what to
        watch out for instead of guessing.
        """
        param = self.get_parameter_by_name("resize_mode")
        if param is None:
            return

        mode = self.get_parameter_value("resize_mode") or ResizeMode.PERCENTAGE

        match mode:
            case ResizeMode.WIDTH | ResizeMode.HEIGHT:
                param.set_badge(
                    variant="note",
                    title="Billing tier depends on the source",
                    message=(
                        "The other dimension scales to preserve the source's aspect ratio, so the "
                        "output pixel count -- and which billing tier it lands in -- isn't known "
                        "until the node runs."
                    ),
                )
            case ResizeMode.WIDTH_HEIGHT:
                width = self.get_parameter_value("target_width") or 0
                height = self.get_parameter_value("target_height") or 0
                if width <= 0 or height <= 0:
                    param.clear_badge()
                    return

                pixels = width * height
                if pixels > STARLIGHT_MAX_OUTPUT_PIXELS:
                    param.set_badge(
                        variant="error",
                        title="Exceeds Topaz's output limit",
                        message=(
                            f"{width}x{height} is {pixels:,} pixels, over Topaz's "
                            f"{STARLIGHT_MAX_OUTPUT_PIXELS:,}-pixel (3840x2160) hard limit. The node "
                            "will fail rather than submit this to Topaz."
                        ),
                    )
                elif pixels > STARLIGHT_1080P_MAX_PIXELS:
                    param.set_badge(
                        variant="warning",
                        title="4K billing tier",
                        message=(
                            f"{width}x{height} is {pixels:,} pixels, over the "
                            f"{STARLIGHT_1080P_MAX_PIXELS:,}-pixel 1080p limit. This bills at the 4K "
                            "per-frame rate, about 2.2x the 1080p rate."
                        ),
                    )
                else:
                    param.set_badge(
                        variant="info",
                        title="1080p billing tier",
                        message=f"{width}x{height} is {pixels:,} pixels, within the cheaper 1080p tier.",
                    )
            case ResizeMode.PERCENTAGE:
                percentage = self.get_parameter_value("percentage") or 200
                multiplier = percentage / 100
                threshold = math.isqrt(int(STARLIGHT_1080P_MAX_PIXELS // (multiplier * multiplier)))
                param.set_badge(
                    variant="note",
                    title="Billing tier depends on the source",
                    message=(
                        f"At {percentage}%, any source larger than roughly {threshold}x{threshold} "
                        f"produces a 4K-tier output (over {STARLIGHT_1080P_MAX_PIXELS:,} output pixels), "
                        "at about 2.2x the 1080p rate."
                    ),
                )
            case _:
                msg = f"Unknown resize mode: {mode!r}"
                raise ValueError(msg)

    # -- source probing ----------------------------------------------------

    def _probe_source(self, video_input: Any) -> VideoMetadata:
        """Resolve the input to a local path and probe it with ffprobe.

        Resolving through ``File`` first is what makes ``{inputs}/clip.mp4`` macro
        paths work; handing the raw value to ffprobe silently fails for those.
        """
        video_url = coerce_media_url_or_data_uri(video_input, kind="video")
        if not video_url:
            msg = f"{self.name} could not resolve the input video."
            raise ValueError(msg)

        try:
            resolved_path = File(video_url).resolve()
        except FileLoadError as e:
            msg = f"{self.name} could not resolve video path {video_url!r}: {e}"
            raise ValueError(msg) from e

        return extract_video_metadata_structured(str(resolved_path))

    @staticmethod
    def _frame_count(metadata: VideoMetadata) -> int:
        """Derive the source frame count, which the proxy requires and never defaults.

        ffprobe omits ``nb_frames`` for plenty of ordinary MP4s, so fall back to
        duration x frame rate rather than letting the request 400 downstream.
        """
        nb_frames = metadata.frame_details.optional_nb_frames
        if nb_frames and nb_frames > 0:
            return nb_frames

        duration = metadata.file_details.optional_duration
        frame_rate = metadata.frame_details.frame_rate
        if duration and duration > 0 and frame_rate > 0:
            return math.ceil(duration * frame_rate)

        return 0

    @staticmethod
    def _to_even(value: int) -> int:
        """Round down to an even number -- odd dimensions break yuv420 encoding."""
        return max(2, value - (value % 2))

    def _output_resolution(self, source_width: int, source_height: int) -> tuple[int, int]:
        mode = self.get_parameter_value("resize_mode") or ResizeMode.PERCENTAGE

        match mode:
            case ResizeMode.WIDTH:
                target = self.get_parameter_value("target_size") or 0
                if target <= 0:
                    msg = f"{self.name} needs a positive target size when resize_mode is width (got {target})."
                    raise ValueError(msg)
                height = round(source_height * (target / source_width))
                width, height = self._to_even(int(target)), self._to_even(height)
            case ResizeMode.HEIGHT:
                target = self.get_parameter_value("target_size") or 0
                if target <= 0:
                    msg = f"{self.name} needs a positive target size when resize_mode is height (got {target})."
                    raise ValueError(msg)
                width = round(source_width * (target / source_height))
                width, height = self._to_even(width), self._to_even(int(target))
            case ResizeMode.WIDTH_HEIGHT:
                target_width = self.get_parameter_value("target_width") or 0
                target_height = self.get_parameter_value("target_height") or 0
                if target_width <= 0 or target_height <= 0:
                    msg = (
                        f"{self.name} needs a positive target width and height when resize_mode is "
                        f"'width and height' (got {target_width}x{target_height})."
                    )
                    raise ValueError(msg)
                width, height = self._to_even(int(target_width)), self._to_even(int(target_height))
            case ResizeMode.PERCENTAGE:
                pct = self.get_parameter_value("percentage") or 0
                if pct <= 0:
                    msg = f"{self.name} needs a positive percentage (got {pct})."
                    raise ValueError(msg)
                width = self._to_even(int(source_width * pct / 100))
                height = self._to_even(int(source_height * pct / 100))
            case _:
                msg = f"Unknown resize mode: {mode!r}"
                raise ValueError(msg)

        if width * height > STARLIGHT_MAX_OUTPUT_PIXELS:
            msg = (
                f"{self.name}: computed output {width}x{height} exceeds Topaz's "
                f"{STARLIGHT_MAX_OUTPUT_PIXELS:,}-pixel (3840x2160) hard limit."
            )
            raise ValueError(msg)

        return width, height

    # -- request -----------------------------------------------------------

    async def _build_payload(self) -> dict[str, Any]:
        video = self.get_parameter_value("video")
        if not video:
            msg = f"{self.name} requires an input video to upscale."
            raise ValueError(msg)

        # ffprobe is synchronous and can take a moment on a long clip; keep it off
        # the event loop.
        metadata = await asyncio.to_thread(self._probe_source, video)

        source_width = metadata.dimensions.width
        source_height = metadata.dimensions.height

        frame_count = self._frame_count(metadata)
        if frame_count <= 0:
            msg = (
                f"{self.name} could not determine the frame count of the input video. "
                "Topaz requires it, and neither nb_frames nor duration x frame rate was "
                "available from the file."
            )
            raise ValueError(msg)

        if frame_count > MAX_STARLIGHT_FRAMES:
            msg = (
                f"{self.name}: the input video has {frame_count} frames, over Starlight's "
                f"{MAX_STARLIGHT_FRAMES}-frame limit. Trim or split the video first."
            )
            raise ValueError(msg)

        output_width, output_height = self._output_resolution(source_width, source_height)

        # Upload after probing: the probe needs the original local path, and there
        # is no point paying for an upload if the file turns out to be unusable.
        video_url = self._public_video_url_parameter.get_public_url_for_parameter()
        if not video_url:
            msg = f"{self.name} could not produce a public URL for the input video."
            raise ValueError(msg)

        source: dict[str, Any] = {
            "container": "mp4",
            "frameCount": frame_count,
            "frameRate": metadata.frame_details.frame_rate,
            "resolution": {"width": source_width, "height": source_height},
            # Topaz fetches the video from this URL itself. `provider` is required and
            # must be one of r2/s3/web-url -- it labels the transport, not the host, so
            # `web-url` is correct for the Azure Blob SAS URL the upload hands back.
            "external": {"provider": "web-url", "presignedUrl": video_url},
        }
        if metadata.file_details.optional_duration:
            source["duration"] = metadata.file_details.optional_duration
        if metadata.file_details.optional_file_size:
            source["size"] = metadata.file_details.optional_file_size

        logger.info(
            "%s upscaling %dx%d (%d frames) to %dx%d",
            self.name,
            source_width,
            source_height,
            frame_count,
            output_width,
            output_height,
        )

        # `filters` is deliberately omitted: the proxy synthesizes
        # [{"model": <routed code>}] when it is absent, and sending our own only
        # risks the "filters[].model must match the requested model" rejection --
        # the code there is the id minus its "topaz-video-" prefix.
        return {
            "source": source,
            "output": {"resolution": {"width": output_width, "height": output_height}},
        }

    # -- result ------------------------------------------------------------

    async def _parse_result(self, result_json: dict[str, Any], _generation_id: str) -> None:
        raw_bytes = result_json.get("raw_bytes")
        if isinstance(raw_bytes, (bytes, bytearray)):
            await self._handle_binary_video_response(bytes(raw_bytes))
            return

        # `download.url` is Topaz's documented shape for a completed generation.
        await self._download_and_save(
            result_json["download"]["url"],
            "video_output",
            lambda v, n: VideoUrlArtifact(value=v, name=n),
            media_kind="video",
            action="upscaled",
        )

    async def _handle_binary_video_response(self, video_bytes: bytes) -> None:
        """Save video bytes served directly by the proxy rather than via a URL."""
        if not video_bytes:
            self._set_safe_defaults()
            self._set_status_results(
                was_successful=False,
                result_details=f"{self.name}: the upscale completed but no video data was received.",
            )
            return

        try:
            dest = self._output_file.build_file()
            saved = await dest.awrite_bytes(video_bytes)
        except (OSError, PermissionError) as e:
            logger.error("%s failed to save the upscaled video: %s", self.name, e)
            self._set_safe_defaults()
            self._set_status_results(
                was_successful=False,
                result_details=f"{self.name}: the upscale succeeded but saving the video failed: {e}",
            )
            return

        self.parameter_output_values["video_output"] = VideoUrlArtifact(value=saved.location, name=saved.name)
        self._set_status_results(
            was_successful=True,
            result_details=f"Upscale successful. Video saved as {saved.name}.",
        )

    def _handle_payload_build_error(self, e: Exception) -> None:
        if isinstance(e, ValueError):
            self._set_safe_defaults()
            self._set_status_results(was_successful=False, result_details=str(e))
            return

        super()._handle_payload_build_error(e)

    def _set_safe_defaults(self) -> None:
        self.parameter_output_values["generation_id"] = ""
        self.parameter_output_values["provider_response"] = None
        self.parameter_output_values["video_output"] = None
