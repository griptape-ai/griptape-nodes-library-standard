from __future__ import annotations

import json
import logging
import subprocess
from typing import Any

from griptape_nodes.exe_types.core_types import ParameterMode
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_types.parameter_dict import ParameterDict
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.exe_types.param_types.parameter_video import ParameterVideo
from griptape_nodes.traits.options import Options

# static_ffmpeg is dynamically installed by the library loader at runtime
from static_ffmpeg import run  # type: ignore[import-untyped]

from griptape_nodes_library.media import coerce_media_url_or_data_uri, prepare_media_data_uri
from griptape_nodes_library.proxy import GriptapeProxyNode

logger = logging.getLogger("griptape_nodes")

__all__ = ["LTXVideoToVideoHDR"]

MODEL_MAPPING = {
    "LTX 2.3 Pro": "ltx-2-3-pro",
}

# Input frame-count limits per LTX HDR upscale docs, keyed by billing tier.
# https://docs.ltx.video/api-documentation/api-reference/async-video-generation/submit-video-to-video-hdr
# (bucket, max_long_side, max_short_side, max_frames)  # noqa: ERA001
_LTX_HDR_INPUT_TIERS: tuple[tuple[str, int, int, int], ...] = (
    ("1080p", 1920, 1080, 181),
    ("1440p", 2560, 1440, 101),
    ("2160p", 3840, 2160, 41),
)


class LTXVideoToVideoHDR(GriptapeProxyNode):
    """Upgrade an SDR video to HDR using LTX AI via Griptape Cloud model proxy.

    Inputs:
        - video (VideoUrlArtifact): SDR source video (required, sent as base64 data URI)

    Outputs:
        - generation_id (str): Griptape Cloud generation id
        - provider_response (dict): Response from API (latest polling response)
        - output_file (str): Path to the saved ZIP of per-frame EXR images
        - project_path (str): Project macro path of the saved ZIP
        - was_successful (bool): Whether the generation succeeded
        - result_details (str): Details about the generation result or error
    """

    SERVICE_NAME = "Griptape"
    API_KEY_NAME = "GT_CLOUD_API_KEY"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # INPUTS / PROPERTIES

        # Model parameter — HDR upscale is only offered on ltx-2-3-pro per the pricing page.
        self.add_parameter(
            ParameterString(
                name="model",
                default_value="LTX 2.3 Pro",
                tooltip="Model to use for video-to-video HDR upscale",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=list(MODEL_MAPPING.keys()))},
            )
        )

        self.add_parameter(
            ParameterVideo(
                name="video",
                tooltip="SDR source video to upgrade to HDR",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"display_name": "input video"},
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

        self._output_file = ProjectFileParameter(
            node=self,
            name="output_file",
            default_filename="ltx_video_to_video_hdr.zip",
        )
        self._output_file.add_parameter()

        self.add_parameter(
            ParameterString(
                name="project_path",
                tooltip="Project macro path of the saved EXR-frames ZIP.",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

        self._create_status_parameters(
            result_details_tooltip="Details about the HDR upscale result or any errors",
            result_details_placeholder="HDR upscale status and details will appear here.",
            parameter_group_initially_collapsed=True,
        )

    def _get_api_model_id(self) -> str:
        model_name = self.get_parameter_value("model") or "LTX 2.3 Pro"
        model_id = MODEL_MAPPING.get(model_name, "ltx-2-3-pro")
        return f"{model_id}:video-to-video-hdr"

    @staticmethod
    def _extract_input_video_url(video_input: Any) -> str | None:
        return coerce_media_url_or_data_uri(video_input, kind="video")

    def _probe_video(self, video_url: str) -> tuple[int, int, int] | None:
        """Probe a video URL/data-URI with ffprobe. Returns (width, height, frames) or None."""
        try:
            _, ffprobe_path = run.get_or_fetch_platform_executables_else_raise()

            cmd = [
                ffprobe_path,
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_streams",
                "-count_frames",
                "-select_streams",
                "v:0",
                video_url,
            ]

            result = subprocess.run(  # noqa: S603
                cmd, capture_output=True, text=True, check=True, timeout=60
            )

            stream_data = json.loads(result.stdout)
            streams = stream_data.get("streams", [])
            if not streams:
                return None

            stream = streams[0]
            width = int(stream.get("width") or 0)
            height = int(stream.get("height") or 0)

            # Prefer nb_read_frames (from -count_frames), fall back to nb_frames.
            frames_str = stream.get("nb_read_frames") or stream.get("nb_frames")
            frames = int(frames_str) if frames_str and str(frames_str).isdigit() else 0

            if width <= 0 or height <= 0:
                return None

            return width, height, frames
        except (
            subprocess.TimeoutExpired,
            subprocess.CalledProcessError,
            json.JSONDecodeError,
            ValueError,
            KeyError,
        ) as e:
            logger.debug("%s ffprobe failed to probe video: %s", self.name, e)
            return None

    @staticmethod
    def _pick_input_tier(width: int, height: int) -> tuple[str, int] | None:
        """Pick the smallest LTX HDR tier that can contain (width, height).

        Returns (bucket_name, max_frames) or None if the input exceeds 4K.
        """
        long_side = max(width, height)
        short_side = min(width, height)
        for bucket, max_long, max_short, max_frames in _LTX_HDR_INPUT_TIERS:
            if long_side <= max_long and short_side <= max_short:
                return bucket, max_frames
        return None

    def _validate_video_input(self, video: Any) -> str | None:
        """Validate video is provided and fits within LTX HDR input frame-count limits."""
        if not video:
            return f"{self.name} requires an input video for HDR upscale."

        video_url = self._extract_input_video_url(video)
        if not video_url:
            return None

        probed = self._probe_video(video_url)
        if probed is None:
            # If we can't probe, let the proxy return the error.
            return None

        width, height, frames = probed
        tier = self._pick_input_tier(width, height)
        if tier is None:
            return (
                f"{self.name}: Input video resolution ({width}x{height}) exceeds the "
                f"maximum supported LTX HDR upscale input (up to 4K)."
            )

        bucket, max_frames = tier
        if frames > 0 and frames > max_frames:
            return (
                f"{self.name}: Input video has {frames} frames at {width}x{height} "
                f"({bucket} tier), which exceeds the LTX HDR upscale limit of "
                f"{max_frames} frames for this resolution."
            )

        return None

    async def _prepare_video_data_uri_async(self, video_input: Any) -> str | None:
        """Convert video input to a base64 data URI."""
        return await prepare_media_data_uri(video_input, kind="video", node_name=self.name)

    async def _build_payload(self) -> dict[str, Any]:
        video = self.get_parameter_value("video")
        if error := self._validate_video_input(video):
            raise ValueError(error)

        try:
            video_data_uri = await self._prepare_video_data_uri_async(video)
        except Exception as e:
            logger.error("%s failed to process video: %s", self.name, e)
            video_data_uri = None

        if not video_data_uri:
            msg = f"{self.name} failed to process input video."
            raise ValueError(msg)

        # The proxy derives duration AND the billing resolution tier server-side
        # from the decoded video, so the request payload only carries video_uri.
        return {"video_uri": video_data_uri}

    async def _parse_result(self, result_json: dict[str, Any], generation_id: str) -> None:
        zip_bytes = result_json.get("raw_bytes")
        if not isinstance(zip_bytes, (bytes, bytearray)):
            msg = f"{self.name} generation completed but no ZIP data received."
            raise TypeError(msg)

        await self._handle_completion_async(bytes(zip_bytes), generation_id)

    async def _handle_completion_async(self, zip_bytes: bytes, generation_id: str) -> None:
        """Save the EXR-frames ZIP to storage and surface its path."""
        if not zip_bytes:
            self._set_safe_defaults()
            self._set_status_results(
                was_successful=False,
                result_details=f"{self.name} generation completed but no ZIP data received.",
            )
            return

        try:
            dest = self._output_file.build_file()
            saved = await dest.awrite_bytes(zip_bytes)
            self.parameter_output_values["project_path"] = saved.location
            logger.info("%s saved EXR-frames ZIP as %s", self.name, saved.name)
            self._set_status_results(
                was_successful=True,
                result_details=f"HDR upscale successful. EXR frames ZIP saved as {saved.name}.",
            )
        except (OSError, PermissionError) as e:
            logger.error("%s failed to save ZIP to storage: %s", self.name, e)
            self._set_safe_defaults()
            self._set_status_results(
                was_successful=False,
                result_details=f"HDR upscale completed but failed to save ZIP to storage: {e}",
            )

    def _extract_error_message(self, response_json: dict[str, Any]) -> str:  # noqa: C901, PLR0912
        if not response_json:
            return f"{self.name} generation failed with no error details provided by API."

        status = str(response_json.get("status") or "").lower()
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
                message = f"{error}: {details}"
            elif error:
                message = error
            elif details:
                message = details
            else:
                message = f"Generation {status or 'failed'} with no details provided"

            return f"{self.name} generation {status or 'failed'}: {message}"

        error = response_json.get("error")
        if error:
            if isinstance(error, dict):
                message = error.get("message") or error.get("type") or str(error)
                return f"{self.name} request failed: {message}"
            if isinstance(error, str):
                return f"{self.name} request failed: {error}"

        return f"{self.name} generation failed.\n\nFull API response:\n{response_json}"

    def _handle_payload_build_error(self, e: Exception) -> None:
        if isinstance(e, ValueError):
            self._set_safe_defaults()
            self._set_status_results(was_successful=False, result_details=str(e))
            return

        super()._handle_payload_build_error(e)

    def _handle_api_key_validation_error(self, e: ValueError) -> None:
        self._set_safe_defaults()
        self._set_status_results(was_successful=False, result_details=str(e))
        logger.error("%s API key validation failed: %s", self.name, e)

    def _set_safe_defaults(self) -> None:
        self.parameter_output_values["generation_id"] = ""
        self.parameter_output_values["provider_response"] = None
        self.parameter_output_values["project_path"] = ""
