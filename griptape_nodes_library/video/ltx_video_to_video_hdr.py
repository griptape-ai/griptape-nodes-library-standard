from __future__ import annotations

import asyncio
import json
import logging
import subprocess
from typing import Any

from griptape.artifacts import VideoUrlArtifact
from griptape_nodes.exe_types.core_types import NodeError, Parameter, ParameterMode
from griptape_nodes.exe_types.param_components.model_access_component import ModelAccessComponent
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_types.parameter_dict import ParameterDict
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.exe_types.param_types.parameter_video import ParameterVideo
from griptape_nodes.files.file import File, FileLoadError
from griptape_nodes.traits.options import Options

# static_ffmpeg is dynamically installed by the library loader at runtime
from static_ffmpeg import run  # type: ignore[import-untyped]

from griptape_nodes_library.media import coerce_media_url_or_data_uri, prepare_media_data_uri, resolve_media_location
from griptape_nodes_library.media import exr_sequence as exr_sequence_helpers
from griptape_nodes_library.proxy import ArtifactKind, GriptapeProxyNode
from griptape_nodes_library.proxy.hosted_artifacts import HostedArtifactError
from griptape_nodes_library.proxy.output_size import estimate_exr_zip_bytes
from griptape_nodes_library.utils.ffmpeg_utils import describe_ffmpeg_failure
from griptape_nodes_library.video.public_video_url_mixin import PublicVideoUrlMixin

logger = logging.getLogger("griptape_nodes")

__all__ = ["LTXVideoToVideoHDR"]

# Migrates values saved before the dropdown stored the provider's own model id: old
# display labels and catalog keys.
LEGACY_MODEL_VALUES = {
    "LTX 2.3 Pro": "ltx-2-3-pro",
    "gtc_ltx_2_3_pro": "ltx-2-3-pro",
}

HDR_MODEL = "ltx-hdr-2"
LEGACY_HDR_MODEL = "ltx-2-3-pro"
DEFAULT_MODEL = HDR_MODEL
MODEL_CHOICES = [HDR_MODEL, LEGACY_HDR_MODEL]

OUTPUT_COLOURSPACES = ["acescg", "acescct", "srgb_linear", "aces2065-1"]
DEFAULT_OUTPUT_COLOURSPACE = "acescg"
COLOURSPACE_PARAMETER = "output_colorspace"

# Input frame-count limits per LTX HDR upscale docs, keyed by billing tier.
# https://docs.ltx.io/api-documentation/api-reference/async-video-generation/submit-video-to-video-hdr
# (bucket, max_long_side, max_short_side, max_frames)  # noqa: ERA001
_LTX_HDR_INPUT_TIERS: tuple[tuple[str, int, int, int], ...] = (
    ("1080p", 1920, 1080, 273),
    ("1440p", 2560, 1440, 193),
    ("2160p", 3840, 2160, 65),
)


class _HdrModelAccess(ModelAccessComponent):
    """Model dropdown that marks the legacy `ltx-2-3-pro` row as deprecated.

    `deprecated_values` only migrates retired values to a current choice, and `ltx-2-3-pro` must stay
    a selectable choice here, so the row is flagged in the dropdown's labels instead. Rebuilt on every
    `refresh()`, so the override lives in `_build_ui_options` rather than being applied once.
    """

    def _build_ui_options(self) -> dict[str, Any]:
        ui_options = super()._build_ui_options()
        for row in ui_options.get("data", []):
            if row.get("name") == LEGACY_HDR_MODEL:
                row["label"] = f"{row.get('label', LEGACY_HDR_MODEL)} (deprecated)"
        return ui_options


class LTXVideoToVideoHDR(PublicVideoUrlMixin, GriptapeProxyNode):
    """Upgrade an SDR video to HDR using LTX AI via Griptape Cloud model proxy.

    Inputs:
        - video (VideoUrlArtifact): SDR source video (required, sent as a public URL, or a base64 data URI as a fallback)
        - model (str): `ltx-hdr-2` (LTX HDR (ACES)) or the deprecated `ltx-2-3-pro`
        - output_colorspace (str): Colour space of the EXR frames. Only for `ltx-hdr-2`; default `acescg`

    Outputs:
        - generation_id (str): Griptape Cloud generation id
        - provider_response (dict): Response from API (latest polling response)
        - output_file (str): Path to the saved ZIP of per-frame EXR images
        - project_path (str): Project macro path of the saved ZIP
        - preview_video (VideoUrlArtifact): The 10-bit HLG MP4 preview, when the generation has one
        - exr_sequence (Sequence): The unpacked EXR frames
        - colour_space (str): OCIO colour space name of the EXR frames
        - was_successful (bool): Whether the generation succeeded
        - result_details (str): Details about the generation result or error
    """

    SERVICE_NAME = "Griptape"
    API_KEY_NAME = "GT_CLOUD_API_KEY"
    SUPPORTS_DIRECT_DOWNLOAD = True

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # INPUTS / PROPERTIES

        # `ltx-hdr-2` is the current model; `ltx-2-3-pro` is the legacy option, routed to the same endpoint.
        model_param = ParameterString(
            name="model",
            default_value=DEFAULT_MODEL,
            tooltip="Model to use for video-to-video HDR upscale",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
        )
        self.add_parameter(model_param)
        # License-policy dropdown: the component adds Options + refresh Button traits and
        # marks the models the license denies; the proxy base refuses a denied selection.
        self._model_access = _HdrModelAccess(
            node=self,
            parameter=model_param,
            model_choices=MODEL_CHOICES,
            default_model=DEFAULT_MODEL,
            deprecated_values=LEGACY_MODEL_VALUES,
        )

        self.add_parameter(
            ParameterString(
                name=COLOURSPACE_PARAMETER,
                default_value=DEFAULT_OUTPUT_COLOURSPACE,
                tooltip="Colour space of the EXR frames. Only used by LTX HDR (ACES).",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=OUTPUT_COLOURSPACES)},
                ui_options={"display_name": "output colour space"},
            )
        )

        self.add_parameter(
            ParameterVideo(
                name="video",
                tooltip="SDR source video to upgrade to HDR",
                allowed_modes={ParameterMode.INPUT},
                hide_property=True,
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

        self.add_parameter(
            ParameterVideo(
                name="preview_video",
                tooltip="10-bit HLG MP4 preview of the HDR result. Empty when the generation has no preview.",
                allowed_modes={ParameterMode.OUTPUT, ParameterMode.PROPERTY},
                settable=False,
                ui_options={"pulse_on_run": True},
            )
        )

        self._preview_file = ProjectFileParameter(
            node=self,
            name="preview_video_file",
            default_filename="ltx_video_to_video_hdr.mp4",
        )
        self._preview_file.add_parameter()

        self._exr_outputs = exr_sequence_helpers.add_exr_sequence_outputs(self)

        self._create_status_parameters(
            result_details_tooltip="Details about the HDR upscale result or any errors",
            result_details_placeholder="HDR upscale status and details will appear here.",
            parameter_group_initially_collapsed=True,
        )

        # Probe results cached for the duration of one run, so the size estimate and the input
        # validation share one ffprobe. None outside a run: the cache never outlives it.
        self._probe_cache: dict[str, tuple[int, int, int] | None] | None = None

        self._update_colourspace_visibility()

        self.set_initial_node_size(height=400)

    def set_parameter_value(self, param_name: str, value: Any, **kwargs: Any) -> None:
        super().set_parameter_value(param_name, value, **kwargs)
        # The engine skips after_value_set when restoring a saved workflow (initial_setup), so the
        # colour space dropdown's visibility must be re-derived from the loaded model here.
        if kwargs.get("initial_setup", False) and param_name == "model":
            self._update_colourspace_visibility()

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        super().after_value_set(parameter, value)
        if parameter.name == "model":
            self._update_colourspace_visibility()

    def _update_colourspace_visibility(self) -> None:
        if self._selected_model() == HDR_MODEL:
            self.show_parameter_by_name(COLOURSPACE_PARAMETER)
        else:
            self.hide_parameter_by_name(COLOURSPACE_PARAMETER)

    def _selected_model(self) -> str:
        """The stored model, migrating a legacy label so it compares against the choices."""
        value = self.get_parameter_value("model")
        if not isinstance(value, str):
            return DEFAULT_MODEL
        return LEGACY_MODEL_VALUES.get(value, value)

    def _selected_colourspace(self) -> str | None:
        """The requested EXR colour space, or None for the legacy model (which always returns ACEScg)."""
        if self._selected_model() != HDR_MODEL:
            return None
        return self.get_parameter_value(COLOURSPACE_PARAMETER) or DEFAULT_OUTPUT_COLOURSPACE

    async def _process_generation(self) -> None:
        self._reset_video_uploads()
        self._probe_cache = {}
        try:
            await super()._process_generation()
        finally:
            self._probe_cache = None
            await self._cleanup_video_uploads()

    def _get_api_model_id(self) -> str:
        return f"{self._get_selected_model_id()}:video-to-video-hdr"

    @staticmethod
    def _extract_input_video_url(video_input: Any) -> str | None:
        return coerce_media_url_or_data_uri(video_input, kind="video")

    def _probe_video(self, video_url: str) -> tuple[int, int, int] | None:
        """Probe a video URL/data-URI with ffprobe. Returns (width, height, frames) or None."""
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
                "-count_frames",
                "-select_streams",
                "v:0",
                location,
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
            FileLoadError,
            ValueError,
            KeyError,
        ) as e:
            logger.debug("%s ffprobe failed to probe video: %s", self.name, describe_ffmpeg_failure(e))
            return None

    def _probe_video_cached(self, video_url: str) -> tuple[int, int, int] | None:
        """`_probe_video`, memoised for the current run so a run probes each input once."""
        cache = self._probe_cache
        if cache is None:
            return self._probe_video(video_url)
        if video_url not in cache:
            cache[video_url] = self._probe_video(video_url)
        return cache[video_url]

    def _estimate_output_bytes(self) -> int | None:
        """Estimate the EXR ZIP size from the input's dimensions and frame count (HLG MP4 ignored: small)."""
        video_url = self._extract_input_video_url(self.get_parameter_value("video"))
        if not video_url:
            return None
        probed = self._probe_video_cached(video_url)
        if probed is None:
            return None
        width, height, frames = probed
        estimate = estimate_exr_zip_bytes(frames, width, height)
        return estimate or None

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
            return "Connect a video to 'input video' to upscale."

        video_url = self._extract_input_video_url(video)
        if not video_url:
            return None

        probed = self._probe_video_cached(video_url)
        if probed is None:
            # If we can't probe, let the proxy return the error.
            return None

        width, height, frames = probed
        tier = self._pick_input_tier(width, height)
        if tier is None:
            return (
                f"The input video is {width}x{height}, larger than LTX HDR upscale accepts (up to 4K). "
                "Resize it and try again."
            )

        bucket, max_frames = tier
        if frames > 0 and frames > max_frames:
            return (
                f"The input video has {frames} frames at {width}x{height} "
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

        # Tier 1: upload to Griptape Cloud and send LTX a public URL it fetches server-side.
        # Tier 2: base64 data URI with a pre-flight size guard.
        # The mixin raises its own user-facing messages (unreadable or too large).
        video_uri = await self._resolve_video_uri(video)

        # The proxy derives duration AND the billing resolution tier server-side
        # from the decoded video, so the request payload only carries video_uri (and, for
        # `ltx-hdr-2`, the colour space; the legacy model has no such option).
        payload: dict[str, Any] = {"video_uri": video_uri}
        colourspace = self._selected_colourspace()
        if colourspace is not None:
            payload["output_colorspace"] = colourspace
        return payload

    def _direct_download_result_urls(self, result_json: dict[str, Any]) -> list[tuple[str, str]]:
        """Map the LTX job result to the hosted-artifact order: EXR ZIP first, then the HLG MP4."""
        result = result_json.get("result") if isinstance(result_json, dict) else None
        if not isinstance(result, dict):
            return []
        urls: list[tuple[str, str]] = []
        for key, kind in (("exr_frames_url", ArtifactKind.ARCHIVE), ("video_url", ArtifactKind.VIDEO)):
            url = result.get(key)
            if isinstance(url, str) and url:
                urls.append((kind, url))
        return urls

    async def _parse_result(self, _result_json: dict[str, Any], generation_id: str) -> None:
        """Save the EXR ZIP and the HLG preview MP4, then unpack the ZIP into an EXR sequence.

        The ZIP and the EXR sequence are the node's results, so a failure saving either fails the run.
        The preview MP4 is a convenience: a missing or failed preview is reported but does not.
        """
        if not await self._save_generated_media(
            generation_id,
            "project_path",
            lambda v, _n: v,
            kind=ArtifactKind.ARCHIVE,
            media_kind="archive",
        ):
            self._clear_derived_outputs()
            return

        zip_location = self.parameter_output_values.get("project_path")
        preview_note = await self._save_preview_video(generation_id)

        try:
            await self._write_exr_sequence(zip_location)
        except Exception as e:
            self._clear_derived_outputs(keep_preview=True)
            logger.warning("%s could not unpack the EXR ZIP: %s", self.name, e)
            self._set_status_results(
                was_successful=False,
                result_details=(
                    f"The EXR ZIP was saved, but it could not be unpacked into an EXR sequence: {e} "
                    f"Use the ZIP in 'output_file' instead.{preview_note}"
                ),
            )
            return

        self._set_status_results(
            was_successful=True,
            result_details=f"HDR upscale finished: the EXR sequence and EXR ZIP were saved.{preview_note}",
        )

    async def _save_preview_video(self, generation_id: str) -> str:
        """Save the HLG MP4 to its own destination and return a sentence describing the outcome.

        Generations made before the upgraded endpoint have no second artifact; that is not an error.
        The artifact is resolved once, so a missing preview costs no extra lookups.
        """
        self.parameter_output_values["preview_video"] = None
        try:
            artifact = await self._hosted_artifact(generation_id, kind=ArtifactKind.VIDEO)
        except HostedArtifactError as e:
            logger.info("%s: generation %s has no preview video, skipping it: %s", self.name, generation_id, e)
            return " This generation has no preview video."
        try:
            saved = await self._save_artifact_to_project(artifact, generation_id, output_file=self._preview_file)
        except Exception as e:
            logger.warning("%s could not save the preview video: %s", self.name, e)
            return f" The preview video could not be saved: {e}"
        self.parameter_output_values["preview_video"] = VideoUrlArtifact(value=saved.location, name=saved.name)
        return " The preview video was saved."

    async def _write_exr_sequence(self, zip_location: Any) -> None:
        """Unpack the saved ZIP from its local path (never buffered in memory) into the EXR sequence output."""
        if not isinstance(zip_location, str) or not zip_location:
            msg = "The saved ZIP location is unknown."
            raise NodeError(msg)
        zip_path = await asyncio.to_thread(File(zip_location).resolve)
        outputs = self._exr_outputs
        sequence = await exr_sequence_helpers.write_exr_zip_to_sequence(
            zip_path, outputs.file_sequence.build_sequence()
        )
        self.parameter_output_values[outputs.sequence_param_name] = sequence
        self.parameter_output_values[outputs.colour_space_param_name] = exr_sequence_helpers.ocio_colour_space_name(
            self._selected_colourspace()
        )

    def _clear_derived_outputs(self, *, keep_preview: bool = False) -> None:
        outputs = self._exr_outputs
        self.parameter_output_values[outputs.sequence_param_name] = None
        self.parameter_output_values[outputs.colour_space_param_name] = ""
        if not keep_preview:
            self.parameter_output_values["preview_video"] = None

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
        if isinstance(e, ValueError):
            self._set_safe_defaults()
            self._set_status_results(was_successful=False, result_details=str(e))
            self._handle_failure_exception(e)
            return

        super()._handle_payload_build_error(e)

    def _set_safe_defaults(self) -> None:
        self.parameter_output_values["generation_id"] = ""
        self.parameter_output_values["provider_response"] = None
        self.parameter_output_values["project_path"] = ""
        self._clear_derived_outputs()
