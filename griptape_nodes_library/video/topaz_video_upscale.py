from __future__ import annotations

import asyncio
import logging
import math
from enum import StrEnum
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from griptape.artifacts.video_url_artifact import VideoUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.param_components.artifact_url.public_artifact_url_parameter import (
    PublicArtifactUrlParameter,
)
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_types.parameter_dict import ParameterDict
from griptape_nodes.exe_types.param_types.parameter_float import ParameterFloat
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
    "Astra 2": "topaz-video-ast-2",
}

DEFAULT_MODEL = "Starlight Precise 2.6"


class TopazVideoFamily(StrEnum):
    """Which Topaz video model family a selection belongs to.

    The frame cap, the creative controls and the output ceiling are all properties of
    the family rather than of the individual version. The values are the display
    spelling so they interpolate straight into badge and error prose.
    """

    STARLIGHT = "Starlight"
    ASTRA = "Astra"


# Kept as a second table rather than folded into MODEL_MAPPING: a flat
# `display name -> api id` MODEL_MAPPING is the convention across every video node in
# this library, and reshaping it here would cost more than it buys for three entries.
# `test_every_model_has_a_registered_family` guards the two against drifting.
MODEL_FAMILIES = {
    "Starlight Precise 2.6": TopazVideoFamily.STARLIGHT,
    "Starlight Precise 2.5": TopazVideoFamily.STARLIGHT,
    "Astra 2": TopazVideoFamily.ASTRA,
}

# Astra's creative controls. These ride *inside* a `filters` entry rather than at the
# top level of the payload. Starlight accepts none of them.
ASTRA_FILTER_PARAMS = ("prompt", "creativity", "sharp", "realism")

# Topaz caps a Starlight job at 9000 frames. The proxy clamps the *billed* volume to
# match, so exceeding it does not overcharge -- but Topaz still rejects the job, and
# failing here is cheaper than discovering it after the upload.
MAX_STARLIGHT_FRAMES = 9000

MAX_ASTRA_FRAMES = 9000
# A prompt drops Astra's ceiling by 20x. As with Starlight the proxy only clamps what
# it *bills*; the job still reaches Topaz, which rejects it.
MAX_ASTRA_FRAMES_WITH_PROMPT = 450

# Topaz's source.container enum is exactly mp4/mov/mkv:
# https://developer.topazlabs.com/reference/api-endpoints/video/create-request.md
#
# ffprobe's format_name can't tell these apart -- mp4 and mov both report
# "mov,mp4,m4a,3gp,3g2,mj2", and mkv and webm both report "matroska,webm" (which
# isn't even a valid Topaz value) -- so container is derived from the file
# extension (or, for a data URI, the MIME subtype) instead. See _derive_container.
CONTAINER_BY_TOKEN: dict[str, str] = {
    "mp4": "mp4",
    "m4v": "mp4",
    "mov": "mov",
    "qt": "mov",
    "quicktime": "mov",
    "mkv": "mkv",
    "matroska": "mkv",
    "x-matroska": "mkv",
}

# Assumed when the input carries no extension or MIME subtype at all -- see
# _derive_container for why that case is not an error.
DEFAULT_CONTAINER = "mp4"

# Topaz's documented hard output ceiling for Starlight:
# https://docs.topazlabs.com/video-ai/project-starlight
STARLIGHT_MAX_OUTPUT_PIXELS = 3840 * 2160

# Topaz publishes a hard output ceiling for Starlight and none for Astra, and the proxy
# enforces neither -- so Astra maps to None rather than to an invented limit that would
# reject jobs Topaz may well accept.
MAX_OUTPUT_PIXELS: dict[TopazVideoFamily, int | None] = {
    TopazVideoFamily.STARLIGHT: STARLIGHT_MAX_OUTPUT_PIXELS,
    TopazVideoFamily.ASTRA: None,
}


class ResizeMode(StrEnum):
    """How the output resolution is derived from the source."""

    WIDTH = "width"
    HEIGHT = "height"
    WIDTH_HEIGHT = "width and height"
    PERCENTAGE = "percentage"


class TopazVideoUpscale(GriptapeProxyNode):
    """Upscale a video with Topaz Starlight Precise or Astra 2 via the Griptape Cloud model proxy.

    Both families share the source probing and the resize modes. Astra additionally
    accepts four creative controls (prompt, creativity, sharp, realism), which are shown
    only while an Astra model is selected.

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
        self.description = "Upscale a video using Topaz Starlight Precise or Astra 2 via the Griptape model proxy."

        # INPUTS / PROPERTIES

        model_param = ParameterString(
            name="model",
            default_value=DEFAULT_MODEL,
            tooltip="Topaz video model to upscale with. Astra adds creative controls; Starlight does not.",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            traits={Options(choices=list(MODEL_MAPPING.keys()))},
        )
        self.add_parameter(model_param)

        # Astra only. Hidden by default because the default model is Starlight, and
        # declaring them hidden matters beyond tidiness: a visible prompt box under
        # Starlight would silently do nothing, since Starlight sends no filters at all.
        prompt_param = ParameterString(
            name="prompt",
            default_value="",
            tooltip=(
                "Astra only. A description of the detail to generate. Setting one drops Topaz's "
                f"frame cap from {MAX_ASTRA_FRAMES:,} to {MAX_ASTRA_FRAMES_WITH_PROMPT}."
            ),
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            multiline=True,
            placeholder_text="Optional: describe the detail Astra should generate...",
            hide=True,
        )
        self.add_parameter(prompt_param)

        creativity_param = ParameterFloat(
            name="creativity",
            default_value=0.5,
            tooltip="Astra only. 0.0 stays faithful to the source; 1.0 invents the most new detail.",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            hide=True,
        )
        creativity_param.add_trait(Slider(min_val=0.0, max_val=1.0))
        self.add_parameter(creativity_param)

        sharp_param = ParameterFloat(
            name="sharp",
            default_value=0.5,
            tooltip="Astra only. Pre-enhance sharpness: 0.0 blurs, 0.5 passes through unchanged, 1.0 sharpens.",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            hide=True,
        )
        sharp_param.add_trait(Slider(min_val=0.0, max_val=1.0))
        self.add_parameter(sharp_param)

        # Topaz documents a default for `sharp` but not for `realism`; 0.5 is our own
        # midpoint choice, picked so the slider is WYSIWYG rather than to match an
        # unpublished provider default.
        realism_param = ParameterFloat(
            name="realism",
            default_value=0.5,
            tooltip="Astra only. Biases the generated detail toward photorealism (0.0-1.0).",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            hide=True,
        )
        realism_param.add_trait(Slider(min_val=0.0, max_val=1.0))
        self.add_parameter(realism_param)

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
            tooltip="How to derive the output resolution from the source.",
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
        # All three run here, after every add_parameter: show/hide silently no-ops on a
        # name that does not exist yet, so refreshing earlier would leave Astra's
        # controls hidden forever with no error to point at.
        self._update_model_visibility()
        self._update_prompt_badge()
        self._update_limit_badge()

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

    def _family(self) -> TopazVideoFamily:
        """Which family the selected model belongs to.

        Raises rather than defaulting: an unmapped name means MODEL_MAPPING and
        MODEL_FAMILIES have drifted, and quietly billing at another model's rate is a
        worse outcome than a loud failure. The Options trait makes an off-list value
        unreachable from the UI anyway.
        """
        model_name = self.get_parameter_value("model") or DEFAULT_MODEL
        family = MODEL_FAMILIES.get(model_name)
        if family is None:
            msg = f"{self.name}: no model family is registered for {model_name!r}."
            raise ValueError(msg)
        return family

    def _has_prompt(self) -> bool:
        """Whether a prompt will actually reach Topaz.

        Mirrors the proxy's own truthiness test on ``filters[].prompt`` so the cap
        enforced here can never disagree with the cap the proxy bills against.
        """
        return bool((self.get_parameter_value("prompt") or "").strip())

    # -- UI reactions ------------------------------------------------------

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        super().after_value_set(parameter, value)

        # Unlike the other match statements in this file, the wildcard here must be a
        # no-op rather than a raise: `parameter.name` is an open set -- `video`,
        # `output_file` and every status parameter also land here.
        match parameter.name:
            case "model":
                self._update_model_visibility()
                self._update_prompt_badge()
                self._update_limit_badge()
            case "resize_mode":
                self._update_resize_visibility(value)
                self._update_limit_badge()
            case "target_size" | "target_width" | "target_height" | "percentage":
                self._update_limit_badge()
            case "prompt":
                self._update_prompt_badge()
            case _:
                return

    def _update_resize_visibility(self, mode: Any) -> None:
        match mode:
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
                msg = f"Unknown resize mode: {mode!r}"
                raise ValueError(msg)

    def _update_model_visibility(self) -> None:
        """Show the creative controls only for the family that accepts them."""
        match self._family():
            case TopazVideoFamily.ASTRA:
                self.show_parameter_by_name(list(ASTRA_FILTER_PARAMS))
            case TopazVideoFamily.STARLIGHT:
                self.hide_parameter_by_name(list(ASTRA_FILTER_PARAMS))
            case family:
                msg = f"Unknown model family: {family!r}"
                raise ValueError(msg)

    def _update_prompt_badge(self) -> None:
        """State Astra's prompt-dependent frame cap while the graph is still being wired.

        The source frame count is not knowable in the editor -- the video usually
        arrives over a connection and is not probed until the node runs -- so this
        states the cap rather than testing against it. ``_build_payload`` remains the
        authority that actually enforces it.
        """
        param = self.get_parameter_by_name("prompt")
        if param is None or self._family() is not TopazVideoFamily.ASTRA:
            return

        if self._has_prompt():
            param.set_badge(
                variant="warning",
                title=f"Caps this job at {MAX_ASTRA_FRAMES_WITH_PROMPT} frames",
                message=(
                    f"Topaz caps a *prompted* Astra job at {MAX_ASTRA_FRAMES_WITH_PROMPT} frames "
                    f"(~15 seconds at 30fps). Clear the prompt to allow up to {MAX_ASTRA_FRAMES:,}."
                ),
            )
        else:
            param.set_badge(
                variant="note",
                title=f"A prompt drops the cap to {MAX_ASTRA_FRAMES_WITH_PROMPT} frames",
                message=(
                    f"Without a prompt Astra accepts up to {MAX_ASTRA_FRAMES:,} frames. Adding one "
                    f"drops Topaz's cap to {MAX_ASTRA_FRAMES_WITH_PROMPT} (~15 seconds at 30fps)."
                ),
            )

    def _update_limit_badge(self) -> None:
        """Warn when the requested output exceeds the family's hard pixel ceiling.

        Only width-and-height mode can be checked here: it is the one mode whose output
        pixel count follows from numbers already on the node. The other modes scale off
        the source, which is not probed until the node runs, and ``_resolve_output_size``
        is the authority that catches them.
        """
        param = self.get_parameter_by_name("resize_mode")
        if param is None:
            return

        family = self._family()
        max_pixels = MAX_OUTPUT_PIXELS[family]
        mode = self.get_parameter_value("resize_mode") or ResizeMode.PERCENTAGE
        width = self.get_parameter_value("target_width") or 0
        height = self.get_parameter_value("target_height") or 0

        # `mode` arrives from the UI as a plain string, so compare by value, not identity.
        if mode != ResizeMode.WIDTH_HEIGHT or max_pixels is None or width <= 0 or height <= 0:
            param.clear_badge()
            return

        pixels = width * height
        if pixels <= max_pixels:
            param.clear_badge()
            return

        param.set_badge(
            variant="error",
            title="Exceeds Topaz's output limit",
            message=(
                f"{width}x{height} is {pixels:,} pixels, over {family}'s "
                f"{max_pixels:,}-pixel (3840x2160) hard limit. The node will fail "
                "rather than submit this to Topaz."
            ),
        )

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

    def _derive_container(self, video_input: Any) -> str:
        """Map the input video to Topaz's exact ``source.container`` enum (mp4/mov/mkv).

        Deliberately independent of ``_probe_source``/``VideoMetadata``: it only needs
        the raw parameter value, and calling the same cheap, pure
        ``coerce_media_url_or_data_uri`` helper here keeps this check from being
        silently bypassed by tests that stub ``_probe_source`` wholesale.

        Strict about a wrong answer, lenient about no answer. A recognizable but
        unsupported token (``.webm``, ``.avi``) raises, because that input really is
        wrong and failing here costs nothing. A *missing* token -- a raw storage key, a
        signed URL that strips the filename -- carries no signal either way, so it falls
        back to ``DEFAULT_CONTAINER`` rather than reject a URL that Topaz would
        likely have accepted.
        """
        video_url = coerce_media_url_or_data_uri(video_input, kind="video") or ""
        if video_url.startswith("data:"):
            header = video_url.removeprefix("data:").split(",", 1)[0]
            token = header.split(";", 1)[0].split("/", 1)[-1].lower()
        else:
            token = Path(urlsplit(video_url).path).suffix.lstrip(".").lower()

        container = CONTAINER_BY_TOKEN.get(token)
        if container is not None:
            return container

        if token:
            msg = f"{self.name}: Topaz only accepts mp4, mov, or mkv source video, but got {token!r}."
            raise ValueError(msg)

        logger.warning(
            "%s could not determine a container for %s (no extension or MIME subtype); assuming %s.",
            self.name,
            video_url,
            DEFAULT_CONTAINER,
        )
        return DEFAULT_CONTAINER

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

        # Astra maps to None here: Topaz publishes no output ceiling for it and the
        # proxy enforces none, so there is nothing to check against.
        max_pixels = MAX_OUTPUT_PIXELS[self._family()]
        if max_pixels is not None and width * height > max_pixels:
            msg = (
                f"{self.name}: computed output {width}x{height} exceeds Topaz's "
                f"{max_pixels:,}-pixel (3840x2160) hard limit."
            )
            raise ValueError(msg)

        return width, height

    # -- request -----------------------------------------------------------

    async def _build_payload(self) -> dict[str, Any]:
        video = self.get_parameter_value("video")
        if not video:
            msg = f"{self.name} requires an input video to upscale."
            raise ValueError(msg)

        container = self._derive_container(video)

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

        max_frames = self._max_frames()
        if frame_count > max_frames:
            msg = (
                f"{self.name}: the input video has {frame_count} frames, over {self._family()}'s "
                f"{max_frames}-frame limit{self._frame_cap_hint()}. Trim or split the video first."
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
            "container": container,
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

        payload: dict[str, Any] = {
            "source": source,
            "output": {"resolution": {"width": output_width, "height": output_height}},
        }

        filters = self._build_filters()
        if filters is not None:
            payload["filters"] = filters

        return payload

    def _build_filters(self) -> list[dict[str, Any]] | None:
        """Astra's creative controls, as the single ``filters`` entry Topaz expects.

        No ``model`` key is sent. The proxy passes filter dicts through verbatim and
        stamps the routed code onto the first entry itself when none carries one, so
        omitting it sidesteps the "filters[].model must match the requested model"
        rejection -- and means this node never has to know that the code is the id
        minus its "topaz-video-" prefix.

        Returns None rather than an empty list for Starlight: the proxy rejects
        ``filters: []`` outright, and omitting the key is what Starlight does today.
        """
        match self._family():
            case TopazVideoFamily.STARLIGHT:
                return None
            case TopazVideoFamily.ASTRA:
                # float() because a value arriving over a connection or a workflow
                # round-trip can be an int, and `json.dumps(1)` emits `1` against a
                # field Topaz documents as a decimal.
                creative: dict[str, Any] = {
                    "creativity": float(self.get_parameter_value("creativity")),
                    "sharp": float(self.get_parameter_value("sharp")),
                    "realism": float(self.get_parameter_value("realism")),
                }
                # Omitted rather than sent blank: the proxy decides whether the job is
                # "prompted" -- and which frame cap to bill against -- from the
                # truthiness of this key.
                prompt = (self.get_parameter_value("prompt") or "").strip()
                if prompt:
                    creative["prompt"] = prompt
                return [creative]
            case family:
                msg = f"Unknown model family: {family!r}"
                raise ValueError(msg)

    def _max_frames(self) -> int:
        match self._family():
            case TopazVideoFamily.STARLIGHT:
                return MAX_STARLIGHT_FRAMES
            case TopazVideoFamily.ASTRA:
                return MAX_ASTRA_FRAMES_WITH_PROMPT if self._has_prompt() else MAX_ASTRA_FRAMES
            case family:
                msg = f"Unknown model family: {family!r}"
                raise ValueError(msg)

    def _frame_cap_hint(self) -> str:
        """Name the way out when the cap is the prompt's doing rather than the clip's."""
        if self._family() is TopazVideoFamily.ASTRA and self._has_prompt():
            return f" for a prompted job (clearing the prompt raises it to {MAX_ASTRA_FRAMES:,})"
        return ""

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
            self._handle_failure_exception(e)
            return

        super()._handle_payload_build_error(e)

    def _set_safe_defaults(self) -> None:
        self.parameter_output_values["generation_id"] = ""
        self.parameter_output_values["provider_response"] = None
        self.parameter_output_values["video_output"] = None
