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
from griptape_nodes.exe_types.param_types.parameter_float import ParameterFloat
from griptape_nodes.exe_types.param_types.parameter_int import ParameterInt
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.exe_types.param_types.parameter_video import ParameterVideo
from griptape_nodes.traits.options import Options
from griptape_nodes.traits.slider import Slider

from griptape_nodes_library.proxy import GriptapeProxyNode
from griptape_nodes_library.utils.ffmpeg_utils import VideoMetadata
from griptape_nodes_library.video.topaz_video_common import (
    TIER_1080P_MAX_PIXELS,
    derive_container,
    frame_count,
    probe_source,
    to_even,
)

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

    The frame cap, the creative controls and the 4K rate spread are all properties of
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

# Topaz's documented hard output ceiling for Starlight (distinct from the 1080p/4K
# billing-tier boundary above): https://docs.topazlabs.com/video-ai/project-starlight
STARLIGHT_MAX_OUTPUT_PIXELS = 3840 * 2160

# What the 4K tier costs relative to the 1080p tier, as UI prose.
RATE_SPREADS = {
    TopazVideoFamily.STARLIGHT: "2.2x",
    TopazVideoFamily.ASTRA: "1.67x",
}

# How the two families compare *to each other* per frame, at the same tier. Griptape
# Cloud bills Astra 13,000 credits/frame at 1080p against Starlight's 4,992, and 21,667
# against 10,908 at 4K -- so Astra is the pricier model by 2.6x and 2.0x respectively.
# (griptape-cloud credits migrations 0087 and 0088.)
#
# This is spelled out in the badge because the RATE_SPREADS above are *within* a family
# and invite exactly the wrong read on their own: Astra's 1.67x is the smaller number
# but the more expensive model.
ASTRA_VS_STARLIGHT_1080P = "2.6x"
ASTRA_VS_STARLIGHT_4K = "2x"

# Topaz publishes a hard output ceiling for Starlight and none for Astra, and the proxy
# enforces neither -- so Astra maps to None rather than to an invented limit that would
# reject jobs Topaz may well accept.
MAX_OUTPUT_PIXELS: dict[TopazVideoFamily, int | None] = {
    TopazVideoFamily.STARLIGHT: STARLIGHT_MAX_OUTPUT_PIXELS,
    TopazVideoFamily.ASTRA: None,
}

_COST_BADGE_TEMPLATE = (
    "{family} is metered **per frame**, not per second, and costs far more than "
    "Topaz's non-generative video models.\n\n"
    "Two rate tiers, picked by output pixel area:\n"
    "- **1080p** (up to 1920x1080) — the cheaper tier\n"
    "- **4K** (anything larger) — roughly **{spread}** the 1080p rate\n\n"
    "{cross_model}\n\n"
    "A 10-second 30fps clip is 300 frames whichever tier it lands in.{extra}\n\n"
    "[Topaz model pricing](https://developer.topazlabs.com/getting-started/model-pricing)"
)

COST_BADGE_MESSAGES = {
    TopazVideoFamily.STARLIGHT: _COST_BADGE_TEMPLATE.format(
        family=TopazVideoFamily.STARLIGHT,
        spread=RATE_SPREADS[TopazVideoFamily.STARLIGHT],
        cross_model=(
            f"Starlight is the cheaper of the two models here: Astra 2 costs about "
            f"**{ASTRA_VS_STARLIGHT_1080P}** as much per frame at 1080p, and about "
            f"**{ASTRA_VS_STARLIGHT_4K}** as much at 4K."
        ),
        extra="",
    ),
    TopazVideoFamily.ASTRA: _COST_BADGE_TEMPLATE.format(
        family=TopazVideoFamily.ASTRA,
        spread=RATE_SPREADS[TopazVideoFamily.ASTRA],
        # The second sentence is doing real work: the 1.67x above is smaller than
        # Starlight's 2.2x, so quoting it alone reads as "Astra is cheaper" when Astra
        # is in fact the pricier model on both tiers.
        cross_model=(
            f"Astra costs about **{ASTRA_VS_STARLIGHT_1080P} Starlight** per frame at 1080p, and "
            f"about **{ASTRA_VS_STARLIGHT_4K} Starlight** at 4K. The "
            f"{RATE_SPREADS[TopazVideoFamily.ASTRA]} above is Astra's own 4K premium, not a "
            "comparison with Starlight."
        ),
        extra=(
            f"\n\nTopaz caps an Astra job at {MAX_ASTRA_FRAMES:,} frames — or "
            f"**{MAX_ASTRA_FRAMES_WITH_PROMPT}** once a prompt is set."
        ),
    ),
}


class ResizeMode(StrEnum):
    """How the output resolution is derived from the source."""

    WIDTH = "width"
    HEIGHT = "height"
    WIDTH_HEIGHT = "width and height"
    PERCENTAGE = "percentage"


class TopazVideoUpscale(GriptapeProxyNode):
    """Upscale a video with Topaz Starlight Precise or Astra 2 via the Griptape Cloud model proxy.

    Both families share the source probing, the resize modes and the per-frame billing
    tiers. Astra additionally accepts four creative controls (prompt, creativity, sharp,
    realism), which are shown only while an Astra model is selected.

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
            "Upscale a video using Topaz Starlight Precise or Astra 2 via the Griptape model "
            "proxy. Billed per frame -- see the cost note on the model parameter."
        )

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
        # All four run here, after every add_parameter: show/hide silently no-ops on a
        # name that does not exist yet, so refreshing earlier would leave Astra's
        # controls hidden forever with no error to point at.
        self._update_model_visibility()
        self._update_cost_badge()
        self._update_prompt_badge()
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
                self._update_cost_badge()
                self._update_prompt_badge()
                self._update_tier_badge()
            case "resize_mode":
                self._update_resize_visibility(value)
                self._update_tier_badge()
            case "target_size" | "target_width" | "target_height" | "percentage":
                self._update_tier_badge()
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

    def _update_cost_badge(self) -> None:
        param = self.get_parameter_by_name("model")
        if param is None:
            return

        param.set_badge(
            variant="warning",
            title="Billed per frame",
            message=COST_BADGE_MESSAGES[self._family()],
        )

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
        family = self._family()
        spread = RATE_SPREADS[family]
        max_pixels = MAX_OUTPUT_PIXELS[family]

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
                if max_pixels is not None and pixels > max_pixels:
                    param.set_badge(
                        variant="error",
                        title="Exceeds Topaz's output limit",
                        message=(
                            f"{width}x{height} is {pixels:,} pixels, over {family}'s "
                            f"{max_pixels:,}-pixel (3840x2160) hard limit. The node will fail "
                            "rather than submit this to Topaz."
                        ),
                    )
                elif pixels > TIER_1080P_MAX_PIXELS:
                    # Astra reaches here with no ceiling of its own, so say so rather
                    # than let an unusually large request look fully sanctioned.
                    unbounded = (
                        ""
                        if max_pixels is not None
                        else f" Topaz documents no output ceiling for {family}, so a very large "
                        "request may still be refused by the provider."
                    )
                    param.set_badge(
                        variant="warning",
                        title="4K billing tier",
                        message=(
                            f"{width}x{height} is {pixels:,} pixels, over the "
                            f"{TIER_1080P_MAX_PIXELS:,}-pixel 1080p limit. This bills at the 4K "
                            f"per-frame rate, about {spread} the 1080p rate.{unbounded}"
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
                threshold = math.isqrt(int(TIER_1080P_MAX_PIXELS // (multiplier * multiplier)))
                param.set_badge(
                    variant="note",
                    title="Billing tier depends on the source",
                    message=(
                        f"At {percentage}%, any source larger than roughly {threshold}x{threshold} "
                        f"produces a 4K-tier output (over {TIER_1080P_MAX_PIXELS:,} output pixels), "
                        f"at about {spread} the 1080p rate."
                    ),
                )
            case _:
                msg = f"Unknown resize mode: {mode!r}"
                raise ValueError(msg)

    # -- source probing ----------------------------------------------------

    def _probe_source(self, video_input: Any) -> VideoMetadata:
        """Probe the source with ffprobe. See ``topaz_video_common.probe_source``."""
        return probe_source(video_input, node_name=self.name)

    def _derive_container(self, video_input: Any) -> str:
        """Map the input to Topaz's container enum. See ``topaz_video_common.derive_container``."""
        return derive_container(video_input, node_name=self.name)

    @staticmethod
    def _frame_count(metadata: VideoMetadata) -> int:
        """Derive the source frame count. See ``topaz_video_common.frame_count``."""
        return frame_count(metadata)

    @staticmethod
    def _to_even(value: int) -> int:
        """Round down to an even number. See ``topaz_video_common.to_even``."""
        return to_even(value)

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
