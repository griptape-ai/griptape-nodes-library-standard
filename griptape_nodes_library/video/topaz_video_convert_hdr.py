from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from griptape.artifacts.video_url_artifact import VideoUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.param_components.artifact_url.public_artifact_url_parameter import (
    PublicArtifactUrlParameter,
)
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_types.parameter_dict import ParameterDict
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.exe_types.param_types.parameter_video import ParameterVideo
from griptape_nodes.traits.options import Options

from griptape_nodes_library.proxy import GriptapeProxyNode
from griptape_nodes_library.utils.ffmpeg_utils import VideoMetadata
from griptape_nodes_library.video.topaz_video_common import (
    derive_container,
    frame_count,
    probe_source,
    to_even,
)

logger = logging.getLogger("griptape_nodes")

__all__ = ["TopazVideoConvertHdr"]

# Both versions are registered in the proxy at the same price and the same request
# shape. 2.5 is the newer model and the default; 2 is kept selectable because 2.5 is
# absent from Topaz's own credit calculator and may not yet be live on the REST API.
MODEL_MAPPING = {
    "Hyperion 2.5": "topaz-video-hyp-2.5",
    "Hyperion 2": "topaz-video-hyp-2",
}

DEFAULT_MODEL = "Hyperion 2.5"

# Topaz documents no Hyperion-specific frame cap, unlike Astra's explicit 9000/450.
# This is the platform-wide job cap, which is also what the proxy clamps billing to.
# Our conservative choice rather than a published Hyperion limit.
MAX_HYPERION_FRAMES = 9000


class OutputFormat(StrEnum):
    """How the HDR result is encoded.

    Values are the display spelling so they interpolate straight into badge prose and
    read correctly in the dropdown.
    """

    H265_MAIN10 = "H.265 Main10 (mp4)"
    PRORES_422_HQ = "ProRes 422 HQ (mov)"


@dataclass(frozen=True)
class OutputEncoding:
    """The three ``output`` fields one format choice fans out to, plus its size ceiling.

    Kept as one table rather than three conditionals: the encoder, the profile and the
    container have to agree -- a ProRes stream in an mp4 is not a thing -- and splitting
    them invites exactly that mismatch.
    """

    video_encoder: str
    video_profile: str
    container: str
    # Topaz's per-encoder maximum output dimension. Hyperion does not scale, so this
    # only bites on an unusually large source, but H.265 tops out well below ProRes.
    # https://developer.topazlabs.com/reference/api-endpoints/video/create-request
    max_dimension: int
    file_extension: str


OUTPUT_ENCODINGS: dict[OutputFormat, OutputEncoding] = {
    # "Main10" is the 10-bit H.265 profile. 8-bit "Main" is not offered: banding in
    # smooth gradients is the classic 8-bit HDR failure, and it would waste the
    # conversion.
    OutputFormat.H265_MAIN10: OutputEncoding(
        video_encoder="H265",
        video_profile="Main10",
        container="mp4",
        max_dimension=8192,
        file_extension="mp4",
    ),
    # ProRes is the mastering path Topaz's own Hyperion page advertises. Note that
    # `videoEncoder`'s published enum is [AV1, H264, H265, VP9] and omits ProRes, while
    # two sibling fields in the *same* schema document it -- `videoProfile` lists the
    # four 422 profiles and the resolution table gives ProRes a 16386 ceiling. The enum
    # is assumed stale, the same way `source.external.provider`'s was. Unverified
    # against the live API; if Topaz rejects it, drop this entry.
    OutputFormat.PRORES_422_HQ: OutputEncoding(
        video_encoder="ProRes",
        video_profile="422 HQ",
        container="mov",
        max_dimension=16386,
        file_extension="mov",
    ),
}

DEFAULT_OUTPUT_FORMAT = OutputFormat.H265_MAIN10


def _default_filename(output_format: OutputFormat) -> str:
    """Default output filename for a format, so the extension tracks the container."""
    return f"topaz_video_convert_hdr.{OUTPUT_ENCODINGS[output_format].file_extension}"


# ffprobe's spellings for transfer characteristics and primaries that mean the source
# is already HDR. Hyperion would still run -- and still bill -- but it would be
# tone-mapping something that does not need it.
HDR_TRANSFERS = frozenset({"smpte2084", "arib-std-b67", "smpte428", "bt2020-10", "bt2020-12"})
HDR_PRIMARIES = frozenset({"bt2020"})

# ffprobe reports progressive footage as "progressive"; anything else here is some
# flavour of interlacing. Topaz notes that SDR-to-HDR "does not operate well with
# interlaced footage".
PROGRESSIVE_FIELD_ORDER = "progressive"

COST_BADGE_MESSAGE = (
    "Hyperion is metered **per frame**, not per second.\n\n"
    "Two rate tiers, picked by output pixel area:\n"
    "- **1080p** (up to 1920x1080) — the cheaper tier\n"
    "- **4K** (anything larger) — roughly **2.2x** the 1080p rate\n\n"
    "Hyperion costs about **2x Starlight** per frame on both tiers, which makes it one "
    "of the more expensive video models here. Because it converts rather than scales, "
    "the output resolution matches the source — so the tier is decided by the clip you "
    "feed it, not by a setting on this node.\n\n"
    "A 10-second 30fps clip is 300 frames whichever tier it lands in.\n\n"
    "[Topaz model pricing](https://developer.topazlabs.com/getting-started/model-pricing)"
)

HDR_PREVIEW_CAVEAT = (
    "The saved file is HDR (BT.2020 primaries, PQ transfer). Preview surfaces that "
    "assume SDR may render it darker or flatter than it really is — check the file "
    "itself rather than the thumbnail."
)


class TopazVideoConvertHdr(GriptapeProxyNode):
    """Convert an SDR video to HDR with Topaz Hyperion via the Griptape Cloud model proxy.

    Hyperion is a tone-mapper, not an upscaler: it takes no creative controls, does not
    change the resolution or the frame rate, and tags the output BT.2020/PQ itself.
    Hyperion 1 exposed an explicit ``transferFunction`` (pq/hlg); 2 and 2.5 dropped it,
    so there is no transfer selector to offer.

    Inputs:
        - video (VideoUrlArtifact): source SDR video to convert

    Outputs:
        - video_output (VideoUrlArtifact): the HDR video, saved to project storage
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
            "Convert an SDR video to HDR using Topaz Hyperion via the Griptape model proxy. "
            "Billed per frame -- see the cost note on the model parameter."
        )

        # Advisories raised while probing the source (already-HDR input, interlacing).
        # Collected rather than raised: both are real cases a user may mean to run.
        self._advisories: list[str] = []

        # INPUTS / PROPERTIES

        model_param = ParameterString(
            name="model",
            default_value=DEFAULT_MODEL,
            tooltip="Topaz Hyperion version to convert with. Both versions are priced identically.",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            traits={Options(choices=list(MODEL_MAPPING.keys()))},
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
                tooltip="Source SDR video to convert to HDR",
                allowed_modes={ParameterMode.INPUT},
                hide_property=True,
                ui_options={"display_name": "input video"},
            ),
            disclaimer_message="Topaz Labs fetches the video from this URL to perform the conversion.",
        )
        self._public_video_url_parameter.add_input_parameters()

        output_format_param = ParameterString(
            name="output_format",
            default_value=DEFAULT_OUTPUT_FORMAT,
            tooltip=(
                "How to encode the HDR result. H.265 Main10 plays in more places; "
                "ProRes 422 HQ is the mastering format."
            ),
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            traits={Options(choices=[f.value for f in OutputFormat])},
        )
        self.add_parameter(output_format_param)

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
                tooltip="The HDR video",
                allowed_modes={ParameterMode.OUTPUT, ParameterMode.PROPERTY},
                settable=False,
                ui_options={"pulse_on_run": True},
            )
        )

        self._output_file = ProjectFileParameter(
            node=self,
            name="output_file",
            default_filename=_default_filename(DEFAULT_OUTPUT_FORMAT),
        )
        self._output_file.add_parameter()

        self._create_status_parameters(
            result_details_tooltip="Details about the conversion result or any errors",
            result_details_placeholder="Conversion status and details will appear here.",
            parameter_group_initially_collapsed=True,
        )

        self.set_initial_node_size(height=360)
        # After every add_parameter: setting a badge on a name that does not exist yet
        # silently no-ops, which would leave the cost note missing with no error to
        # point at.
        self._update_cost_badge()
        self._update_format_badge()

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

    def _encoding(self) -> OutputEncoding:
        """The encoder/profile/container triple for the selected format.

        Raises rather than defaulting: an unmapped value means OutputFormat and
        OUTPUT_ENCODINGS have drifted, and silently encoding to the wrong container is
        worse than a loud failure. The Options trait makes an off-list value
        unreachable from the UI anyway.
        """
        selected = self.get_parameter_value("output_format") or DEFAULT_OUTPUT_FORMAT
        try:
            output_format = OutputFormat(selected)
        except ValueError:
            msg = f"{self.name}: unknown output format {selected!r}."
            raise ValueError(msg) from None

        encoding = OUTPUT_ENCODINGS.get(output_format)
        if encoding is None:
            msg = f"{self.name}: no encoding is registered for {output_format!r}."
            raise ValueError(msg)
        return encoding

    # -- UI reactions ------------------------------------------------------

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        super().after_value_set(parameter, value)

        # Unlike the match statements elsewhere in this file, the wildcard here must be
        # a no-op rather than a raise: `parameter.name` is an open set -- `video`,
        # `output_file` and every status parameter also land here.
        match parameter.name:
            case "output_format":
                self._update_format_badge()
                if isinstance(value, str):
                    self._sync_output_filename(value)
            case _:
                return

    def _sync_output_filename(self, output_format: str) -> None:
        """Rewrite the output filename's extension to match the chosen container.

        Only rewrites while the filename is still one of the format-derived defaults, so a
        filename the user typed is never clobbered.
        """
        current_value = self.get_parameter_value("output_file")
        default_filenames = {_default_filename(fmt) for fmt in OutputFormat}
        if current_value not in default_filenames:
            return

        try:
            updated_value = _default_filename(OutputFormat(output_format))
        except ValueError:
            return

        if current_value == updated_value:
            return

        self.set_parameter_value("output_file", updated_value)
        self.publish_update_to_parameter("output_file", updated_value)

    def _update_cost_badge(self) -> None:
        param = self.get_parameter_by_name("model")
        if param is None:
            return

        param.set_badge(variant="warning", title="Billed per frame", message=COST_BADGE_MESSAGE)

    def _update_format_badge(self) -> None:
        param = self.get_parameter_by_name("output_format")
        if param is None:
            return

        match OutputFormat(self.get_parameter_value("output_format") or DEFAULT_OUTPUT_FORMAT):
            case OutputFormat.H265_MAIN10:
                param.set_badge(
                    variant="info",
                    title="10-bit H.265 in an mp4",
                    message=(f"Widely playable, and small enough to move around. {HDR_PREVIEW_CAVEAT}"),
                )
            case OutputFormat.PRORES_422_HQ:
                param.set_badge(
                    variant="warning",
                    title="Large mastering files",
                    message=(
                        "ProRes 422 HQ is the format Topaz recommends for grading, and is "
                        "many times larger than H.265. A `.mov` will not play in a browser "
                        f"preview at all. {HDR_PREVIEW_CAVEAT}"
                    ),
                )
            case output_format:
                msg = f"Unknown output format: {output_format!r}"
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

    def _source_advisories(self, metadata: VideoMetadata) -> list[str]:
        """Flag sources Hyperion will process but probably should not be given.

        Both cases warn rather than block. A file mislabelled as BT.2020 is a real
        thing, and so is deliberately re-running a conversion; refusing either would
        be worse than saying so.
        """
        advisories: list[str] = []
        color = metadata.color_details

        transfer = (color.optional_color_transfer or "").lower()
        primaries = (color.optional_color_primaries or "").lower()
        if transfer in HDR_TRANSFERS or primaries in HDR_PRIMARIES:
            advisories.append(
                f"The source already looks like HDR (transfer {transfer or 'unset'!r}, "
                f"primaries {primaries or 'unset'!r}). Hyperion converts SDR footage; "
                "running it on HDR input still bills for every frame."
            )

        field_order = (color.optional_field_order or "").lower()
        if field_order and field_order != PROGRESSIVE_FIELD_ORDER:
            advisories.append(
                f"The source reports interlaced field order {field_order!r}. Topaz notes "
                "that SDR-to-HDR conversion does not operate well on interlaced footage; "
                "deinterlace it first for a usable result."
            )

        return advisories

    def _output_resolution(self, source_width: int, source_height: int) -> tuple[int, int]:
        """Hyperion converts rather than scales, so the output matches the source.

        Topaz requires ``output.resolution`` regardless, and the proxy reads it to pick
        the billing tier, so it is echoed back rather than omitted.
        """
        width, height = to_even(source_width), to_even(source_height)

        max_dimension = self._encoding().max_dimension
        if width > max_dimension or height > max_dimension:
            encoding = self._encoding()
            msg = (
                f"{self.name}: the source is {source_width}x{source_height}, over the "
                f"{max_dimension}-pixel limit for {encoding.video_encoder}. Choose a "
                "different output format or downscale the source first."
            )
            raise ValueError(msg)

        return width, height

    # -- request -----------------------------------------------------------

    async def _build_payload(self) -> dict[str, Any]:
        video = self.get_parameter_value("video")
        if not video:
            msg = f"{self.name} requires an input video to convert."
            raise ValueError(msg)

        container = self._derive_container(video)

        # ffprobe is synchronous and can take a moment on a long clip; keep it off
        # the event loop.
        metadata = await asyncio.to_thread(self._probe_source, video)

        source_width = metadata.dimensions.width
        source_height = metadata.dimensions.height

        source_frames = self._frame_count(metadata)
        if source_frames <= 0:
            msg = (
                f"{self.name} could not determine the frame count of the input video. "
                "Topaz requires it, and neither nb_frames nor duration x frame rate was "
                "available from the file."
            )
            raise ValueError(msg)

        if source_frames > MAX_HYPERION_FRAMES:
            msg = (
                f"{self.name}: the input video has {source_frames} frames, over the "
                f"{MAX_HYPERION_FRAMES:,}-frame job limit. Trim or split the video first."
            )
            raise ValueError(msg)

        output_width, output_height = self._output_resolution(source_width, source_height)

        self._advisories = self._source_advisories(metadata)
        for advisory in self._advisories:
            logger.warning("%s: %s", self.name, advisory)

        # Upload after probing: the probe needs the original local path, and there
        # is no point paying for an upload if the file turns out to be unusable.
        video_url = self._public_video_url_parameter.get_public_url_for_parameter()
        if not video_url:
            msg = f"{self.name} could not produce a public URL for the input video."
            raise ValueError(msg)

        source: dict[str, Any] = {
            "container": container,
            "frameCount": source_frames,
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

        encoding = self._encoding()

        logger.info(
            "%s converting %dx%d (%d frames) to HDR as %s/%s",
            self.name,
            source_width,
            source_height,
            source_frames,
            encoding.video_encoder,
            encoding.video_profile,
        )

        # `output` carries the encoder triple but not `frameRate`, `audioCodec` or
        # `audioTransfer`, which the published schema marks required. The live
        # validator does not enforce them -- TopazVideoUpscale ships sending only
        # `resolution` -- and guessing an audio disposition for a source whose audio
        # streams we have not probed is the worse risk.
        return {
            "source": source,
            "output": {
                "resolution": {"width": output_width, "height": output_height},
                "videoEncoder": encoding.video_encoder,
                "videoProfile": encoding.video_profile,
                "container": encoding.container,
            },
        }
        # No `filters`: Hyperion accepts no creative controls, and the proxy stamps the
        # routed model code onto the filters it synthesizes itself. Sending our own
        # would only risk the "filters[].model must match the requested model"
        # rejection.

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
            action="converted to HDR",
        )

    async def _handle_binary_video_response(self, video_bytes: bytes) -> None:
        """Save video bytes served directly by the proxy rather than via a URL."""
        if not video_bytes:
            self._set_safe_defaults()
            self._set_status_results(
                was_successful=False,
                result_details=f"{self.name}: the conversion completed but no video data was received.",
            )
            return

        try:
            dest = self._output_file.build_file()
            saved = await dest.awrite_bytes(video_bytes)
        except (OSError, PermissionError) as e:
            logger.error("%s failed to save the converted video: %s", self.name, e)
            self._set_safe_defaults()
            self._set_status_results(
                was_successful=False,
                result_details=f"{self.name}: the conversion succeeded but saving the video failed: {e}",
            )
            return

        self.parameter_output_values["video_output"] = VideoUrlArtifact(value=saved.location, name=saved.name)
        self._set_status_results(
            was_successful=True,
            result_details=f"Conversion successful. Video saved as {saved.name}. {HDR_PREVIEW_CAVEAT}",
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
