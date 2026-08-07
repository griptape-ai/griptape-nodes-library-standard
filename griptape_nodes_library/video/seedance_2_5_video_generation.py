from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, ClassVar

from griptape.artifacts import AudioArtifact, ImageArtifact, ImageUrlArtifact
from griptape.artifacts.audio_url_artifact import AudioUrlArtifact
from griptape.artifacts.video_url_artifact import VideoUrlArtifact
from griptape_nodes.exe_types.core_types import ParameterGroup, ParameterList, ParameterMode
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_types.parameter_bool import ParameterBool
from griptape_nodes.exe_types.param_types.parameter_dict import ParameterDict
from griptape_nodes.exe_types.param_types.parameter_image import ParameterImage
from griptape_nodes.exe_types.param_types.parameter_int import ParameterInt
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.exe_types.param_types.parameter_video import ParameterVideo
from griptape_nodes.traits.options import Options
from griptape_nodes.utils.artifact_normalization import normalize_artifact_input, normalize_artifact_list

from griptape_nodes_library.assets import (
    ASSET_KIND_AUDIO,
    ASSET_KIND_IMAGE,
    ASSET_KIND_VIDEO,
    ASSET_REFERENCE_TYPE_NAMES,
    get_provider_asset_kind,
    is_provider_asset_reference,
)
from griptape_nodes_library.video.seedance_common import (
    SeedanceProxyNode,
    coerce_video_url,
    extract_video_url,
)

logger = logging.getLogger("griptape_nodes")

__all__ = ["Seedance25VideoGeneration"]

SEEDANCE_2_5_MODEL_ID = "dreamina-seedance-2-5-260628"

# Reference asset limits, per the BytePlus Seedance 2.5 docs. Each kind has its own cap and the
# combined count across all three kinds is capped separately.
MAX_REFERENCE_IMAGES = 30
MAX_REFERENCE_VIDEOS = 10
MAX_REFERENCE_AUDIO = 10
MAX_TOTAL_REFERENCE_ASSETS = 50

MIN_DURATION = 4
MAX_DURATION = 30
SMART_DURATION = -1

RESOLUTION_CHOICES = ["480p", "720p"]
OUTPUT_FORMAT_CHOICES = ["mp4", "mov"]
DEFAULT_OUTPUT_FORMAT = "mp4"
OUTPUT_FILENAME_BASE = "seedance_2_5_video"
LAST_FRAME_FILENAME = "seedance_2_5_last_frame.png"

ALL_RATIO_CHOICES = ("adaptive", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16")
ADAPTIVE_ONLY_RATIO_CHOICES = ("adaptive",)
ALL_DURATION_CHOICES = (SMART_DURATION, *range(MIN_DURATION, MAX_DURATION + 1))
SMART_ONLY_DURATION_CHOICES = (SMART_DURATION,)

REFERENCE_IMAGES_PARAMETER = "reference_images"
REFERENCE_VIDEOS_PARAMETER = "reference_videos"
REFERENCE_AUDIO_PARAMETER = "reference_audio"
REFERENCE_PARAMETERS = (REFERENCE_IMAGES_PARAMETER, REFERENCE_VIDEOS_PARAMETER, REFERENCE_AUDIO_PARAMETER)
FRAME_PARAMETERS = ("first_frame", "last_frame")

# Seedance 2.5 registers private assets through the Griptape Cloud proxy, the same as the 2.0 models.
SUPPORTS_PRIVATE_ASSETS = True


class SeedanceTask(StrEnum):
    """The five task types Seedance 2.5 classifies a request into.

    The provider infers the task from ``content.role`` plus the prompt's intent — there is no
    ``task_type`` field — and applies per-task constraints to ``ratio`` and ``duration``. A
    violation is reported asynchronously, after the task has queued, so this node makes the task
    an explicit choice and validates its constraints before submitting.
    """

    TEXT_TO_VIDEO = "Text to Video"
    FIRST_LAST_FRAME = "First/Last Frame"
    REFERENCE_TO_VIDEO = "Reference to Video"
    VIDEO_EDITING = "Video Editing"
    VIDEO_EXTENSION = "Video Extension"


@dataclass(frozen=True)
class TaskConstraints:
    """The provider constraints that apply to one Seedance 2.5 task type.

    Args:
        allows_frames: Whether first_frame/last_frame inputs belong to this task.
        allows_references: Whether the reference image/video/audio lists belong to this task.
        requires_reference_video: Whether at least one reference video is mandatory.
        ratio_choices: The ``ratio`` values the provider accepts for this task.
        duration_choices: The ``duration`` values the provider accepts for this task.
        trigger_keywords: Words the prompt must contain for the provider to classify the request
            as this task. Empty when the task needs no prompt trigger.
    """

    allows_frames: bool
    allows_references: bool
    requires_reference_video: bool
    ratio_choices: tuple[str, ...]
    duration_choices: tuple[int, ...]
    trigger_keywords: tuple[str, ...]


# Single source of truth for the per-task constraints, straight from the Seedance 2.5
# task-specific constraints table. Adding a task type is one row here.
TASK_CONSTRAINTS: dict[SeedanceTask, TaskConstraints] = {
    SeedanceTask.TEXT_TO_VIDEO: TaskConstraints(
        allows_frames=False,
        allows_references=False,
        requires_reference_video=False,
        ratio_choices=ALL_RATIO_CHOICES,
        duration_choices=ALL_DURATION_CHOICES,
        trigger_keywords=(),
    ),
    SeedanceTask.FIRST_LAST_FRAME: TaskConstraints(
        allows_frames=True,
        allows_references=False,
        requires_reference_video=False,
        # The output keeps the first frame's aspect ratio, so no ratio can be specified.
        ratio_choices=ADAPTIVE_ONLY_RATIO_CHOICES,
        duration_choices=ALL_DURATION_CHOICES,
        trigger_keywords=(),
    ),
    SeedanceTask.REFERENCE_TO_VIDEO: TaskConstraints(
        allows_frames=False,
        allows_references=True,
        requires_reference_video=False,
        ratio_choices=ALL_RATIO_CHOICES,
        duration_choices=ALL_DURATION_CHOICES,
        trigger_keywords=(),
    ),
    SeedanceTask.VIDEO_EDITING: TaskConstraints(
        allows_frames=False,
        allows_references=True,
        requires_reference_video=True,
        # The output keeps the edited video's aspect ratio and duration.
        ratio_choices=ADAPTIVE_ONLY_RATIO_CHOICES,
        duration_choices=SMART_ONLY_DURATION_CHOICES,
        trigger_keywords=("edit", "add", "delete", "remove", "modify", "replace", "change"),
    ),
    SeedanceTask.VIDEO_EXTENSION: TaskConstraints(
        allows_frames=False,
        allows_references=True,
        requires_reference_video=True,
        # The output keeps the extended video's aspect ratio.
        ratio_choices=ADAPTIVE_ONLY_RATIO_CHOICES,
        duration_choices=ALL_DURATION_CHOICES,
        trigger_keywords=("extend", "continue"),
    ),
}

PROMPT_TIPS_MESSAGE = (
    "Reference assets are addressed positionally in the prompt: @Image 1, @Video 1, @Audio 1 "
    "follow the order of the reference lists below.\n\n"
    "Audio cues use special characters: () for music, <> for sound effects, {} for dialogue, "
    "and 【】 for subtitles.\n\n"
    "Video Editing and Video Extension are classified from the prompt, so the prompt must name "
    "what you want done — e.g. 'Remove everyone in @Video 1 except the protagonist' or "
    "'Extend @Video 1 backward'."
)

REFERENCE_VIDEO_UPLOAD_MESSAGE = (
    "This node requires a public URL for each reference video.\n\n"
    "Seedance does not accept video base64, so each reference video is uploaded to Griptape Cloud "
    "static storage to obtain a temporary public URL, which the Seedance service then fetches. "
    "The upload is deleted when the run finishes."
)


def _get_task_constraints(task: str) -> TaskConstraints:
    """Return the constraint record for a task value, defaulting to the least restrictive task.

    A task value can arrive over a connection, so an unrecognized value must not raise here —
    ``_validate_parameters`` reports it with an actionable message instead.
    """
    for member in SeedanceTask:
        if task == member:
            return TASK_CONSTRAINTS[member]
    return TASK_CONSTRAINTS[SeedanceTask.TEXT_TO_VIDEO]


class Seedance25VideoGeneration(SeedanceProxyNode):
    """Generate a video using Dreamina Seedance 2.5 via the Griptape Cloud model proxy.

    Seedance 2.5 classifies each request into one of five task types from the media roles and the
    prompt's intent, and locks `ratio`/`duration` per task. The `task` parameter makes that choice
    explicit so the node can show the right inputs, narrow the settings to the values the provider
    accepts, and reject a mismatched request before it is queued (the provider only reports task
    constraint violations asynchronously, after the task starts processing).

    Tasks:
    - Text to Video: prompt only
    - First/Last Frame: first and/or last frame images (ratio locked to adaptive)
    - Reference to Video: up to 30 images + 10 videos + 10 audio clips as references
    - Video Editing: edit a reference video (ratio locked to adaptive, duration locked to -1)
    - Video Extension: extend a reference video (ratio locked to adaptive)

    Inputs:
        - prompt (str): Text prompt for the video
        - task (str): One of the five Seedance 2.5 task types (default: Text to Video)
        - resolution (str): Output resolution (default: 720p, options: 480p, 720p)
        - ratio (str): Output aspect ratio (default: adaptive)
        - duration (int): Video duration in seconds (4-30, or -1 to let the model choose)
        - generate_audio (bool): Generate audio with video (default: False)
        - output_format (str): Output container, mp4 or mov (default: mp4)
        - watermark (bool): Add an "AI Generated" watermark (default: False)
        - return_last_frame (bool): Also return the video's last frame as a PNG (default: False)
        - first_frame/last_frame: Optional frame images (First/Last Frame task only)
        - reference_images/reference_videos/reference_audio: Optional reference media

    Outputs:
        - generation_id (str): Griptape Cloud generation id
        - provider_response (dict): Verbatim response from API
        - video_url (VideoUrlArtifact): Saved static video URL
        - last_frame_url (ImageUrlArtifact): Saved last frame, when return_last_frame is set
        - was_successful (bool): Whether generation succeeded
        - result_details (str): Details about the result or error
    """

    REFERENCE_ASSET_KINDS: ClassVar[dict[str, str]] = {
        REFERENCE_IMAGES_PARAMETER: ASSET_KIND_IMAGE,
        REFERENCE_VIDEOS_PARAMETER: ASSET_KIND_VIDEO,
        REFERENCE_AUDIO_PARAMETER: ASSET_KIND_AUDIO,
    }

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.category = "API Nodes"
        self.description = "Generate video via Seedance 2.5 through Griptape Cloud model proxy"

        self.add_parameter(
            ParameterString(
                name="task",
                default_value=SeedanceTask.TEXT_TO_VIDEO,
                tooltip=(
                    "Which Seedance 2.5 task to run. The provider infers this from the media roles and the "
                    "prompt, and locks aspect ratio and duration per task, so choosing it here keeps the "
                    "settings and inputs consistent with what the provider accepts."
                ),
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"display_name": "Task"},
                traits={Options(choices=[task.value for task in SeedanceTask])},
            )
        )

        prompt_parameter = ParameterString(
            name="prompt",
            tooltip=(
                "Text prompt for the video. Reference assets are addressed as @Image 1, @Video 1, @Audio 1 "
                "in the order of the reference lists."
            ),
            multiline=True,
            placeholder_text="Describe the video...",
            allow_output=False,
            ui_options={"display_name": "Prompt"},
        )
        self.add_parameter(prompt_parameter)
        prompt_parameter.set_badge(variant="tip", title="Prompt Tips", message=PROMPT_TIPS_MESSAGE)

        # First/Last Frame inputs
        self.add_parameter(
            ParameterImage(
                name="first_frame",
                default_value=None,
                tooltip="Optional first frame image",
                allowed_modes={ParameterMode.INPUT},
                hide_property=True,
                ui_options={"display_name": "First Frame"},
            )
        )

        self.add_parameter(
            ParameterImage(
                name="last_frame",
                default_value=None,
                tooltip="Optional last frame image",
                allowed_modes={ParameterMode.INPUT},
                hide_property=True,
                ui_options={"display_name": "Last Frame"},
            )
        )

        # Reference inputs
        self.add_parameter(
            ParameterList(
                name=REFERENCE_IMAGES_PARAMETER,
                input_types=["ImageUrlArtifact", "ImageArtifact", "str", ASSET_REFERENCE_TYPE_NAMES[ASSET_KIND_IMAGE]],
                default_value=[],
                tooltip=(
                    f"Optional reference images (up to {MAX_REFERENCE_IMAGES}). Connect a Seedance Human "
                    "Reference Asset to register an image as a private asset."
                ),
                allowed_modes={ParameterMode.INPUT},
                ui_options={"display_name": "Reference Images", "expander": True, "hide_property": True},
                child_prefix="Reference Image",
                max_items=MAX_REFERENCE_IMAGES,
            )
        )

        reference_videos = ParameterList(
            name=REFERENCE_VIDEOS_PARAMETER,
            input_types=["VideoUrlArtifact", ASSET_REFERENCE_TYPE_NAMES[ASSET_KIND_VIDEO]],
            type="VideoUrlArtifact",
            default_value=[],
            tooltip=(
                f"Optional reference videos (up to {MAX_REFERENCE_VIDEOS}, 2-30s each and no more than 30s "
                "combined). Seedance only accepts public URLs or uploaded asset URLs for videos. Connect a "
                "Seedance Human Reference Asset to register a video as a private asset."
            ),
            allowed_modes={ParameterMode.INPUT},
            ui_options={"display_name": "Reference Videos", "expander": True, "hide_property": True},
            child_prefix="Reference Video",
            max_items=MAX_REFERENCE_VIDEOS,
        )
        self.add_parameter(reference_videos)
        reference_videos.set_badge(
            variant="cloud-upload",
            title="Media Upload",
            message=REFERENCE_VIDEO_UPLOAD_MESSAGE,
            hide_clear_button=False,
        )

        self.add_parameter(
            ParameterList(
                name=REFERENCE_AUDIO_PARAMETER,
                input_types=["AudioArtifact", "AudioUrlArtifact", "str", ASSET_REFERENCE_TYPE_NAMES[ASSET_KIND_AUDIO]],
                default_value=[],
                tooltip=(
                    f"Optional reference audio (up to {MAX_REFERENCE_AUDIO} clips, 2-30s each and no more than "
                    "30s combined). URLs, asset:// IDs, or base64/data URIs are supported. Seedance 2.5 accepts "
                    "audio without any image or video."
                ),
                allowed_modes={ParameterMode.INPUT},
                ui_options={"display_name": "Reference Audio", "expander": True, "hide_property": True},
                child_prefix="Reference Audio",
                max_items=MAX_REFERENCE_AUDIO,
            )
        )

        # Generation settings
        with ParameterGroup(name="Generation Settings") as settings_group:
            ParameterString(
                name="resolution",
                default_value="720p",
                tooltip="Output resolution (480p or 720p)",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=list(RESOLUTION_CHOICES))},
            )

            ParameterString(
                name="ratio",
                default_value="adaptive",
                tooltip=(
                    "Output aspect ratio. Locked to adaptive for First/Last Frame, Video Editing, and Video "
                    "Extension, where the output keeps the input's aspect ratio."
                ),
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=list(ALL_RATIO_CHOICES))},
            )

            ParameterInt(
                name="duration",
                default_value=SMART_DURATION,
                tooltip=(
                    f"Video duration in seconds ({MIN_DURATION}-{MAX_DURATION}, or {SMART_DURATION} to let the "
                    f"model choose). Locked to {SMART_DURATION} for Video Editing, where the output matches the "
                    "edited video's duration."
                ),
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=list(ALL_DURATION_CHOICES))},
            )

            ParameterBool(
                name="generate_audio",
                default_value=False,
                tooltip="Generate audio with video",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )

            ParameterString(
                name="output_format",
                default_value=DEFAULT_OUTPUT_FORMAT,
                tooltip=(
                    "Output container. mp4 is broadly compatible; mov preserves more color precision for "
                    "post-production and is recommended for video editing and extension."
                ),
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=list(OUTPUT_FORMAT_CHOICES))},
            )

            ParameterBool(
                name="watermark",
                default_value=False,
                tooltip='Add an "AI Generated" watermark to the lower-right corner of the video',
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )

            ParameterBool(
                name="return_last_frame",
                default_value=False,
                tooltip=(
                    "Also return the video's last frame as a PNG. Useful for chaining runs: feed it into the "
                    "next generation's first frame to continue a sequence."
                ),
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )

        self.add_node_element(settings_group)

        # Outputs
        self.add_parameter(
            ParameterDict(
                name="provider_response",
                tooltip="Verbatim response from API",
                allowed_modes={ParameterMode.OUTPUT},
                hide_property=True,
                hide=True,
            )
        )

        self.add_parameter(
            ParameterVideo(
                name="video_url",
                tooltip="Saved video as URL artifact",
                allowed_modes={ParameterMode.OUTPUT, ParameterMode.PROPERTY},
                settable=False,
                ui_options={"pulse_on_run": True},
            )
        )

        self.add_parameter(
            ParameterImage(
                name="last_frame_url",
                tooltip="Saved last frame of the generated video, when Return Last Frame is enabled",
                allowed_modes={ParameterMode.OUTPUT},
                settable=False,
                hide=True,
                ui_options={"display_name": "Last Frame Image"},
            )
        )

        self._output_file = ProjectFileParameter(
            node=self,
            name="output_file",
            default_filename=self._default_output_filename(DEFAULT_OUTPUT_FORMAT),
        )
        self._output_file.add_parameter()

        self._last_frame_file = ProjectFileParameter(
            node=self,
            name="last_frame_file",
            default_filename=LAST_FRAME_FILENAME,
            ui_options={"hide": True},
        )
        self._last_frame_file.add_parameter()

        self._create_status_parameters(
            result_details_tooltip="Details about the video generation result or any errors",
            result_details_placeholder="Generation status and details will appear here.",
            parameter_group_initially_collapsed=True,
        )

        # Set initial visibility
        self._update_parameter_visibility()

    @staticmethod
    def _default_output_filename(output_format: str) -> str:
        return f"{OUTPUT_FILENAME_BASE}.{output_format}"

    def set_parameter_value(self, param_name: str, value: Any, **kwargs: Any) -> None:
        super().set_parameter_value(param_name, value, **kwargs)
        if not kwargs.get("initial_setup", False):
            return
        # after_value_set is skipped during initial_setup (workflow load), but the UI still needs
        # to reflect the loaded task and output format.
        self._react_to_parameter_change(param_name, self.get_parameter_value(param_name), sync_only=True)

    def after_value_set(self, parameter: Any, value: Any) -> None:
        if parameter.name in FRAME_PARAMETERS:
            artifact = normalize_artifact_input(value, ImageUrlArtifact, accepted_types=(ImageArtifact,))
            if artifact != value:
                self.set_parameter_value(parameter.name, artifact)

        if parameter.name == REFERENCE_IMAGES_PARAMETER and isinstance(value, list):
            updated_list = normalize_artifact_list(value, ImageUrlArtifact, accepted_types=(ImageArtifact,))
            if updated_list != value:
                self.set_parameter_value(REFERENCE_IMAGES_PARAMETER, updated_list)

        if parameter.name == REFERENCE_AUDIO_PARAMETER and isinstance(value, list):
            updated_list = normalize_artifact_list(value, AudioUrlArtifact, accepted_types=(AudioArtifact,))
            if updated_list != value:
                self.set_parameter_value(REFERENCE_AUDIO_PARAMETER, updated_list)

        self._react_to_parameter_change(parameter.name, value, sync_only=False)
        return super().after_value_set(parameter, value)

    def _react_to_parameter_change(self, param_name: str, value: Any, *, sync_only: bool) -> None:
        if param_name in {"task", "return_last_frame"}:
            self._update_parameter_visibility()

        if param_name == "output_format" and isinstance(value, str):
            if not sync_only:
                self._sync_output_filename(value)

    def _sync_output_filename(self, output_format: str) -> None:
        """Rewrite the output filename's extension to match the chosen container.

        Only rewrites while the filename is still one of the format-derived defaults, so a
        filename the user typed is never clobbered.
        """
        current_value = self.get_parameter_value("output_file")
        default_filenames = {self._default_output_filename(fmt) for fmt in OUTPUT_FORMAT_CHOICES}
        if current_value not in default_filenames:
            return

        updated_value = self._default_output_filename(output_format)
        if current_value == updated_value:
            return

        self.set_parameter_value("output_file", updated_value)
        self.publish_update_to_parameter("output_file", updated_value)

    def _update_parameter_visibility(self) -> None:
        """Show the inputs that belong to the selected task and lock the settings it constrains."""
        task = self.get_parameter_value("task") or SeedanceTask.TEXT_TO_VIDEO
        constraints = _get_task_constraints(task)

        if constraints.allows_frames:
            self.show_parameter_by_name(list(FRAME_PARAMETERS))
        else:
            self.hide_parameter_by_name(list(FRAME_PARAMETERS))

        if constraints.allows_references:
            self.show_parameter_by_name(list(REFERENCE_PARAMETERS))
        else:
            self.hide_parameter_by_name(list(REFERENCE_PARAMETERS))

        self._update_option_choices_for_task("ratio", list(constraints.ratio_choices), fallback="adaptive")
        self._update_option_choices_for_task("duration", list(constraints.duration_choices), fallback=SMART_DURATION)

        if self.get_parameter_value("return_last_frame"):
            self.show_parameter_by_name(["last_frame_url", "last_frame_file"])
        else:
            self.hide_parameter_by_name(["last_frame_url", "last_frame_file"])

    def _update_option_choices_for_task(self, parameter_name: str, choices: list[Any], *, fallback: Any) -> None:
        """Narrow a parameter's Options to the task's accepted values, coercing an invalid current value."""
        parameter = self.get_parameter_by_name(parameter_name)
        if parameter is None:
            return

        existing_traits = parameter.find_elements_by_type(Options)
        if existing_traits:
            parameter.remove_trait(trait_type=existing_traits[0])
        parameter.add_trait(Options(choices=choices))

        if self.get_parameter_value(parameter_name) not in choices:
            self.set_parameter_value(parameter_name, fallback)

    def _get_api_model_id(self) -> str:
        return SEEDANCE_2_5_MODEL_ID

    async def _process_generation(self) -> None:
        self._pending_asset_uploads = []
        try:
            await super()._process_generation()
        finally:
            self._cleanup_pending_asset_uploads()

    def validate_before_node_run(self) -> list[Exception] | None:
        """Validate parameters before execution."""
        exceptions = super().validate_before_node_run() or []

        try:
            self._validate_parameters(self._get_parameters())
        except ValueError as e:
            exceptions.append(e)

        return exceptions if exceptions else None

    def _get_parameters(self) -> dict[str, Any]:
        first_frame = normalize_artifact_input(
            self.get_parameter_value("first_frame"),
            ImageUrlArtifact,
            accepted_types=(ImageArtifact,),
        )
        last_frame = normalize_artifact_input(
            self.get_parameter_value("last_frame"),
            ImageUrlArtifact,
            accepted_types=(ImageArtifact,),
        )

        reference_images = self.get_parameter_value(REFERENCE_IMAGES_PARAMETER) or []
        normalized_reference_images = (
            normalize_artifact_list(reference_images, ImageUrlArtifact, accepted_types=(ImageArtifact,))
            if reference_images
            else []
        )
        reference_audio = self.get_parameter_value(REFERENCE_AUDIO_PARAMETER) or []
        normalized_reference_audio = (
            normalize_artifact_list(reference_audio, AudioUrlArtifact, accepted_types=(AudioArtifact,))
            if reference_audio
            else []
        )

        return {
            "prompt": self.get_parameter_value("prompt") or "",
            "task": self.get_parameter_value("task") or SeedanceTask.TEXT_TO_VIDEO,
            "resolution": self.get_parameter_value("resolution") or "720p",
            "ratio": self.get_parameter_value("ratio") or "adaptive",
            "duration": self.get_parameter_value("duration"),
            "generate_audio": self.get_parameter_value("generate_audio"),
            "output_format": self.get_parameter_value("output_format") or DEFAULT_OUTPUT_FORMAT,
            "watermark": self.get_parameter_value("watermark"),
            "return_last_frame": self.get_parameter_value("return_last_frame"),
            "first_frame": first_frame,
            "last_frame": last_frame,
            REFERENCE_IMAGES_PARAMETER: normalized_reference_images,
            # An empty list slot yields a falsy child value; drop those so counts and @Video N
            # ordering reflect only the videos actually connected.
            REFERENCE_VIDEOS_PARAMETER: [
                value for value in self.get_parameter_value(REFERENCE_VIDEOS_PARAMETER) or [] if value
            ],
            REFERENCE_AUDIO_PARAMETER: normalized_reference_audio,
        }

    def _validate_parameters(self, params: dict[str, Any]) -> None:
        """Validate the task, its media inputs, and its ratio/duration constraints."""
        task = params["task"]
        if task not in set(SeedanceTask):
            supported = ", ".join(member.value for member in SeedanceTask)
            msg = f"{self.name}: unknown task {task!r}. Supported tasks: {supported}."
            raise ValueError(msg)

        constraints = TASK_CONSTRAINTS[SeedanceTask(task)]

        self._validate_media_matches_task(params, task, constraints)
        self._validate_reference_counts(params)
        self._validate_trigger_keywords(params, task, constraints)
        self._validate_ratio_and_duration(params, task, constraints)

        # Private-asset references require Griptape auth (not BYOK). Gate first so the user gets a
        # specific message before the per-kind check.
        self._validate_private_asset_auth(params)

        # Private-asset reference kind must match the receiving input (early feedback; the
        # authoritative check happens at build time in _append_private_asset).
        if self._private_assets_active():
            self._validate_private_asset_kinds(params)

    def _validate_media_matches_task(self, params: dict[str, Any], task: str, constraints: TaskConstraints) -> None:
        """Raise if media is connected to inputs the selected task does not use."""
        has_frames = any(params.get(name) is not None for name in FRAME_PARAMETERS)
        has_references = any(params.get(name) for name in REFERENCE_PARAMETERS)

        if has_frames and not constraints.allows_frames:
            msg = (
                f"{self.name}: first_frame/last_frame inputs are only used by the "
                f"{SeedanceTask.FIRST_LAST_FRAME.value} task. Switch the task to "
                f"{SeedanceTask.FIRST_LAST_FRAME.value}, or clear the frame inputs."
            )
            raise ValueError(msg)

        if has_references and not constraints.allows_references:
            reference_tasks = ", ".join(
                member.value for member in SeedanceTask if TASK_CONSTRAINTS[member].allows_references
            )
            msg = (
                f"{self.name}: the reference image/video/audio inputs are only used by the "
                f"{reference_tasks} tasks. Switch the task, or clear the reference inputs."
            )
            raise ValueError(msg)

        if constraints.requires_reference_video and not params.get(REFERENCE_VIDEOS_PARAMETER):
            msg = (
                f"{self.name}: the {task} task requires at least one reference video — the video to "
                f"{'edit' if task == SeedanceTask.VIDEO_EDITING else 'extend'}. Connect a video to "
                "Reference Videos, or switch the task."
            )
            raise ValueError(msg)

    def _validate_reference_counts(self, params: dict[str, Any]) -> None:
        """Raise if any reference list exceeds the provider's cap for that kind.

        The per-kind caps sum to MAX_TOTAL_REFERENCE_ASSETS, so satisfying all three also satisfies
        the documented total cap; there is no separate total check to make.
        """
        caps = {
            REFERENCE_IMAGES_PARAMETER: (MAX_REFERENCE_IMAGES, "reference images"),
            REFERENCE_VIDEOS_PARAMETER: (MAX_REFERENCE_VIDEOS, "reference videos"),
            REFERENCE_AUDIO_PARAMETER: (MAX_REFERENCE_AUDIO, "reference audio clips"),
        }
        for parameter_name, (cap, label) in caps.items():
            count = len(params.get(parameter_name) or [])
            if count > cap:
                msg = f"{self.name}: Seedance 2.5 supports up to {cap} {label}, got {count}."
                raise ValueError(msg)

    def _validate_trigger_keywords(self, params: dict[str, Any], task: str, constraints: TaskConstraints) -> None:
        """Raise if the prompt lacks a keyword the provider needs to classify this task.

        Seedance 2.5 derives Video Editing and Video Extension from the prompt's intent, so a
        prompt with no such wording is classified as a different task and rejected asynchronously.
        """
        if not constraints.trigger_keywords:
            return

        prompt = (params.get("prompt") or "").lower()
        if any(keyword in prompt for keyword in constraints.trigger_keywords):
            return

        keywords = ", ".join(constraints.trigger_keywords)
        msg = (
            f"{self.name}: the {task} task is inferred from the prompt, which must say what to do to the "
            f"reference video. Include at least one of: {keywords}."
        )
        raise ValueError(msg)

    def _validate_ratio_and_duration(self, params: dict[str, Any], task: str, constraints: TaskConstraints) -> None:
        """Raise if ratio or duration is outside what the provider accepts for this task.

        The dropdowns are already narrowed per task, but either value can arrive over a connection.
        """
        ratio = params.get("ratio")
        if ratio not in constraints.ratio_choices:
            accepted = ", ".join(constraints.ratio_choices)
            msg = (
                f"{self.name}: the {task} task only supports ratio {accepted}, got {ratio!r}. "
                "The output keeps the aspect ratio of its input."
            )
            raise ValueError(msg)

        duration = params.get("duration")
        if duration is None:
            return
        if duration not in constraints.duration_choices:
            if constraints.duration_choices == SMART_ONLY_DURATION_CHOICES:
                msg = (
                    f"{self.name}: the {task} task only supports duration {SMART_DURATION}, got {duration}. "
                    "The output duration matches the input video."
                )
            else:
                msg = (
                    f"{self.name}: Seedance 2.5 supports duration between {MIN_DURATION}-{MAX_DURATION} seconds "
                    f"or {SMART_DURATION} to let the model choose, got {duration}."
                )
            raise ValueError(msg)

    def _iter_reference_asset_checks(self, params: dict[str, Any]) -> list[tuple[Any, str]]:
        """List (reference value, expected kind) pairs across all reference inputs."""
        return [
            (item, asset_kind)
            for parameter_name, asset_kind in self.REFERENCE_ASSET_KINDS.items()
            for item in params.get(parameter_name) or []
        ]

    def _validate_private_asset_auth(self, params: dict[str, Any]) -> None:
        """Raise if a private-asset reference is connected while the node uses BYOK auth.

        Provider assets are registered through the Griptape Cloud proxy on the org's behalf, so
        they are unavailable when the user brings their own provider key.
        """
        if not self._is_byok_enabled():
            return
        for value, _ in self._iter_reference_asset_checks(params):
            if is_provider_asset_reference(value):
                msg = (
                    f"{self.name}: private-asset references (Seedance Human Reference Asset) require "
                    "Griptape authentication and are not available when using your own provider key. "
                    "Switch off the customer key option, or remove the private-asset reference inputs."
                )
                raise ValueError(msg)

    def _validate_private_asset_kinds(self, params: dict[str, Any]) -> None:
        """Raise if a private-asset reference's kind doesn't match its receiving input."""
        for value, expected_kind in self._iter_reference_asset_checks(params):
            if is_provider_asset_reference(value):
                actual_kind = get_provider_asset_kind(value)
                if actual_kind != expected_kind:
                    msg = (
                        f"{self.name}: a {actual_kind or 'unknown'} private-asset reference is connected to a "
                        f"{expected_kind} reference input. Set the Seedance Human Reference Asset's Asset Kind "
                        f"to {expected_kind}, or connect it to the matching reference input."
                    )
                    raise ValueError(msg)

    def _is_byok_enabled(self) -> bool:
        """Whether the node is configured to use the customer's own key (BYOK) instead of Griptape auth."""
        return bool(self._api_key_provider and self._api_key_provider.is_user_auth_enabled())

    def _private_assets_active(self) -> bool:
        """Whether private-asset registration applies for this run.

        Requires Griptape auth: provider assets are registered through the Griptape Cloud proxy on
        the org's behalf, which does not apply when the user brings their own provider key (BYOK).
        """
        return SUPPORTS_PRIVATE_ASSETS and not self._is_byok_enabled()

    async def _build_payload(self) -> dict[str, Any]:
        """Build the request payload for the Seedance 2.5 API."""
        params = self._get_parameters()
        self._log(
            f"{self.name} parameter summary: "
            f"model_id={SEEDANCE_2_5_MODEL_ID}, "
            f"task={params['task']}, "
            f"first_frame_present={params['first_frame'] is not None}, "
            f"last_frame_present={params['last_frame'] is not None}, "
            f"reference_images={len(params[REFERENCE_IMAGES_PARAMETER])}, "
            f"reference_videos={len(params[REFERENCE_VIDEOS_PARAMETER])}, "
            f"reference_audio={len(params[REFERENCE_AUDIO_PARAMETER])}"
        )

        payload: dict[str, Any] = {"model": SEEDANCE_2_5_MODEL_ID}
        if params["resolution"]:
            payload["resolution"] = params["resolution"]
        if params["ratio"]:
            payload["ratio"] = params["ratio"]
        if params["duration"] is not None:
            payload["duration"] = int(params["duration"])
        if params["generate_audio"] is not None:
            payload["generate_audio"] = bool(params["generate_audio"])
        if params["output_format"]:
            payload["output_format"] = params["output_format"]
        if params["watermark"] is not None:
            payload["watermark"] = bool(params["watermark"])
        if params["return_last_frame"] is not None:
            payload["return_last_frame"] = bool(params["return_last_frame"])

        content_list: list[dict[str, Any]] = [{"type": "text", "text": params["prompt"].strip()}]
        await self._add_media_inputs_async(content_list, params)
        payload["content"] = content_list

        return payload

    async def _add_media_inputs_async(self, content_list: list[dict[str, Any]], params: dict[str, Any]) -> None:
        """Append the selected task's media entries to the content list.

        List order matters: it is what ``@Image N`` / ``@Video N`` / ``@Audio N`` in the prompt
        resolve against.
        """
        task = SeedanceTask(params["task"])
        constraints = TASK_CONSTRAINTS[task]

        if constraints.allows_frames:
            self._log(f"{self.name} building first/last-frame content")
            for parameter_name in FRAME_PARAMETERS:
                frame_url = await self._prepare_frame_url_async(params[parameter_name], frame_label=parameter_name)
                if frame_url:
                    content_list.append({"type": "image_url", "image_url": {"url": frame_url}, "role": parameter_name})
            return

        if not constraints.allows_references:
            self._log(f"{self.name} text-only task, no media inputs")
            return

        self._log(f"{self.name} building multimodal reference content for {task.value}")
        supports_assets = self._private_assets_active()
        order_log: list[str] = []

        for idx, ref_image in enumerate(params[REFERENCE_IMAGES_PARAMETER], start=1):
            if supports_assets and is_provider_asset_reference(ref_image):
                asset_url = await self._append_private_asset(
                    ref_image, expected_kind=ASSET_KIND_IMAGE, label=f"reference image {idx}"
                )
                content_list.append({"type": "image_url", "image_url": {"url": asset_url}, "role": "reference_image"})
                order_log.append(f"Image {idx}: private asset")
            else:
                ref_url = await self._prepare_frame_url_async(ref_image, frame_label="reference_image")
                if ref_url:
                    content_list.append({"type": "image_url", "image_url": {"url": ref_url}, "role": "reference_image"})
                    order_log.append(f"Image {idx}: reference")

        for idx, ref_video in enumerate(params[REFERENCE_VIDEOS_PARAMETER], start=1):
            if supports_assets and is_provider_asset_reference(ref_video):
                asset_url = await self._append_private_asset(
                    ref_video, expected_kind=ASSET_KIND_VIDEO, label=f"reference video {idx}"
                )
                content_list.append({"type": "video_url", "video_url": {"url": asset_url}, "role": "reference_video"})
                order_log.append(f"Video {idx}: private asset")
            else:
                video_url = self._get_reference_video_url(ref_video, label=f"reference video {idx}")
                content_list.append({"type": "video_url", "video_url": {"url": video_url}, "role": "reference_video"})
                order_log.append(f"Video {idx}: reference")

        for idx, ref_audio in enumerate(params[REFERENCE_AUDIO_PARAMETER], start=1):
            if supports_assets and is_provider_asset_reference(ref_audio):
                asset_url = await self._append_private_asset(
                    ref_audio, expected_kind=ASSET_KIND_AUDIO, label=f"reference audio {idx}"
                )
                content_list.append({"type": "audio_url", "audio_url": {"url": asset_url}, "role": "reference_audio"})
                order_log.append(f"Audio {idx}: private asset")
            else:
                audio_url = await self._prepare_audio_url_async(ref_audio, audio_label="reference_audio")
                if audio_url:
                    content_list.append(
                        {"type": "audio_url", "audio_url": {"url": audio_url}, "role": "reference_audio"}
                    )
                    order_log.append(f"Audio {idx}: reference")

        if order_log:
            self._log(f"{self.name} resolved reference order: " + "; ".join(order_log))

    def _get_reference_video_url(self, value: Any, *, label: str) -> str:
        """Resolve one reference video to a URL Seedance can fetch.

        Public URLs and ``asset://`` IDs pass through; anything else is uploaded to Griptape Cloud
        static storage for a temporary public URL, because Seedance does not accept video base64.
        """
        direct_url = coerce_video_url(value)
        if direct_url:
            return direct_url

        try:
            public_url = self._resolve_public_url_for_media(value, artifact_type="VideoUrlArtifact")
        except Exception as e:
            msg = f"{self.name}: failed to prepare a public URL for {label}: {e}"
            raise ValueError(msg) from e

        coerced_url = coerce_video_url(public_url)
        if not coerced_url:
            msg = (
                f"{self.name}: {label} only supports public URLs, uploaded asset URLs, or asset:// IDs. "
                "Seedance 2.5 does not accept video base64."
            )
            raise ValueError(msg)
        return coerced_url

    async def _parse_result(self, result_json: dict[str, Any], generation_id: str) -> None:
        """Parse the result and set output parameters."""
        extracted_url = extract_video_url(result_json)
        if not extracted_url:
            self.parameter_output_values["video_url"] = None
            self._set_status_results(
                was_successful=False,
                result_details=f"{self.name} generation completed but no video URL was found in the response.",
            )
            return

        await self._download_and_save(extracted_url, "video_url", lambda v, n: VideoUrlArtifact(value=v, name=n))

        if self.get_parameter_value("return_last_frame"):
            await self._save_last_frame(result_json)

    async def _save_last_frame(self, result_json: dict[str, Any]) -> None:
        """Download the returned last frame, if any, into its own project file.

        The video has already been saved at this point, so a missing or unretrievable last frame is
        logged and leaves ``last_frame_url`` unset rather than failing the whole node.
        """
        content = result_json.get("content")
        last_frame_url = content.get("last_frame_url") if isinstance(content, dict) else None
        if not isinstance(last_frame_url, str) or not last_frame_url:
            self._log(f"{self.name} return_last_frame was requested but the response contained no last frame URL")
            return

        try:
            frame_bytes = await self._download_bytes_from_url(last_frame_url)
            dest = self._last_frame_file.build_file()
            saved = await dest.awrite_bytes(frame_bytes)
        except Exception as e:
            self._log(f"{self.name} failed to retrieve the last frame from {last_frame_url}: {e}")
            return

        self.parameter_output_values["last_frame_url"] = ImageUrlArtifact(value=saved.location, name=saved.name)
        self._log(f"{self.name} saved last frame as {saved.name}")

    def _set_safe_defaults(self) -> None:
        """Clear all output parameters on error."""
        self.parameter_output_values["generation_id"] = ""
        self.parameter_output_values["provider_response"] = None
        self.parameter_output_values["video_url"] = None
        self.parameter_output_values["last_frame_url"] = None
