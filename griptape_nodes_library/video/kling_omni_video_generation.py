from __future__ import annotations

import logging
from decimal import Decimal, InvalidOperation
from functools import partial
from typing import Any

from griptape.artifacts import ImageArtifact, ImageUrlArtifact
from griptape.artifacts.video_url_artifact import VideoUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterGroup, ParameterList, ParameterMode
from griptape_nodes.exe_types.param_components.artifact_url.public_artifact_url_parameter import (
    PublicArtifactUrlParameter,
)
from griptape_nodes.exe_types.param_components.model_access_component import ModelAccessComponent
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_types.parameter_bool import ParameterBool
from griptape_nodes.exe_types.param_types.parameter_dict import ParameterDict
from griptape_nodes.exe_types.param_types.parameter_image import ParameterImage
from griptape_nodes.exe_types.param_types.parameter_int import ParameterInt
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.exe_types.param_types.parameter_video import ParameterVideo
from griptape_nodes.exe_types.validators import (
    AllowedFormats,
    AspectRatioRange,
    FrameRate,
    MaxDuration,
    MaxFileSize,
    MinDuration,
    PixelArea,
    SideRange,
)
from griptape_nodes.traits.options import Options
from griptape_nodes.utils.artifact_normalization import normalize_artifact_input

from griptape_nodes_library.media import prepare_media_data_uri
from griptape_nodes_library.proxy import ArtifactKind, GriptapeProxyNode
from griptape_nodes_library.video import kling_4

logger = logging.getLogger("griptape_nodes")

__all__ = ["KlingOmniVideoGeneration"]

# Constants
MAX_PROMPT_LENGTH = 2500
MAX_IMAGES_WITH_VIDEO = 4
MAX_IMAGES_WITHOUT_VIDEO = 7
MAX_IMAGES_FOR_END_FRAME = 2

# Kling's documented limits on POST /v1/videos/omni-video, the same on both model pages.
# Images: "Formats: .jpg / .jpeg / .png", "File size: ≤10MB", "Dimensions: min 300px, aspect
# ratio 1:2.5 ~ 2.5:1". Reference video: "Format: MP4/MOV only", "Max 1 video, size ≤200MB,
# aspect ratio between 1:2.5 and 2.5:1".
# Errors link the page for the selected model.
INPUT_LIMITS_URLS = {
    "kling-v3-omni": "https://kling.ai/document-api/api/video/3-0-omni/video-omni/legacy",
    "kling-video-o1": "https://kling.ai/document-api/api/video/o1/video-omni/legacy",
}
ASPECT_RATIO_RANGE = AspectRatioRange("1:2.5", "2.5:1")
IMAGE_LIMITS = (
    AllowedFormats("image/jpeg", "image/png"),
    MaxFileSize(10 * 1024 * 1024),
    SideRange(min=300),
    ASPECT_RATIO_RANGE,
)
# Reference video duration and frame size: the wider bounds of two Kling pages, on purpose.
#
# Both pages document this same endpoint (POST /v1/videos/omni-video) and list both models it
# serves (kling-video-o1 and kling-v3-omni), but they give different reference-video limits and
# neither says which model each set belongs to:
#   3.0 Omni page: "Duration: 3s～15.5s" and "Resolution: width and height: 700px～4553px; and
#   the area does not exceed 8294400"
#   https://kling.ai/document-api/api/video/3-0-omni/video-omni/legacy
#   O1 page: "Duration: ≥3 seconds, upper limit depends on model version" and
#   "Resolution: 720px-2160px (width and height)"
#   https://kling.ai/document-api/api/video/o1/video-omni/legacy
#
# A pre-run check must only refuse what Kling is certain to reject, and refusing a valid video is
# worse than letting a borderline one reach Kling. So the checks use the bounds neither page
# contradicts: 3 to 15.5 s (the O1 page gives no upper number) and 700 to 4553 px with the
# 8,294,400 px area cap. A 3000 px clip with O1 selected passes here even though the O1 page says
# 2160 px; Kling has the final word.
#
# Kling's newer API pages do split these by model (O1: 3-10 s and 700-2160 px), but they document
# a different endpoint, so they aren't used:
#   https://kling.ai/document-api/api/video/o1/video-omni
#   https://kling.ai/document-api/api/video/3-0-omni/video-omni
REFERENCE_VIDEO_LIMITS = (
    MaxFileSize(200 * 1024 * 1024),
    AllowedFormats("video/mp4", "video/quicktime"),
    MinDuration(3),
    MaxDuration(15.5),
    SideRange(min=700, max=4553),
    PixelArea(max_megapixels=8.2944),
    ASPECT_RATIO_RANGE,
    # "Frame rate: 24-60fps (output will be 24fps)", the same on both pages above.
    FrameRate(min=24, max=60),
)
MAX_MULTI_PROMPT_COUNT = 6
MODE_STD = "std"
MODE_PRO = "pro"
MODE_4K = "4k"
BASE_MODE_CHOICES = [MODE_STD, MODE_PRO]
DEFAULT_MODE = MODE_PRO


# Migrates values saved before the dropdown stored the provider's own model id: old
# display labels, catalog display names, and catalog keys.
LEGACY_MODEL_VALUES: dict[str, str] = {
    "Kling Omni": "kling-video-o1",
    "Kling v3.0 Omni": "kling-v3-omni",
    "gtc_kling_v3_omni": "kling-v3-omni",
    "gtc_kling_video_o1_omni": "kling-video-o1",
}
MODEL_CAPABILITIES: dict[str, dict[str, Any]] = {
    kling_4.KLING_4_0: {
        "modes": [],
        "durations": kling_4.KLING_4_CAPABILITIES[kling_4.KLING_4_0].durations,
    },
    kling_4.KLING_4_0_FLASH: {
        "modes": [],
        "durations": kling_4.KLING_4_CAPABILITIES[kling_4.KLING_4_0_FLASH].durations,
    },
    "kling-video-o1": {
        "modes": BASE_MODE_CHOICES,
        "durations": list(range(3, 11)),
        # Text-to-video and start-frame-only generation restrict to 5s or 10s
        "restricted_durations": [5, 10],
        "supports_4k_with_reference_video": False,
    },
    "kling-v3-omni": {
        "modes": [MODE_STD, MODE_PRO, MODE_4K],
        "durations": list(range(3, 16)),
        # When reference video is set (std/pro), duration caps at 10s
        "reference_video_max_duration": 10,
        "supports_4k_with_reference_video": False,
    },
}


class KlingOmniVideoGeneration(GriptapeProxyNode):
    """Generate a video using Kling Omni AI via Griptape Cloud model proxy.

    The Omni model supports various capabilities through templated prompts with elements,
    images, and videos. Use <<<element_1>>>, <<<image_1>>>, <<<video_1>>> in prompts.

    Inputs:
        - model_name (str): Model selection ("kling-video-o1" or "kling-v3-omni")
        - multi_shot (bool): Enable multi-prompt mode (default: False)
        - prompt (str): Text prompt with optional templates (max 2500 chars, required when multi_shot=False)
        - shot_count (int): Number of shots in multi-shot mode (1-6, required when multi_shot=True)
        - shot_{n}_prompt (str): Prompt for shot n (required when multi_shot=True and n <= shot_count)
        - shot_{n}_duration (str): Duration for shot n as number-as-string (required when multi_shot=True and n <= shot_count)
        - reference_images (list[ImageArtifact]): Reference images for generation (optional)
        - first_frame_image (ImageArtifact|ImageUrlArtifact|str): First frame image (optional)
        - end_frame_image (ImageArtifact|ImageUrlArtifact|str): End frame image (optional)
        - element_ids (str): Comma-separated element IDs (optional)
        - reference_video (VideoUrlArtifact): Reference video for editing or style reference (optional, max 1)
        - video_refer_type (str): Video reference type (base: edit video, feature: use as feature reference)
        - video_keep_sound (bool): Keep original video sound (default: False)
        - mode (str): Video generation mode (std: Standard, pro: Professional)
        - aspect_ratio (str): Aspect ratio (required when not using first frame or video editing)
        - duration (int): Video length in seconds (3-10s)
        (Always polls for result: 5s interval, 20 min timeout)

    Outputs:
        - generation_id (str): Griptape Cloud generation id
        - provider_response (dict): Verbatim response from API (latest polling response)
        - video_url (VideoUrlArtifact): Saved video URL
        - kling_video_id (str): The video ID from Kling AI
        - was_successful (bool): Whether the generation succeeded
        - result_details (str): Details about the generation result or error
    """

    # Set by before_value_set from the outgoing model; a new node starts on v1-v3.
    _entering_kling_4: bool = True

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # INPUTS / PROPERTIES
        model_name_param = ParameterString(
            name="model_name",
            default_value="kling-v3-omni",
            tooltip="Model to use for video generation",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            ui_options={"display_name": "Model"},
        )
        self.add_parameter(model_name_param)
        # License-policy dropdown: the component adds Options + refresh Button traits and
        # marks the models the license denies; the proxy base refuses a denied selection.
        self._model_access = ModelAccessComponent(
            node=self,
            parameter=model_name_param,
            model_choices=[kling_4.KLING_4_0, kling_4.KLING_4_0_FLASH, "kling-video-o1", "kling-v3-omni"],
            default_model="kling-v3-omni",
            deprecated_values=LEGACY_MODEL_VALUES,
        )

        self.add_parameter(
            ParameterBool(
                name="multi_shot",
                default_value=False,
                tooltip="Enable multi-shot mode. When true, uses shot prompts and sends shot_type='customize'.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"display_name": "multi shot"},
            )
        )
        self.add_parameter(
            ParameterString(
                name="prompt",
                tooltip="Text prompt with optional templates like <<<image_1>>>, <<<element_1>>> (max 2500 chars)",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                multiline=True,
                placeholder_text="Describe the video... Use <<<image_1>>> to reference images.",
            )
        )
        with ParameterGroup(name="Multi-Shot Settings") as multi_shot_settings_group:
            ParameterInt(
                name="shot_count",
                default_value=1,
                tooltip="Number of shots (1-6) when multi-shot is enabled.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=[1, 2, 3, 4, 5, 6])},
                ui_options={"display_name": "shot count", "hide": True},
            )
        self.add_node_element(multi_shot_settings_group)

        for shot_index in range(1, MAX_MULTI_PROMPT_COUNT + 1):
            with ParameterGroup(name=f"Shot {shot_index}") as shot_group:
                ParameterString(
                    name=f"shot_{shot_index}_prompt",
                    tooltip=f"Prompt for shot {shot_index}",
                    allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                    multiline=True,
                    placeholder_text=f"Describe shot {shot_index}...",
                    ui_options={"display_name": f"shot {shot_index} prompt", "hide": True},
                )
                ParameterString(
                    name=f"shot_{shot_index}_duration",
                    default_value="1",
                    tooltip=f"Duration for shot {shot_index} as number-as-string.",
                    allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                    ui_options={"display_name": f"shot {shot_index} duration", "hide": True},
                )
            self.add_node_element(shot_group)

        # Image Inputs Group
        self.add_parameter(
            ParameterImage(
                name="first_frame_image",
                tooltip="First frame image (optional). Accepts ImageArtifact, ImageUrlArtifact, URL, or Base64.",
                allowed_modes={ParameterMode.INPUT},
                hide_property=True,
                ui_options={"display_name": "first frame"},
                pre_run_validators=IMAGE_LIMITS,
            )
        )
        self.add_parameter(
            ParameterImage(
                name="end_frame_image",
                tooltip="End frame image (optional). Requires first frame to be set.",
                allowed_modes={ParameterMode.INPUT},
                hide_property=True,
                ui_options={"display_name": "end frame"},
                pre_run_validators=IMAGE_LIMITS,
            )
        )
        self.add_parameter(
            ParameterList(
                name="reference_images",
                item_converters=[
                    partial(normalize_artifact_input, artifact_type=ImageUrlArtifact, accepted_types=(ImageArtifact,))
                ],
                input_types=[
                    "ImageArtifact",
                    "ImageUrlArtifact",
                    "str",
                    "list[ImageArtifact]",
                    "list[ImageUrlArtifact]",
                    "list[str]",
                ],
                default_value=[],
                tooltip="Reference images for generation. These appear first in the template (<<<image_1>>>, <<<image_2>>>, etc.)",
                allowed_modes={ParameterMode.INPUT},
                ui_options={"expander": True, "display_name": "reference images", "hide_property": True},
                pre_run_validators=IMAGE_LIMITS,
            )
        )

        # Advanced References Group
        with ParameterGroup(name="Advanced References") as advanced_group:
            ParameterString(
                name="element_ids",
                default_value="",
                tooltip=(
                    "Comma-separated element IDs (e.g., '123,456,789'). With Kling 4.0 models they are "
                    "referenced in the prompt as @element_1, @element_2 and so on."
                ),
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"placeholder_text": "e.g., 123,456,789", "hide": True},
            )
            ParameterString(
                name="voice_ids",
                default_value="",
                tooltip=(
                    "Comma-separated voice IDs, up to 3 (Kling 4.0 only). Referenced in the prompt as "
                    "@voice_1, @voice_2 and so on. Create voices with Kling's voice API or use a preset voice ID."
                ),
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"placeholder_text": "e.g., 829837252279803904", "hide": True},
            )
        advanced_group.ui_options = {"hide": False}
        self.add_node_element(advanced_group)

        # Use PublicArtifactUrlParameter for video upload handling
        self._public_video_url_parameter = PublicArtifactUrlParameter(
            node=self,
            artifact_url_parameter=ParameterVideo(
                name="reference_video",
                tooltip="Reference video for editing or style reference (optional, max 1)",
                allowed_modes={ParameterMode.INPUT},
                ui_options={"placeholder_text": "https://example.com/video.mp4"},
                hide_property=True,
                pre_run_validators=REFERENCE_VIDEO_LIMITS,
            ),
            disclaimer_message="The Kling Omni service utilizes this URL to access the video for generation.",
        )
        self._public_video_url_parameter.add_input_parameters()

        # Generation Settings Group
        with ParameterGroup(name="Generation Settings") as gen_settings_group:
            ParameterString(
                name="video_refer_type",
                default_value="base",
                tooltip="Video reference type: 'feature' is the feature reference video, 'base' is the video to be edited",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=["base", "feature"])},
            )
            ParameterBool(
                name="video_keep_sound",
                default_value=False,
                tooltip="Keep original video sound (only applies when video_refer_type is 'base')",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"display_name": "keep video sound"},
            )
            ParameterString(
                name="mode",
                default_value=DEFAULT_MODE,
                tooltip="Video generation mode. Kling v3.0 Omni also supports 4k when reference_video is not set.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=[MODE_STD, MODE_PRO, MODE_4K])},
            )
            ParameterString(
                name="aspect_ratio",
                default_value="16:9",
                tooltip=(
                    "Aspect ratio (required when not using first frame or video editing). Kling 4.0 models also "
                    "offer 21:9 and adaptive, which follows the prompt, then the base video, keyframes or reference video."
                ),
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=[*kling_4.GENERATED_ASPECT_RATIOS, kling_4.ASPECT_RATIO_ADAPTIVE])},
            )
            ParameterString(
                name="resolution",
                default_value=kling_4.RESOLUTION_1080P,
                tooltip="Output resolution (Kling 4.0 models). Kling 4.0 Flash is 720p only.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=[kling_4.RESOLUTION_720P, kling_4.RESOLUTION_1080P])},
                ui_options={"hide": True},
            )
            ParameterString(
                name="audio",
                default_value="native",
                tooltip=(
                    "Kling 4.0 models: 'native' generates audio with the video; 'original' keeps the audio of the "
                    "reference video (needs a reference video and no voice IDs)."
                ),
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=kling_4.AUDIO_MODES)},
                ui_options={"hide": True},
            )
            ParameterInt(
                name="duration",
                default_value=5,
                tooltip="Video length in seconds. Range and restrictions vary by model.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=list(range(3, 16)))},
            )
        self.add_node_element(gen_settings_group)

        # OUTPUTS
        self.add_parameter(
            ParameterDict(
                name="provider_response",
                tooltip="Verbatim response from API (latest polling response)",
                allowed_modes={ParameterMode.OUTPUT},
                ui_options={"hide_property": True},
                hide=True,
            )
        )

        self.add_parameter(
            ParameterVideo(
                name="video_url",
                tooltip="Saved video as URL artifact for downstream display",
                allowed_modes={ParameterMode.OUTPUT, ParameterMode.PROPERTY},
                settable=False,
                ui_options={"pulse_on_run": True},
            )
        )

        self.add_parameter(
            ParameterString(
                name="kling_video_id",
                tooltip="The video ID from Kling AI",
                allowed_modes={ParameterMode.OUTPUT},
                placeholder_text="The Kling AI video ID",
            )
        )

        self._output_file = ProjectFileParameter(
            node=self,
            name="output_file",
            default_filename="kling_omni_video.mp4",
        )
        self._output_file.add_parameter()

        # Create status parameters for success/failure tracking
        self._create_status_parameters(
            result_details_tooltip="Details about the video generation result or any errors",
            result_details_placeholder="Generation status and details will appear here.",
            parameter_group_initially_collapsed=True,
        )
        self._update_mode_choices()
        self._update_multi_shot_parameter_visibility()
        self._update_kling_4_visibility()
        self.set_initial_node_size(height=1375)

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        """Handle parameter value changes to normalize image inputs."""
        super().after_value_set(parameter, value)

        if parameter.name in {"model_name", "reference_video", "end_frame_image"}:
            self._update_mode_choices()
            # Update duration choices inline (WAN pattern)
            model_name = self.get_parameter_value("model_name") or "kling-v3-omni"
            capabilities = MODEL_CAPABILITIES.get(model_name, {})
            has_reference_video = bool(self.get_parameter_value("reference_video"))
            has_end_frame = bool(self.get_parameter_value("end_frame_image"))
            new_durations = list(capabilities.get("durations", list(range(3, 11))))
            # kling-video-o1: text/start-frame-only → [5, 10]; end frame or reference video → full [3–10]
            restricted_durations = capabilities.get("restricted_durations")
            if restricted_durations and not has_reference_video and not has_end_frame:
                new_durations = list(restricted_durations)
            # kling-v3-omni: reference video with std/pro mode caps at 10s
            ref_video_max = capabilities.get("reference_video_max_duration")
            if ref_video_max and has_reference_video:
                new_durations = [d for d in new_durations if d <= ref_video_max]
            if not new_durations:
                new_durations = [5]
            current_duration = self.get_parameter_value("duration")
            if current_duration in new_durations:
                self._update_option_choices("duration", new_durations, current_duration)  # type: ignore[arg-type]
            else:
                self._update_option_choices("duration", new_durations, new_durations[0])  # type: ignore[arg-type]

        if parameter.name in {"multi_shot", "shot_count"}:
            self._update_multi_shot_parameter_visibility()

        if parameter.name == "model_name":
            self._update_kling_4_visibility(entering_kling_4=self._entering_kling_4)
        elif parameter.name in {"voice_ids", "end_frame_image", "audio"}:
            # Clearing a value an unsupported model can't use hides its input again.
            self._update_kling_4_visibility()

    def before_value_set(self, parameter: Parameter, value: Any) -> Any:
        if parameter.name == "model_name":
            # Read before the new model is stored: entering 4.0 from v1-v3 migrates
            # the legacy 16:9 default, switching between 4.0 variants does not.
            self._entering_kling_4 = not kling_4.is_kling_4(self._selected_model())
        return super().before_value_set(parameter, value)

    def _selected_model(self) -> str:
        return self.get_parameter_value("model_name") or "kling-v3-omni"

    def _wants_original_audio(self) -> bool:
        return (self.get_parameter_value("audio") or "native") == "original"

    def _update_kling_4_visibility(self, *, entering_kling_4: bool = False) -> None:
        """Kling 4.0 has no mode, multi-shot or keep-sound; it adds resolution, audio and voices.

        An input the selected model can't use stays visible while it holds a value, so
        the validation error that blocks the run points at something the user can clear.
        """
        model_id = self._selected_model()
        aspect_ratios = list(kling_4.GENERATED_ASPECT_RATIOS)
        if not kling_4.is_kling_4(model_id):
            self.hide_parameter_by_name("resolution")
            # Audio always holds a mode, so only 'original' (which v1-v3 can't honour) counts as set.
            for name, populated in (
                ("voice_ids", bool(kling_4.parse_id_list(self.get_parameter_value("voice_ids")))),
                ("audio", self._wants_original_audio()),
            ):
                if populated:
                    self.show_parameter_by_name(name)
                else:
                    self.hide_parameter_by_name(name)
            self.show_parameter_by_name(["mode", "multi_shot", "video_keep_sound", "end_frame_image"])
            aspect_ratios = ["16:9", "9:16", "1:1"]
            current = self.get_parameter_value("aspect_ratio")
            self._update_option_choices("aspect_ratio", aspect_ratios, current if current in aspect_ratios else "16:9")
            self._update_multi_shot_parameter_visibility()
            return

        capabilities = kling_4.KLING_4_CAPABILITIES[model_id]
        if self.get_parameter_value("multi_shot"):
            self.set_parameter_value("multi_shot", False)
        self.hide_parameter_by_name(["mode", "multi_shot", "video_keep_sound", "shot_count"])
        for shot_index in range(1, MAX_MULTI_PROMPT_COUNT + 1):
            self.hide_parameter_by_name([f"shot_{shot_index}_prompt", f"shot_{shot_index}_duration"])
        self.show_parameter_by_name(["prompt", "resolution", "audio", "element_ids"])
        if capabilities.supports_voices or self.get_parameter_value("voice_ids"):
            self.show_parameter_by_name("voice_ids")
        else:
            self.hide_parameter_by_name("voice_ids")
        if capabilities.supports_keyframes or self.get_parameter_value("end_frame_image"):
            self.show_parameter_by_name("end_frame_image")
        else:
            self.hide_parameter_by_name("end_frame_image")

        self._update_option_choices(
            "resolution",
            capabilities.resolutions,
            kling_4.resolution_choice(model_id, self.get_parameter_value("resolution")),
        )
        aspect_ratios.append(kling_4.ASPECT_RATIO_ADAPTIVE)
        current = self.get_parameter_value("aspect_ratio")
        # The v1-v3 default, 16:9, would override the framing of a first frame or
        # base video on 4.0; adaptive follows them and falls back to 16:9. Only on
        # entering 4.0, so a 16:9 picked on a 4.0 model survives switching to Flash.
        # A saved aspect_ratio still wins, since it is restored after model_name.
        if entering_kling_4 and current in (None, "", "16:9"):
            current = kling_4.ASPECT_RATIO_ADAPTIVE
        self._update_option_choices("aspect_ratio", aspect_ratios, current)

    def _get_supported_modes(self) -> list[str]:
        """Return the valid mode choices for the selected model and inputs."""
        model_name = self.get_parameter_value("model_name") or "kling-v3-omni"
        capabilities = MODEL_CAPABILITIES.get(model_name, {"modes": BASE_MODE_CHOICES})
        supported_modes = list(capabilities.get("modes", BASE_MODE_CHOICES))

        if self.get_parameter_value("reference_video") and not capabilities.get(
            "supports_4k_with_reference_video", True
        ):
            supported_modes = [mode for mode in supported_modes if mode != MODE_4K]

        return supported_modes

    def _update_mode_choices(self) -> None:
        """Keep the mode dropdown aligned with the selected model and inputs."""
        supported_modes = self._get_supported_modes()
        if not supported_modes:
            return
        current_mode = self.get_parameter_value("mode")
        next_mode = current_mode if current_mode in supported_modes else DEFAULT_MODE
        if next_mode not in supported_modes:
            next_mode = supported_modes[0]

        self._update_option_choices("mode", supported_modes, next_mode)

    def _update_multi_shot_parameter_visibility(self) -> None:
        """Toggle legacy prompt vs per-shot inputs based on multi-shot settings."""
        is_multi_shot = bool(self.get_parameter_value("multi_shot"))
        raw_shot_count = self.get_parameter_value("shot_count") or 1
        try:
            shot_count = int(raw_shot_count)
        except (TypeError, ValueError):
            shot_count = 1
        shot_count = max(1, min(MAX_MULTI_PROMPT_COUNT, shot_count))

        if is_multi_shot:
            self.hide_parameter_by_name("prompt")
            self.show_parameter_by_name("shot_count")
            for shot_index in range(1, MAX_MULTI_PROMPT_COUNT + 1):
                prompt_name = f"shot_{shot_index}_prompt"
                duration_name = f"shot_{shot_index}_duration"
                if shot_index <= shot_count:
                    self.show_parameter_by_name([prompt_name, duration_name])
                else:
                    self.hide_parameter_by_name([prompt_name, duration_name])
        else:
            self.show_parameter_by_name("prompt")
            self.hide_parameter_by_name("shot_count")
            for shot_index in range(1, MAX_MULTI_PROMPT_COUNT + 1):
                self.hide_parameter_by_name([f"shot_{shot_index}_prompt", f"shot_{shot_index}_duration"])

    async def aprocess(self) -> None:
        try:
            await super().aprocess()
        finally:
            # Always cleanup uploaded video artifact
            await self._public_video_url_parameter.adelete_uploaded_artifact()

    def _get_api_model_id(self) -> str:
        """Get the API model ID for this generation.

        Builds the URL-path model id from the current selection's provider id.
        """
        return f"{self._get_selected_model_id()}:omnivideo"

    def _build_customize_multi_prompt_payload(self, shot_count: Any) -> list[dict[str, Any]]:
        """Build multi_prompt payload from shot input parameters."""
        try:
            shot_count_int = int(shot_count)
        except (TypeError, ValueError):
            shot_count_int = 1
        shot_count_int = max(1, min(MAX_MULTI_PROMPT_COUNT, shot_count_int))
        return [
            {
                "index": shot_index,
                "prompt": (self.get_parameter_value(f"shot_{shot_index}_prompt") or "").strip(),
                "duration": str(self.get_parameter_value(f"shot_{shot_index}_duration") or "").strip(),
            }
            for shot_index in range(1, shot_count_int + 1)
        ]

    def _validate_customize_multi_shot(self, exceptions: list[Exception], shot_count: Any, duration: Any) -> None:  # noqa: C901
        """Validate customize multi-shot inputs."""
        try:
            shot_count_int = int(shot_count)
        except (TypeError, ValueError):
            exceptions.append(ValueError(f"'shot count' must be an integer between 1 and {MAX_MULTI_PROMPT_COUNT}."))
            shot_count_int = 1

        if not (1 <= shot_count_int <= MAX_MULTI_PROMPT_COUNT):
            exceptions.append(
                ValueError(f"'shot count' must be between 1 and {MAX_MULTI_PROMPT_COUNT} (got {shot_count_int}).")
            )
        shot_count_int = max(1, min(MAX_MULTI_PROMPT_COUNT, shot_count_int))

        total_duration = Decimal(0)
        try:
            requested_duration = Decimal(str(duration))
        except InvalidOperation:
            exceptions.append(ValueError(f"Duration must be numeric (got {duration})."))
            requested_duration = Decimal(0)

        for shot_index in range(1, shot_count_int + 1):
            component_prompt = (self.get_parameter_value(f"shot_{shot_index}_prompt") or "").strip()
            component_duration = self.get_parameter_value(f"shot_{shot_index}_duration")

            if not component_prompt:
                exceptions.append(ValueError(f"Shot {shot_index} prompt must be a non-empty string."))
            elif len(component_prompt) > MAX_PROMPT_LENGTH:
                exceptions.append(
                    ValueError(
                        f"Shot {shot_index} prompt exceeds {MAX_PROMPT_LENGTH} characters "
                        f"(got: {len(component_prompt)} characters)."
                    )
                )

            if not isinstance(component_duration, str):
                exceptions.append(ValueError(f"Shot {shot_index} duration must be a string number."))
                continue

            try:
                component_duration_decimal = Decimal(component_duration)
            except InvalidOperation:
                exceptions.append(
                    ValueError(
                        f"Shot {shot_index} has invalid duration '{component_duration}'. Expected a number-as-string."
                    )
                )
                continue

            if component_duration_decimal < 1:
                exceptions.append(
                    ValueError(f"Shot {shot_index} duration must be at least 1 (got {component_duration}).")
                )
            if component_duration_decimal > requested_duration:
                exceptions.append(
                    ValueError(
                        f"Shot {shot_index} duration cannot exceed requested duration "
                        f"{duration} (got {component_duration})."
                    )
                )
            total_duration += component_duration_decimal

        if total_duration != requested_duration:
            exceptions.append(ValueError(f"Multi-shot durations must sum to {duration} (got {total_duration})."))

    async def _build_payload(self) -> dict[str, Any]:  # noqa: C901, PLR0912, PLR0915
        """Build the request payload for Kling Omni API.

        Images are ordered: reference images FIRST, then start/end frames at the END.
        This allows users to reference <<<image_1>>>, <<<image_2>>> in the order they
        supplied the reference images.

        Returns:
            dict: The request payload (model field excluded, handled by base class)
        """
        model_name = self.get_parameter_value("model_name") or "kling-v3-omni"
        if kling_4.is_kling_4(model_name):
            return await self._build_kling_4_payload(model_name)
        prompt = (self.get_parameter_value("prompt") or "").strip()
        multi_shot = bool(self.get_parameter_value("multi_shot"))
        shot_count = self.get_parameter_value("shot_count") or 1
        reference_images = self.get_parameter_value("reference_images") or []
        first_frame_image = await self._prepare_image_data_url_async(self.get_parameter_value("first_frame_image"))
        end_frame_image = await self._prepare_image_data_url_async(self.get_parameter_value("end_frame_image"))
        element_ids = (self.get_parameter_value("element_ids") or "").strip()

        # Get video parameters - use PublicArtifactUrlParameter to get public URL
        reference_video_param = self.get_parameter_value("reference_video")
        reference_video_url = None
        if reference_video_param:
            reference_video_url = await self._public_video_url_parameter.aget_public_url_for_parameter()

        video_keep_sound = self.get_parameter_value("video_keep_sound")
        if video_keep_sound is None:
            video_keep_sound = False

        video_refer_type = self.get_parameter_value("video_refer_type") or "base"
        mode = self.get_parameter_value("mode") or DEFAULT_MODE
        aspect_ratio = self.get_parameter_value("aspect_ratio") or "16:9"
        duration = self.get_parameter_value("duration") or 5

        payload: dict[str, Any] = {
            "model_name": model_name,
            "mode": mode,
            "aspect_ratio": aspect_ratio,
            "duration": int(duration),
            "multi_shot": multi_shot,
        }
        if multi_shot:
            payload["prompt"] = ""
            payload["multi_prompt"] = self._build_customize_multi_prompt_payload(shot_count)
            payload["shot_type"] = "customize"
        else:
            payload["prompt"] = prompt

        # Build image_list array: reference images FIRST, then start/end frames at END
        image_list = []

        # Add reference images first (so they are image_1, image_2, etc. in order)
        # Reference images don't have a "type" field
        ref_images_input = reference_images
        if not isinstance(ref_images_input, list):
            ref_images_input = [ref_images_input] if ref_images_input else []

        for ref_image in ref_images_input:
            if ref_image:
                ref_image_url = await self._prepare_image_data_url_async(ref_image)
                if ref_image_url:
                    image_list.append({"image_url": ref_image_url})

        # Add start and end frames at the END with explicit type field
        if first_frame_image:
            image_list.append({"image_url": first_frame_image, "type": "first_frame"})
        if end_frame_image:
            image_list.append({"image_url": end_frame_image, "type": "end_frame"})

        if image_list:
            payload["image_list"] = image_list

        # Parse element IDs from comma-separated string
        if element_ids:
            element_list = []
            try:
                element_ids_parts = [part.strip() for part in element_ids.split(",") if part.strip()]
                element_list = [{"element_id": int(eid)} for eid in element_ids_parts]
            except ValueError:
                pass  # Validation already happened in validate_before_node_run
            if element_list:
                payload["element_list"] = element_list

        # Add video if provided
        if reference_video_url:
            keep_sound_str = "yes" if video_keep_sound else "no"
            payload["video_list"] = [
                {
                    "video_url": reference_video_url,
                    "refer_type": video_refer_type,
                    "keep_original_sound": keep_sound_str,
                }
            ]

        return payload

    def _reference_image_inputs(self) -> list[Any]:
        images = self.get_parameter_value("reference_images") or []
        if not isinstance(images, list):
            images = [images]
        return [image for image in images if image]

    async def _build_kling_4_payload(self, model_id: str) -> dict[str, Any]:
        """Kling 4.0 `contents[]`: first frame first, last frame last, reference images between.

        Each reference gets an id the prompt can use: @image_1.., @video_1, @element_1.., @voice_1..
        """
        capabilities = kling_4.KLING_4_CAPABILITIES[model_id]
        contents: list[dict[str, Any]] = [
            {"type": "prompt", "text": (self.get_parameter_value("prompt") or "").strip()}
        ]

        # A supplied image that won't load raises, so no reference id is silently dropped.
        if self.get_parameter_value("first_frame_image"):
            first_frame = await kling_4.prepare_image(
                self.get_parameter_value("first_frame_image"), label="first frame", node_name=self.name
            )
            contents.append(kling_4.image_item("first_frame", first_frame, "first_frame"))

        for index, reference_image in enumerate(self._reference_image_inputs(), start=1):
            image = await kling_4.prepare_image(reference_image, label="reference images", node_name=self.name)
            contents.append(kling_4.image_item("refer_image", image, f"image_{index}"))

        if capabilities.supports_keyframes and self.get_parameter_value("end_frame_image"):
            last_frame = await kling_4.prepare_image(
                self.get_parameter_value("end_frame_image"), label="end frame", node_name=self.name
            )
            contents.append(kling_4.image_item("last_frame", last_frame, "last_frame"))

        if self.get_parameter_value("reference_video"):
            video_url = await self._public_video_url_parameter.aget_public_url_for_parameter()
            video_type = (
                "base_video" if (self.get_parameter_value("video_refer_type") or "base") == "base" else "feature_video"
            )
            contents.append({"type": video_type, "url": video_url, "id": "video_1"})

        for index, element_id in enumerate(kling_4.parse_id_list(self.get_parameter_value("element_ids")), start=1):
            contents.append({"type": "element", "element_id": element_id, "id": f"element_{index}"})

        if capabilities.supports_voices:
            for index, voice_id in enumerate(kling_4.parse_id_list(self.get_parameter_value("voice_ids")), start=1):
                contents.append({"type": "voice", "voice_id": voice_id, "id": f"voice_{index}"})

        return {
            "contents": contents,
            "settings": kling_4.build_settings(
                resolution=self.get_parameter_value("resolution") or capabilities.default_resolution,
                duration=int(self.get_parameter_value("duration") or 5),
                aspect_ratio=self.get_parameter_value("aspect_ratio") or kling_4.ASPECT_RATIO_ADAPTIVE,
                audio=self.get_parameter_value("audio") or "native",
            ),
        }

    def _validate_kling_4(self, exceptions: list[Exception], model_id: str) -> list[Exception]:  # noqa: C901
        capabilities = kling_4.KLING_4_CAPABILITIES[model_id]
        exceptions += kling_4.validate_common(
            model_id,
            prompt=self.get_parameter_value("prompt"),
            resolution=self.get_parameter_value("resolution"),
            duration=self.get_parameter_value("duration"),
        )

        has_end_frame = bool(self.get_parameter_value("end_frame_image"))
        if has_end_frame and not capabilities.supports_keyframes:
            exceptions.append(
                ValueError(f"{model_id} does not take an end frame. Clear 'end frame', or select Kling 4.0 to use it.")
            )
        image_count = (
            len(self._reference_image_inputs())
            + (1 if self.get_parameter_value("first_frame_image") else 0)
            + (1 if has_end_frame and capabilities.supports_keyframes else 0)
        )
        if image_count > kling_4.MAX_IMAGES:
            exceptions.append(ValueError(f"Kling 4.0 accepts up to {kling_4.MAX_IMAGES} images (got {image_count})."))

        element_ids = kling_4.parse_id_list(self.get_parameter_value("element_ids"))
        if len(element_ids) > kling_4.MAX_ELEMENTS:
            exceptions.append(
                ValueError(f"Kling 4.0 accepts up to {kling_4.MAX_ELEMENTS} elements (got {len(element_ids)}).")
            )

        voice_ids = kling_4.parse_id_list(self.get_parameter_value("voice_ids"))
        if voice_ids and not capabilities.supports_voices:
            exceptions.append(
                ValueError(f"{model_id} does not take voice IDs. Clear 'voice_ids', or select Kling 4.0 to use them.")
            )
            voice_ids = []
        if len(voice_ids) > kling_4.MAX_VOICES:
            exceptions.append(
                ValueError(f"Kling 4.0 accepts up to {kling_4.MAX_VOICES} voices (got {len(voice_ids)}).")
            )

        has_video = bool(self.get_parameter_value("reference_video"))
        total = image_count + len(element_ids) + len(voice_ids) + (1 if has_video else 0)
        if total > kling_4.MAX_REFERENCES:
            exceptions.append(
                ValueError(
                    f"Kling 4.0 accepts up to {kling_4.MAX_REFERENCES} images, videos, elements and voices "
                    f"combined (got {total})."
                )
            )

        if self._wants_original_audio():
            if not has_video:
                exceptions.append(ValueError("Audio 'original' keeps the reference video's audio; connect a video."))
            if voice_ids:
                exceptions.append(ValueError("Audio 'original' cannot be combined with voice IDs."))

        return exceptions

    async def _parse_result(self, result_json: dict[str, Any], generation_id: str) -> None:
        """Save the hosted video and record Kling's own id for it.

        Kling 4.0 reports the id in `data[].outputs[]`; v1-v3 in `data.task_result.videos[]`.
        """
        video_id = kling_4.kling_video_id(result_json)
        if video_id:
            self.parameter_output_values["kling_video_id"] = video_id
            logger.info("Video ID: %s", video_id)

        await self._save_generated_media(
            generation_id,
            "video_url",
            lambda v, n: VideoUrlArtifact(value=v, name=n),
            kind=ArtifactKind.VIDEO,
        )

    def _set_safe_defaults(self) -> None:
        """Clear output parameters on error."""
        self.parameter_output_values["video_url"] = None
        self.parameter_output_values["kling_video_id"] = ""

    def validate_before_node_run(self) -> list[Exception] | None:  # noqa: C901, PLR0912, PLR0915
        """Validate parameters before execution."""
        exceptions = super().validate_before_node_run() or []

        if kling_4.is_kling_4(self._selected_model()):
            return self._validate_kling_4(exceptions, self._selected_model()) or None

        # The v1-v3 payload has no voices or audio mode; running without them would drop them silently.
        model_id = self._selected_model()
        if kling_4.parse_id_list(self.get_parameter_value("voice_ids")):
            exceptions.append(
                ValueError(f"{model_id} does not take voice IDs. Clear 'voice_ids', or select Kling 4.0 to use them.")
            )
        if self._wants_original_audio():
            exceptions.append(
                ValueError(
                    f"{model_id} does not take 'audio'. Set it back to 'native' and use 'video_keep_sound' "
                    "to keep the reference video's audio, or select Kling 4.0."
                )
            )

        # Get parameter values
        prompt = (self.get_parameter_value("prompt") or "").strip()
        multi_shot = bool(self.get_parameter_value("multi_shot"))
        shot_count = self.get_parameter_value("shot_count") or 1
        reference_images = self.get_parameter_value("reference_images") or []
        first_frame_image = self.get_parameter_value("first_frame_image")
        end_frame_image = self.get_parameter_value("end_frame_image")
        element_ids = (self.get_parameter_value("element_ids") or "").strip()
        reference_video_param = self.get_parameter_value("reference_video")
        duration = self.get_parameter_value("duration") or 5
        mode = self.get_parameter_value("mode") or DEFAULT_MODE

        if multi_shot:
            self._validate_customize_multi_shot(exceptions, shot_count, duration)
        else:
            # Validate prompt is provided
            if not prompt:
                exceptions.append(ValueError("A prompt is required to generate video. Enter one in 'prompt'."))

            # Validate prompt length
            if len(prompt) > MAX_PROMPT_LENGTH:
                exceptions.append(
                    ValueError(f"'prompt' exceeds {MAX_PROMPT_LENGTH} characters (got: {len(prompt)} characters).")
                )

        # Parse element IDs from comma-separated string
        element_list = []
        if element_ids:
            try:
                element_ids_parts = [part.strip() for part in element_ids.split(",") if part.strip()]
                element_list = [{"element_id": int(eid)} for eid in element_ids_parts]
            except ValueError:
                exceptions.append(ValueError("'element_ids' must be comma-separated integers."))

        # Build video list
        has_video = bool(reference_video_param)
        supported_modes = self._get_supported_modes()
        if mode not in supported_modes:
            if mode == MODE_4K and has_video:
                exceptions.append(
                    ValueError(
                        f"Model Kling v3.0 Omni does not support mode '{MODE_4K}' when reference_video is set. "
                        f"Valid modes: {', '.join(supported_modes)}"
                    )
                )
            else:
                exceptions.append(
                    ValueError(
                        f"Selected configuration does not support mode '{mode}'. "
                        f"Valid modes: {', '.join(supported_modes)}"
                    )
                )

        # Get reference images (already a list from ParameterList)
        ref_images_input = reference_images
        if not isinstance(ref_images_input, list):
            ref_images_input = [ref_images_input] if ref_images_input else []

        # Validate end frame requires first frame
        if end_frame_image and not first_frame_image:
            exceptions.append(ValueError("'end frame' requires 'first frame' to be set."))

        # Count total images and elements
        total_image_count = len(ref_images_input)
        if first_frame_image:
            total_image_count += 1
        if end_frame_image:
            total_image_count += 1
        total_element_count = len(element_list)

        # Validate image + element count limits
        if has_video:
            if total_image_count + total_element_count > MAX_IMAGES_WITH_VIDEO:
                exceptions.append(
                    ValueError(
                        f"When using reference videos, the sum of images ({total_image_count}) "
                        f"and elements ({total_element_count}) cannot exceed {MAX_IMAGES_WITH_VIDEO}."
                    )
                )
        elif total_image_count + total_element_count > MAX_IMAGES_WITHOUT_VIDEO:
            exceptions.append(
                ValueError(
                    f"The sum of images ({total_image_count}) "
                    f"and elements ({total_element_count}) cannot exceed {MAX_IMAGES_WITHOUT_VIDEO}."
                )
            )

        # Validate end frame not allowed with >2 images
        if end_frame_image and total_image_count > MAX_IMAGES_FOR_END_FRAME:
            exceptions.append(
                ValueError(f"End frame is not supported when there are more than {MAX_IMAGES_FOR_END_FRAME} images.")
            )

        # Validate video editing cannot be used with first/end frames
        video_refer_type = self.get_parameter_value("video_refer_type") or "base"
        has_base_video = has_video and video_refer_type == "base"
        if has_base_video and (first_frame_image or end_frame_image):
            exceptions.append(
                ValueError("Video editing (refer_type='base') cannot be used with first or end frame images.")
            )

        # kling-video-o1: text-to-video and start-frame-only generation restrict to 5s or 10s
        model_name = self.get_parameter_value("model_name") or "kling-v3-omni"
        payload_model = model_name
        capabilities = MODEL_CAPABILITIES.get(payload_model, {})

        if payload_model == "kling-video-o1":
            is_text_or_first_frame = not has_base_video and not end_frame_image
            restricted = capabilities.get("restricted_durations", [5, 10])
            if is_text_or_first_frame and duration not in restricted:
                exceptions.append(
                    ValueError(
                        f"kling-video-o1 text-to-video and start-frame-only generation "
                        f"only support {restricted} second durations (got {duration}s)."
                    )
                )

        # kling-v3-omni: reference video with std/pro mode caps duration at 10s
        if payload_model == "kling-v3-omni" and has_video:
            video_refer_type = self.get_parameter_value("video_refer_type") or "base"
            mode = self.get_parameter_value("mode") or DEFAULT_MODE
            ref_video_max = capabilities.get("reference_video_max_duration", 10)
            if mode != MODE_4K and duration > ref_video_max:
                exceptions.append(
                    ValueError(
                        f"kling-v3-omni with reference video (std/pro mode) "
                        f"supports a maximum duration of {ref_video_max}s (got {duration}s)."
                    )
                )

        return exceptions if exceptions else None

    def input_limits_url(self) -> str | None:
        """Kling's limits page for the selected model; the 3.0 Omni page when the selection is unknown."""
        return INPUT_LIMITS_URLS.get(self._get_selected_model_id(), INPUT_LIMITS_URLS["kling-v3-omni"])

    async def _prepare_image_data_url_async(self, image_input: Any) -> str | None:
        """Load an image input as raw base64; Kling expects no ``data:`` prefix."""
        data_uri = await prepare_media_data_uri(image_input, kind="image", node_name=self.name)
        if not data_uri:
            return None
        return data_uri.split(";base64,", 1)[-1]
