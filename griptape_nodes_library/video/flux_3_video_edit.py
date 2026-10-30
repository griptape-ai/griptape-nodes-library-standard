from __future__ import annotations

import logging
from contextlib import suppress
from typing import Any, ClassVar

from griptape.artifacts.video_url_artifact import VideoUrlArtifact
from griptape_nodes.exe_types.core_types import NodeError, ParameterGroup, ParameterMode
from griptape_nodes.exe_types.param_components.model_access_component import ModelAccessComponent
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_types.parameter_dict import ParameterDict
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.exe_types.param_types.parameter_video import ParameterVideo
from griptape_nodes.traits.options import Options

from griptape_nodes_library.proxy import ArtifactKind, GriptapeProxyNode
from griptape_nodes_library.video.flux_3_common import (
    SAFETY_TOLERANCE_OPTIONS,
    Flux3SafetyTolerance,
    has_media_value,
    parse_flux_3_safety_tolerance,
    prepare_mp4_video_data_uri,
    probe_video_duration,
)

logger = logging.getLogger("griptape_nodes")

__all__ = ["Flux3VideoEdit"]

DEFAULT_MODEL = "flux-video-edit-v1"
MODEL_OPTIONS = [DEFAULT_MODEL]
MAX_INPUT_DURATION_SECONDS = 15.0
MAX_PROMPT_LENGTH = 4096

LEGACY_MODEL_VALUES: dict[str, str] = {
    "FLUX Video Edit [fast]": DEFAULT_MODEL,
    "gtc_flux_video_edit_v1": DEFAULT_MODEL,
}


class Flux3VideoEdit(GriptapeProxyNode):
    """Edit videos using Black Forest Labs FLUX Video Edit [fast] via Griptape model proxy.

    Inputs:
        - model (str): FLUX video edit model ("flux-video-edit-v1")
        - prompt (str): Edit instruction (1 to 4096 characters)
        - video (VideoUrlArtifact): Source MP4 video clip to edit (up to 15 seconds)
        - safety_tolerance (str): Content moderation preset ("least restrictive", "moderate", or "most restrictive")

    Outputs:
        - generation_id (str): Generation ID from the API
        - provider_response (dict): Verbatim response from the model proxy
        - video_url (VideoUrlArtifact): Edited video
        - was_successful (bool): Whether the edit succeeded
        - result_details (str): Details about the edit result or error
    """

    LEGACY_MODEL_VALUES: ClassVar[dict[str, str]] = LEGACY_MODEL_VALUES

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.category = "API Nodes"
        self.description = "Edit videos using FLUX Video Edit [fast] via Griptape model proxy"

        model_param = ParameterString(
            name="model",
            default_value=DEFAULT_MODEL,
            tooltip="Select the FLUX video edit model to use",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
        )
        self.add_parameter(model_param)
        self._model_access = ModelAccessComponent(
            node=self,
            parameter=model_param,
            model_choices=MODEL_OPTIONS,
            default_model=DEFAULT_MODEL,
            deprecated_values=LEGACY_MODEL_VALUES,
        )

        self.add_parameter(
            ParameterString(
                name="prompt",
                tooltip="Instruction describing how to edit the video (1-4096 characters)",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                multiline=True,
                placeholder_text="Describe the edits you want to make...",
                allow_output=False,
                ui_options={"display_name": "Prompt"},
            )
        )

        video_param = ParameterVideo(
            name="video",
            default_value="",
            tooltip="Input MP4 video to edit (up to 15 seconds, max 50MB)",
            allowed_modes={ParameterMode.INPUT},
            hide_property=True,
            ui_options={"display_name": "Video"},
        )
        video_param.set_badge(
            variant="info",
            title="Video requirements",
            message=(
                f"- .mp4 file\n- {MAX_INPUT_DURATION_SECONDS:g} seconds or less\n- At least 160x160 pixels\n\n"
                "Use a Trim Video node to convert or shorten a video."
            ),
        )
        self.add_parameter(video_param)

        with ParameterGroup(name="Generation Settings", ui_options={"collapsed": True}) as generation_settings_group:
            ParameterString(
                name="safety_tolerance",
                default_value=Flux3SafetyTolerance.MODERATE.value,
                tooltip="Content moderation strictness level",
                allow_output=False,
                traits={Options(choices=SAFETY_TOLERANCE_OPTIONS)},
            )

        self.add_node_element(generation_settings_group)

        # OUTPUTS
        self.add_parameter(
            ParameterDict(
                name="provider_response",
                tooltip="Verbatim response from Griptape model proxy",
                allowed_modes={ParameterMode.OUTPUT},
                hide_property=True,
                hide=True,
            )
        )

        self.add_parameter(
            ParameterVideo(
                name="video_url",
                tooltip="Edited video",
                allowed_modes={ParameterMode.OUTPUT, ParameterMode.PROPERTY},
                settable=False,
                ui_options={"pulse_on_run": True},
            )
        )

        self._output_file = ProjectFileParameter(
            node=self,
            name="output_file",
            default_filename="flux_3_video_edit.mp4",
        )
        self._output_file.add_parameter()

        self._create_status_parameters(
            result_details_tooltip="Details about the video editing result or any errors",
            result_details_placeholder="Editing status and details will appear here.",
            parameter_group_initially_collapsed=True,
        )

    def _log(self, message: str) -> None:
        with suppress(Exception):
            logger.info(message)

    def _probe_duration(self, video_value: str) -> float | None:
        return probe_video_duration(video_value, node_name=self.name)

    async def _prepare_video_data_uri(self, video_input: Any) -> str:
        return await prepare_mp4_video_data_uri(
            video_input,
            param_label="Video",
            action_verb="edit",
            service_label="FLUX Video Edit",
            max_duration_seconds=MAX_INPUT_DURATION_SECONDS,
            duration_prober=self._probe_duration,
        )

    def validate_before_node_run(self) -> list[Exception] | None:
        exceptions = super().validate_before_node_run() or []

        prompt = (self.get_parameter_value("prompt") or "").strip()
        if not prompt:
            exceptions.append(NodeError("Enter an edit instruction in 'Prompt'."))
        elif len(prompt) > MAX_PROMPT_LENGTH:
            exceptions.append(
                NodeError(f"'Prompt' must be at most {MAX_PROMPT_LENGTH} characters. It has {len(prompt)} characters.")
            )

        video_value = self.get_parameter_value("video")
        if not has_media_value(video_value):
            exceptions.append(NodeError("Connect a video to 'Video'."))

        return exceptions or None

    async def _build_payload(self) -> dict[str, Any]:
        prompt = (self.get_parameter_value("prompt") or "").strip()
        video_data_uri = await self._prepare_video_data_uri(self.get_parameter_value("video"))
        safety_tolerance = parse_flux_3_safety_tolerance(self.get_parameter_value("safety_tolerance"))

        return {
            "video": video_data_uri,
            "prompt": prompt,
            "safety_tolerance": safety_tolerance,
        }

    async def _parse_result(self, _result_json: dict[str, Any], generation_id: str) -> None:
        await self._save_generated_media(
            generation_id,
            "video_url",
            lambda v, n: VideoUrlArtifact(value=v, name=n),
            kind=ArtifactKind.VIDEO,
            action="edited",
        )

    def _set_safe_defaults(self) -> None:
        self.parameter_output_values["generation_id"] = ""
        self.parameter_output_values["provider_response"] = None
        self.parameter_output_values["video_url"] = None
