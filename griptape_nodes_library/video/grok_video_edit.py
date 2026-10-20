from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import urlsplit

from griptape.artifacts.video_url_artifact import VideoUrlArtifact
from griptape_nodes.exe_types.core_types import ParameterMode
from griptape_nodes.exe_types.input_checks import check_duration, check_mime_type, input_subject, local_video_duration
from griptape_nodes.exe_types.param_components.artifact_url.public_artifact_url_parameter import (
    PublicArtifactUrlParameter,
)
from griptape_nodes.exe_types.param_components.model_access_component import ModelAccessComponent
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_types.parameter_dict import ParameterDict
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.exe_types.param_types.parameter_video import ParameterVideo
from griptape_nodes.files.file import File, FileLoadError
from griptape_nodes.files.path_utils import is_url

from griptape_nodes_library.media import coerce_media_url_or_data_uri, is_publicly_reachable_url
from griptape_nodes_library.media.public_urls import adelete_uploaded_artifacts
from griptape_nodes_library.proxy import ArtifactKind, GriptapeProxyNode

logger = logging.getLogger("griptape_nodes")

__all__ = ["GrokVideoEdit"]

# xAI's documented input limits for video editing. xAI accepts a request that breaks them and
# then fails the job with no reason, so they are checked before the node runs.
XAI_VIDEO_EDIT_DOCS_URL = "https://docs.x.ai/developers/model-capabilities/video/editing"
MAX_INPUT_DURATION_SECONDS = 8.7
SUPPORTED_MIME_TYPE = "video/mp4"


class GrokVideoEdit(GriptapeProxyNode):
    """Edit videos using Grok video models via Griptape model proxy.

    Inputs:
        - model (str): Grok video model to use
        - prompt (str): Prompt for video editing
        - video (VideoUrlArtifact): Input video to edit (required)

    Outputs:
        - generation_id (str): Generation ID from the API
        - provider_response (dict): Verbatim response from the model proxy
        - video_url (VideoUrlArtifact): Edited video
        - was_successful (bool): Whether the edit succeeded
        - result_details (str): Details about the edit result or error
    """

    # Migrates values saved before the dropdown stored the provider's own model id.
    LEGACY_MODEL_VALUES: ClassVar[dict[str, str]] = {
        "Grok Imagine Video": "grok-imagine-video",
        "gtc_grok_imagine_video": "grok-imagine-video",
    }

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.category = "API Nodes"
        self.description = "Edit videos using Grok video models via Griptape model proxy"

        model_param = ParameterString(
            name="model",
            default_value="grok-imagine-video",
            tooltip="Select the Grok video model to use",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
        )
        self.add_parameter(model_param)
        # License-policy dropdown: the component adds Options + refresh Button traits and
        # marks the models the license denies; the proxy base refuses a denied selection.
        self._model_access = ModelAccessComponent(
            node=self,
            parameter=model_param,
            model_choices=["grok-imagine-video"],
            default_model="grok-imagine-video",
            deprecated_values=self.LEGACY_MODEL_VALUES,
        )

        self.add_parameter(
            ParameterString(
                name="prompt",
                tooltip="Prompt for video editing",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                multiline=True,
                placeholder_text="Describe the edits you want to make...",
                allow_output=False,
            )
        )

        video_param = ParameterVideo(
            name="video",
            default_value="",
            tooltip="Input video to edit",
            allowed_modes={ParameterMode.INPUT},
            hide_property=True,
            ui_options={"display_name": "Video"},
        )
        video_param.set_badge(
            variant="info",
            title="Video requirements",
            message=(
                f"- .mp4 file\n- {MAX_INPUT_DURATION_SECONDS:g} seconds or less\n\n"
                "Use a Trim Video node to convert or shorten a video.\n\n"
                "Running this node uploads a local video to a short-lived public URL, which xAI fetches."
            ),
        )
        self.add_parameter(video_param)
        self._video_param = video_param
        self._video_upload: PublicArtifactUrlParameter | None = None

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
            default_filename="grok_video_edit.mp4",
        )
        self._output_file.add_parameter()

        self._create_status_parameters(
            result_details_tooltip="Details about the video editing result or any errors",
            result_details_placeholder="Editing status and details will appear here.",
            parameter_group_initially_collapsed=True,
        )

        self.set_initial_node_size(height=1145)

    @staticmethod
    def _has_media_value(value: Any) -> bool:
        if value is None:
            return False
        if hasattr(value, "value"):
            return bool(value.value)
        return bool(value)

    @staticmethod
    def _video_mime_type(video_value: str) -> str | None:
        """Return the MIME type named by a data URI's header or a path's or URL's extension, or None.

        No extension carries no signal (a signed URL that strips the filename, say), so it is
        let through. Only a URL is split, so a local path keeps a "#" or "?" in its filename.
        """
        if video_value.startswith("data:"):
            return video_value.removeprefix("data:").split(",", 1)[0].split(";", 1)[0].lower() or None
        location = urlsplit(video_value).path if is_url(video_value) else video_value
        extension = Path(location).suffix.lstrip(".").lower()
        return f"video/{extension}" if extension else None

    def _find_input_limit_errors(self, video_input: Any) -> list[Exception]:
        """Return an error per xAI input limit the video breaks, or an empty list.

        The duration is only read for a file on this machine. A remote URL or a data URI goes
        unchecked and xAI enforces the limit.
        """
        video_value = coerce_media_url_or_data_uri(video_input, kind="video")
        if not video_value:
            return []
        subject = input_subject("Video", noun="video")
        errors = [
            check_mime_type(
                self._video_mime_type(video_value),
                subject=subject,
                allowed=(SUPPORTED_MIME_TYPE,),
                docs_url=XAI_VIDEO_EDIT_DOCS_URL,
            ),
            check_duration(
                local_video_duration(video_value),
                subject=subject,
                maximum=MAX_INPUT_DURATION_SECONDS,
                docs_url=XAI_VIDEO_EDIT_DOCS_URL,
            ),
        ]
        return [error for error in errors if error is not None]

    async def aprocess(self) -> None:
        try:
            await super().aprocess()
        finally:
            # Only once polling has finished: xAI fetches the video from this URL when the job
            # starts, so deleting it any earlier would pull the video out from under a queued job.
            if self._video_upload is not None:
                await adelete_uploaded_artifacts([self._video_upload], node_name=self.name)

    async def _upload_video(self) -> str | None:
        """Return a public URL for the video input, or None to fall back to a data URI.

        A local video is uploaded to Griptape Cloud so xAI can fetch it, keeping it out of the
        request body. A public http(s) URL is passed through. The helper is built here rather than
        in __init__ because it raises without a Griptape Cloud credential, which should cost the
        upload, not the node.
        """
        try:
            if self._video_upload is None:
                self._video_upload = PublicArtifactUrlParameter(
                    node=self,
                    artifact_url_parameter=self._video_param,
                )
            public_url = await self._video_upload.aget_public_url_for_parameter()
        except Exception as e:  # noqa: BLE001 - any upload failure should degrade to a data URI
            logger.warning("%s: public-URL upload failed, sending the video inline instead: %s", self.name, e)
            return None

        # The helper passes through any http(s) URL not spelled `localhost`, so a static-server URL
        # on a loopback or LAN IP comes back unuploaded. xAI can't fetch that; the local read can.
        if not is_publicly_reachable_url(public_url):
            return None
        return public_url

    async def _prepare_video_url(self, video_input: Any) -> str:
        """Return a URL xAI can read the video from, raising a user-facing error if it can't be sent.

        These errors are raised during execution, where the engine already prefixes the message
        with the node name several times, so they leave the name out.
        """
        video_value = coerce_media_url_or_data_uri(video_input, kind="video")
        if not video_value:
            msg = "Video input has no usable value."
            raise ValueError(msg)

        public_url = await self._upload_video()
        if public_url:
            return public_url

        if video_value.startswith("data:"):
            return video_value

        try:
            return await File(video_value).aread_data_uri(fallback_mime="video/mp4")
        except FileLoadError as e:
            msg = f"Could not load video from '{video_value}': {e}"
            raise ValueError(msg) from e

    def _get_api_model_id(self) -> str:
        return f"{self._get_selected_model_id()}:edit"

    def _get_payload_model_id(self) -> str:
        return self._get_selected_model_id() or "grok-imagine-video"

    def validate_before_node_run(self) -> list[Exception] | None:
        exceptions = super().validate_before_node_run() or []

        prompt = (self.get_parameter_value("prompt") or "").strip()
        if not prompt:
            exceptions.append(ValueError("A prompt is required for video editing. Enter one in 'prompt'."))

        video_value = self.get_parameter_value("video")
        if not self._has_media_value(video_value):
            exceptions.append(ValueError("A video is required for editing. Connect one to 'Video'."))
        else:
            exceptions.extend(self._find_input_limit_errors(video_value))

        return exceptions if exceptions else None

    async def _build_payload(self) -> dict[str, Any]:
        prompt = (self.get_parameter_value("prompt") or "").strip()
        api_model_id = self._get_payload_model_id()
        video_url = await self._prepare_video_url(self.get_parameter_value("video"))

        return {
            "model": api_model_id,
            "prompt": prompt,
            "video": {"url": video_url},
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
        self.parameter_output_values["video_url"] = None
