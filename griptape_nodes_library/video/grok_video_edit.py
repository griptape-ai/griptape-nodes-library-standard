from __future__ import annotations

import logging
from pathlib import PureWindowsPath
from typing import Any, ClassVar
from urllib.parse import unquote, urlsplit

from griptape.artifacts.video_url_artifact import VideoUrlArtifact
from griptape_nodes.exe_types.core_types import NodeError, NodeErrorLink, ParameterMode
from griptape_nodes.exe_types.param_components.artifact_url.public_artifact_url_parameter import (
    PublicArtifactUrlParameter,
)
from griptape_nodes.exe_types.param_components.model_access_component import ModelAccessComponent
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_types.parameter_dict import ParameterDict
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.exe_types.param_types.parameter_video import ParameterVideo
from griptape_nodes.exe_types.validators import (
    LIMITS_LINK_LABEL,
    AllowedFormats,
    MaxDuration,
    MediaInfo,
    MediaNeed,
    MediaValidator,
)
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
MP4_MIME_TYPE = "video/mp4"
MP4_EXTENSION = ".mp4"
_IS_MP4 = AllowedFormats(MP4_MIME_TYPE)


class Mp4FileName(MediaValidator):
    """xAI's rule that a video's name ends in .mp4, or that a data URI is typed video/mp4.

    xAI fails a job whose video URL ends in another extension, even for an MP4 saved as .mov, and
    gives no reason. The file's name normally reaches xAI intact: the uploaded URL ends in it, and
    an inline video is typed from it. A name with no extension, such as a signed URL, passes.

    The exception is a local name containing '#' or '?': the engine's upload currently cuts the
    name at that character, so 'Take #2.mp4' passes here but is uploaded as 'Take '
    (griptape-ai/griptape-nodes-engine#6048).
    """

    needs = frozenset({MediaNeed.VIDEO})

    def check_media(
        self,
        value: Any,  # noqa: ARG002
        media: MediaInfo,
        *,
        subject: str,
        docs_url: str | None,
        position: int | None = None,  # noqa: ARG002
    ) -> NodeError | None:
        location = media.location
        if location is None:
            return None
        # A file read as another format already gets AllowedFormats' "Convert it to MP4", and
        # converting it gives it an .mp4 name too, so this stays quiet rather than repeat it.
        if media.video is not None and _IS_MP4.check_media(None, media, subject=subject, docs_url=None) is not None:
            return None
        if location.startswith("data:"):
            mime_type = location.removeprefix("data:").split(",", 1)[0].split(";", 1)[0].strip().lower()
            if mime_type == MP4_MIME_TYPE:
                return None
            return _limit_error(
                f"{subject} is a data URI that isn't typed {MP4_MIME_TYPE}, which xAI requires. Convert it to MP4.",
                docs_url,
            )
        name = _file_name(location)
        if PureWindowsPath(name).suffix.lower() in ("", MP4_EXTENSION):
            return None
        return _limit_error(
            f"{subject} is named '{name}', and xAI only accepts a file whose name ends in {MP4_EXTENSION}. "
            f"If it's already an MP4, rename it to end in {MP4_EXTENSION}; otherwise convert it to MP4.",
            docs_url,
        )


def _file_name(location: str) -> str:
    """The last part of a path or URL, without a URL's query, such as a signed URL's signature."""
    path = unquote(urlsplit(location).path) if is_url(location) else location
    return PureWindowsPath(path.rstrip("/\\")).name


def _limit_error(message: str, docs_url: str | None) -> NodeError:
    """An input-limit error linking xAI's docs, as the engine's validators word and link theirs."""
    if not docs_url:
        return NodeError(message)
    return NodeError(message, links=[NodeErrorLink(label=LIMITS_LINK_LABEL, url=docs_url)])


# The format and duration are read from the file itself, so a MOV renamed to .mp4 is still refused,
# and the name is checked too, so an MP4 saved as .mov is refused before xAI fails it. Only a file
# on this machine is read: a remote URL or a data URI has only its name or type checked here.
VIDEO_LIMITS = (
    AllowedFormats(MP4_MIME_TYPE),
    Mp4FileName(),
    MaxDuration(MAX_INPUT_DURATION_SECONDS),
)


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
            pre_run_validators=VIDEO_LIMITS,
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
        request body. The helper is built here rather than in __init__ because it raises without a
        Griptape Cloud credential, which should cost the upload, not the node.
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

        # xAI fetches a public URL itself, so it goes as is, without the upload helper, which
        # can't be built without a Griptape Cloud credential.
        if is_publicly_reachable_url(video_value):
            return video_value

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

        return exceptions if exceptions else None

    def input_limits_url(self) -> str | None:
        return XAI_VIDEO_EDIT_DOCS_URL

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
