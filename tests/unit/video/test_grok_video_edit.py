from __future__ import annotations

from typing import Any

import pytest
from griptape.artifacts.video_url_artifact import VideoUrlArtifact
from griptape_nodes.exe_types.core_types import NodeError
from griptape_nodes.exe_types.param_components.artifact_url import (
    public_artifact_url_parameter as public_artifact_url_parameter_module,
)
from griptape_nodes.exe_types.param_components.artifact_url.public_artifact_url_parameter import (
    PublicArtifactUrlParameter,
)
from griptape_nodes.files import file as file_module
from griptape_nodes.files.file import FileLoadError
from griptape_nodes.retained_mode.events.os_events import FileIOFailureReason

from griptape_nodes_library.proxy import GriptapeProxyNode
from griptape_nodes_library.video import grok_video_edit as grok_video_edit_module
from griptape_nodes_library.video.grok_video_edit import XAI_VIDEO_EDIT_DOCS_URL, GrokVideoEdit

RESOLVED = "data:video/mp4;base64,RESOLVED"
UPLOADED = "https://cloud.griptape.ai/api/buckets/b/signed/clip.mp4"


def _node() -> GrokVideoEdit:
    """A bare node whose upload is unavailable, so the video goes inline as a data URI."""
    node = GrokVideoEdit.__new__(GrokVideoEdit)
    node.name = "GrokVideoEdit"

    async def no_upload() -> None:
        return None

    node._upload_video = no_upload  # type: ignore[method-assign]
    return node


@pytest.fixture
def captured_paths(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    paths: list[str] = []
    original_init = file_module.File.__init__

    def capture_init(self: file_module.File, path: str, *args: Any, **kwargs: Any) -> None:
        paths.append(path)
        original_init(self, path, *args, **kwargs)

    async def fake_aread(self: file_module.File, fallback_mime: str = "application/octet-stream") -> str:  # noqa: ARG001
        return RESOLVED

    monkeypatch.setattr(file_module.File, "__init__", capture_init)
    monkeypatch.setattr(file_module.File, "aread_data_uri", fake_aread)
    return paths


@pytest.mark.asyncio
class TestPrepareVideoUrlFallback:
    @pytest.mark.parametrize(
        "video_input",
        [
            "{outputs}/clip.mp4",
            VideoUrlArtifact(value="{outputs}/clip.mp4"),
            {"type": "VideoUrlArtifact", "value": "{outputs}/clip.mp4"},
        ],
        ids=["string", "artifact", "serialized-dict"],
    )
    async def test_macro_path_is_handed_to_file(self, video_input: Any, captured_paths: list[str]) -> None:
        assert await _node()._prepare_video_url(video_input) == RESOLVED
        assert captured_paths == ["{outputs}/clip.mp4"]

    async def test_data_uri_passes_through(self, captured_paths: list[str]) -> None:
        assert await _node()._prepare_video_url("data:video/mp4;base64,ABC") == "data:video/mp4;base64,ABC"
        assert captured_paths == []

    async def test_load_failure_raises_with_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def failing_aread(self: file_module.File, fallback_mime: str = "application/octet-stream") -> str:  # noqa: ARG001
            raise FileLoadError(FileIOFailureReason.FILE_NOT_FOUND, "file not found")

        monkeypatch.setattr(file_module.File, "aread_data_uri", failing_aread)

        with pytest.raises(ValueError, match=r"^Could not load video from '\{outputs\}/missing\.mp4'"):
            await _node()._prepare_video_url("{outputs}/missing.mp4")

    async def test_empty_input_raises(self) -> None:
        with pytest.raises(ValueError, match="no usable value"):
            await _node()._prepare_video_url(VideoUrlArtifact(value=""))


def _stub_duration(monkeypatch: pytest.MonkeyPatch, duration: float | None) -> None:
    monkeypatch.setattr(grok_video_edit_module, "local_video_duration", lambda _value: duration)


def _validated_node(monkeypatch: pytest.MonkeyPatch, video: Any, duration: float | None = 5.0) -> GrokVideoEdit:
    """A real node with a prompt and a video set, whose duration probe reports `duration`."""
    _stub_duration(monkeypatch, duration)
    node = GrokVideoEdit(name="GrokVideoEdit")
    node.set_parameter_value("prompt", "make it snow")
    node.set_parameter_value("video", video)
    return node


def _messages(node: GrokVideoEdit) -> list[str]:
    return [str(error) for error in node.validate_before_node_run() or []]


class TestInputLimits:
    @pytest.mark.parametrize(
        ("video_input", "format_name"),
        [
            ("{inputs}/videos/IMG_2689.mov", "MOV"),
            ("data:video/quicktime;base64,ABC", "MOV"),
            ("https://example.com/clip.webm?sig=1", "WEBM"),
        ],
        ids=["mov-path", "quicktime-data-uri", "webm-url"],
    )
    def test_non_mp4_video_fails_validation(
        self, monkeypatch: pytest.MonkeyPatch, video_input: str, format_name: str
    ) -> None:
        assert _messages(_validated_node(monkeypatch, video_input)) == [
            f"The video in 'Video' is a {format_name} file. Convert it to MP4."
        ]

    @pytest.mark.parametrize("video_input", ["{inputs}/Take #2.mp4", "/Users/me/Shots/edit.v3?final.mp4"])
    def test_local_mp4_with_url_characters_in_its_name_passes_validation(
        self, monkeypatch: pytest.MonkeyPatch, video_input: str
    ) -> None:
        assert _messages(_validated_node(monkeypatch, video_input)) == []

    def test_local_mov_with_a_hash_in_its_name_fails_validation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert _messages(_validated_node(monkeypatch, "{inputs}/Take #2.mov")) == [
            "The video in 'Video' is a MOV file. Convert it to MP4."
        ]

    def test_url_without_extension_passes_validation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert _messages(_validated_node(monkeypatch, "https://example.com/signed/abc123")) == []

    def test_over_long_video_fails_validation_with_duration(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert _messages(_validated_node(monkeypatch, "{inputs}/clip.mp4", duration=8.798)) == [
            "The video in 'Video' is 8.80 seconds long and the limit is 8.7. Use a Trim Video node to shorten it."
        ]

    def test_limit_errors_link_xai_docs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        errors = _validated_node(monkeypatch, "{inputs}/clip.mp4", duration=12.0).validate_before_node_run() or []

        assert len(errors) == 1
        assert isinstance(errors[0], NodeError)
        assert [link.url for link in errors[0].links or []] == [XAI_VIDEO_EDIT_DOCS_URL]

    def test_wrong_container_and_too_long_are_reported_together(self, monkeypatch: pytest.MonkeyPatch) -> None:
        messages = _messages(_validated_node(monkeypatch, "{inputs}/IMG_2689.mov", duration=8.798))

        assert len(messages) == 2
        assert "is a MOV file" in messages[0]
        assert "is 8.80 seconds long" in messages[1]

    def test_video_at_limit_passes_validation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert _messages(_validated_node(monkeypatch, "{inputs}/clip.mp4", duration=8.7)) == []

    def test_unknown_duration_passes_validation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert _messages(_validated_node(monkeypatch, "{inputs}/clip.mp4", duration=None)) == []

    def test_serialized_artifact_is_checked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        video = {"type": "VideoUrlArtifact", "value": "{inputs}/clip.mp4"}

        assert len(_messages(_validated_node(monkeypatch, video, duration=12.0))) == 1

    def test_missing_video_reports_only_that(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert _messages(_validated_node(monkeypatch, "")) == [
            "A video is required for editing. Connect one to 'Video'."
        ]


def _uploading_node(video: str = "{inputs}/clip.mp4") -> GrokVideoEdit:
    """A real node with a video set."""
    node = GrokVideoEdit(name="GrokVideoEdit")
    node.set_parameter_value("video", video)
    return node


@pytest.mark.asyncio
class TestUpload:
    async def test_local_video_is_sent_as_uploaded_url(
        self, monkeypatch: pytest.MonkeyPatch, captured_paths: list[str]
    ) -> None:
        monkeypatch.setattr(PublicArtifactUrlParameter, "get_public_url_for_parameter", lambda _self: UPLOADED)
        node = _uploading_node()

        assert await node._prepare_video_url(node.get_parameter_value("video")) == UPLOADED
        assert captured_paths == []

    async def test_payload_carries_the_uploaded_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(PublicArtifactUrlParameter, "get_public_url_for_parameter", lambda _self: UPLOADED)
        node = _uploading_node()
        node.set_parameter_value("prompt", "make it snow")

        assert await node._build_payload() == {
            "model": "grok-imagine-video",
            "prompt": "make it snow",
            "video": {"url": UPLOADED},
        }

    async def test_failed_upload_falls_back_to_data_uri(
        self, monkeypatch: pytest.MonkeyPatch, captured_paths: list[str]
    ) -> None:
        def failing_upload(_self: PublicArtifactUrlParameter) -> str:
            msg = "No Griptape Cloud storage buckets found!"
            raise RuntimeError(msg)

        monkeypatch.setattr(PublicArtifactUrlParameter, "get_public_url_for_parameter", failing_upload)
        node = _uploading_node()

        assert await node._prepare_video_url(node.get_parameter_value("video")) == RESOLVED
        assert captured_paths == ["{inputs}/clip.mp4"]

    async def test_missing_cloud_credential_falls_back_to_data_uri(
        self, monkeypatch: pytest.MonkeyPatch, captured_paths: list[str]
    ) -> None:
        monkeypatch.setattr(
            public_artifact_url_parameter_module, "resolve_cloud_credential", lambda *_args, **_kwargs: None
        )
        node = _uploading_node()

        assert await node._prepare_video_url(node.get_parameter_value("video")) == RESOLVED
        assert captured_paths == ["{inputs}/clip.mp4"]

    async def test_unreachable_passthrough_falls_back_to_data_uri(
        self, monkeypatch: pytest.MonkeyPatch, captured_paths: list[str]
    ) -> None:
        static_url = "http://127.0.0.1:8124/workspace/clip.mp4"
        node = _uploading_node(video=static_url)

        assert await node._prepare_video_url(node.get_parameter_value("video")) == RESOLVED
        assert captured_paths == [static_url]

    async def test_public_url_is_passed_through(
        self, monkeypatch: pytest.MonkeyPatch, captured_paths: list[str]
    ) -> None:
        public_url = "https://example.com/clip.mp4"
        node = _uploading_node(video=public_url)

        assert await node._prepare_video_url(node.get_parameter_value("video")) == public_url
        assert captured_paths == []

    @pytest.mark.parametrize("run_fails", [False, True], ids=["success", "failure"])
    async def test_upload_is_deleted_after_the_run(self, monkeypatch: pytest.MonkeyPatch, *, run_fails: bool) -> None:
        deleted: list[bool] = []
        monkeypatch.setattr(PublicArtifactUrlParameter, "get_public_url_for_parameter", lambda _self: UPLOADED)
        monkeypatch.setattr(PublicArtifactUrlParameter, "delete_uploaded_artifact", lambda _self: deleted.append(True))
        node = _uploading_node()

        async def run(_self: GriptapeProxyNode) -> None:
            await node._prepare_video_url(node.get_parameter_value("video"))
            assert deleted == []
            if run_fails:
                msg = "generation failed"
                raise RuntimeError(msg)

        monkeypatch.setattr(GriptapeProxyNode, "aprocess", run)

        if run_fails:
            with pytest.raises(RuntimeError, match="generation failed"):
                await node.aprocess()
        else:
            await node.aprocess()
        assert deleted == [True]

    @pytest.mark.parametrize("run_fails", [False, True], ids=["success", "failure"])
    async def test_failed_delete_does_not_change_the_run_outcome(
        self, monkeypatch: pytest.MonkeyPatch, *, run_fails: bool
    ) -> None:
        def failing_delete(_self: PublicArtifactUrlParameter) -> None:
            msg = "403 Forbidden"
            raise RuntimeError(msg)

        monkeypatch.setattr(PublicArtifactUrlParameter, "get_public_url_for_parameter", lambda _self: UPLOADED)
        monkeypatch.setattr(PublicArtifactUrlParameter, "delete_uploaded_artifact", failing_delete)
        node = _uploading_node()

        async def run(_self: GriptapeProxyNode) -> None:
            await node._prepare_video_url(node.get_parameter_value("video"))
            if run_fails:
                msg = "generation failed"
                raise RuntimeError(msg)

        monkeypatch.setattr(GriptapeProxyNode, "aprocess", run)

        if run_fails:
            with pytest.raises(RuntimeError, match="generation failed"):
                await node.aprocess()
        else:
            await node.aprocess()


def test_node_builds_without_cloud_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        public_artifact_url_parameter_module, "resolve_cloud_credential", lambda *_args, **_kwargs: None
    )
    assert GrokVideoEdit(name="GrokVideoEdit").get_parameter_by_name("video") is not None
