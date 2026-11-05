"""Tests for the library's media helpers that build on the engine's ``coerce_media_url_or_data_uri``.

The coercion rules themselves are tested in the engine, where the function lives.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from griptape.artifacts.video_url_artifact import VideoUrlArtifact
from griptape_nodes.files import file as file_module

from griptape_nodes_library.media import prepare_media_data_uri, resolve_media_location
from griptape_nodes_library.video.ltx_video_retake import LTXVideoRetake
from griptape_nodes_library.video.ltx_video_to_video_hdr import LTXVideoToVideoHDR

# ---------------------------------------------------------------------------
# prepare_media_data_uri
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestPrepareMediaDataUri:
    """Verifies the prepare wrapper threads coerced values into ``File()``."""

    async def test_macro_path_string_is_handed_to_file(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, str] = {}

        from griptape_nodes.files import file as file_module

        original_init = file_module.File.__init__

        def capture_init(self: file_module.File, path: str, *args: Any, **kwargs: Any) -> None:
            captured["path"] = path
            original_init(self, path, *args, **kwargs)

        async def fake_aread(self: file_module.File, fallback_mime: str = "application/octet-stream") -> str:
            return "data:video/mp4;base64,RESOLVED"

        monkeypatch.setattr(file_module.File, "__init__", capture_init)
        monkeypatch.setattr(file_module.File, "aread_data_uri", fake_aread)

        result = await prepare_media_data_uri("{inputs}/clip.mp4", kind="video")
        assert result == "data:video/mp4;base64,RESOLVED"
        assert captured["path"] == "{inputs}/clip.mp4"

    async def test_data_uri_input_is_returned_without_calling_file(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from griptape_nodes.files import file as file_module

        async def fail_aread(self: file_module.File, fallback_mime: str = "application/octet-stream") -> str:
            raise AssertionError("File.aread_data_uri must not be called for data: URIs")

        monkeypatch.setattr(file_module.File, "aread_data_uri", fail_aread)

        result = await prepare_media_data_uri("data:video/mp4;base64,ALREADY_BASE64", kind="video")
        assert result == "data:video/mp4;base64,ALREADY_BASE64"

    async def test_returns_none_for_falsy_input(self) -> None:
        assert await prepare_media_data_uri(None, kind="video") is None
        assert await prepare_media_data_uri("", kind="video") is None
        assert await prepare_media_data_uri(0, kind="image") is None

    async def test_returns_none_when_helper_resolves_to_none(self) -> None:
        assert await prepare_media_data_uri(SimpleNamespace(), kind="video") is None

    async def test_file_load_error_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from griptape_nodes.files import file as file_module
        from griptape_nodes.retained_mode.events.os_events import FileIOFailureReason

        async def fail_aread(self: file_module.File, fallback_mime: str = "application/octet-stream") -> str:
            raise file_module.FileLoadError(FileIOFailureReason.FILE_NOT_FOUND, "nope")

        monkeypatch.setattr(file_module.File, "aread_data_uri", fail_aread)

        assert await prepare_media_data_uri("{inputs}/missing.mp4", kind="video") is None

    async def test_fallback_mime_override_is_passed_to_file(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, str] = {}

        from griptape_nodes.files import file as file_module

        async def fake_aread(self: file_module.File, fallback_mime: str = "application/octet-stream") -> str:
            captured["fallback_mime"] = fallback_mime
            return "data:image/jpeg;base64,RESOLVED"

        monkeypatch.setattr(file_module.File, "aread_data_uri", fake_aread)

        await prepare_media_data_uri("{inputs}/foo.jpg", kind="image", fallback_mime="image/jpeg")
        assert captured["fallback_mime"] == "image/jpeg"

    async def test_default_fallback_mime_matches_kind(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, str] = {}

        from griptape_nodes.files import file as file_module

        async def fake_aread(self: file_module.File, fallback_mime: str = "application/octet-stream") -> str:
            captured["fallback_mime"] = fallback_mime
            return "data:image/png;base64,RESOLVED"

        monkeypatch.setattr(file_module.File, "aread_data_uri", fake_aread)

        await prepare_media_data_uri("{inputs}/foo.png", kind="image")
        assert captured["fallback_mime"] == "image/png"


# ---------------------------------------------------------------------------
# LTX input-extractor renames
# ---------------------------------------------------------------------------


class TestLTXVideoRetakeExtractInputVideoUrl:
    """``_extract_input_video_url`` delegates to ``coerce_media_url_or_data_uri``."""

    def test_extracts_url_from_video_url_artifact(self) -> None:
        node = LTXVideoRetake.__new__(LTXVideoRetake)
        artifact = VideoUrlArtifact("https://example.com/foo.mp4")
        assert node._extract_input_video_url(artifact) == "https://example.com/foo.mp4"

    def test_extracts_macro_path_from_video_url_artifact(self) -> None:
        node = LTXVideoRetake.__new__(LTXVideoRetake)
        artifact = VideoUrlArtifact("{inputs}/foo.mp4")
        assert node._extract_input_video_url(artifact) == "{inputs}/foo.mp4"

    def test_extracts_plain_string_macro_path(self) -> None:
        node = LTXVideoRetake.__new__(LTXVideoRetake)
        assert node._extract_input_video_url("{inputs}/foo.mp4") == "{inputs}/foo.mp4"

    def test_returns_none_for_none(self) -> None:
        node = LTXVideoRetake.__new__(LTXVideoRetake)
        assert node._extract_input_video_url(None) is None

    def test_returns_none_for_empty_string(self) -> None:
        node = LTXVideoRetake.__new__(LTXVideoRetake)
        assert node._extract_input_video_url("") is None

    def test_extracts_value_from_generic_artifact_with_value_attr(self) -> None:
        node = LTXVideoRetake.__new__(LTXVideoRetake)
        artifact = SimpleNamespace(value="{inputs}/foo.mp4", base64=None)
        assert node._extract_input_video_url(artifact) == "{inputs}/foo.mp4"

    def test_wraps_raw_base64_from_video_artifact(self) -> None:
        node = LTXVideoRetake.__new__(LTXVideoRetake)
        artifact = SimpleNamespace(value=None, base64="RAW_VIDEO_B64")
        assert node._extract_input_video_url(artifact) == "data:video/mp4;base64,RAW_VIDEO_B64"


class TestLTXVideoToVideoHDRExtractInputVideoUrl:
    def test_extracts_url_from_video_url_artifact(self) -> None:
        artifact = VideoUrlArtifact("https://example.com/foo.mp4")
        assert LTXVideoToVideoHDR._extract_input_video_url(artifact) == "https://example.com/foo.mp4"

    def test_extracts_plain_string_macro_path(self) -> None:
        assert LTXVideoToVideoHDR._extract_input_video_url("{inputs}/foo.mp4") == "{inputs}/foo.mp4"

    def test_returns_none_for_none(self) -> None:
        assert LTXVideoToVideoHDR._extract_input_video_url(None) is None


# ---------------------------------------------------------------------------
# LTXVideoRetake._prepare_video_data_uri_async still works after migration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestLTXVideoRetakePrepareDataUriAsync:
    async def test_macro_path_string_is_handed_to_file(self, monkeypatch: pytest.MonkeyPatch) -> None:
        node = LTXVideoRetake.__new__(LTXVideoRetake)
        node.name = "LTXVideoRetake"
        captured: dict[str, str] = {}

        from griptape_nodes.files import file as file_module

        original_init = file_module.File.__init__

        def capture_init(self: file_module.File, path: str, *args: Any, **kwargs: Any) -> None:
            captured["path"] = path
            original_init(self, path, *args, **kwargs)

        async def fake_aread(self: file_module.File, fallback_mime: str = "application/octet-stream") -> str:
            return "data:video/mp4;base64,RESOLVED"

        monkeypatch.setattr(file_module.File, "__init__", capture_init)
        monkeypatch.setattr(file_module.File, "aread_data_uri", fake_aread)

        result = await node._prepare_video_data_uri_async("{inputs}/clip.mp4")
        assert result == "data:video/mp4;base64,RESOLVED"
        assert captured["path"] == "{inputs}/clip.mp4"

    async def test_data_uri_input_is_returned_without_calling_file(self, monkeypatch: pytest.MonkeyPatch) -> None:
        node = LTXVideoRetake.__new__(LTXVideoRetake)
        node.name = "LTXVideoRetake"

        from griptape_nodes.files import file as file_module

        async def fail_aread(self: file_module.File, fallback_mime: str = "application/octet-stream") -> str:
            raise AssertionError("File.aread_data_uri must not be called for data: URIs")

        monkeypatch.setattr(file_module.File, "aread_data_uri", fail_aread)

        result = await node._prepare_video_data_uri_async("data:video/mp4;base64,ALREADY_BASE64")
        assert result == "data:video/mp4;base64,ALREADY_BASE64"


# ---------------------------------------------------------------------------
# resolve_media_location
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "location",
    ["data:image/png;base64,iVBORw0KGgo=", "data:video/mp4;base64,AAAA", "https://example.com/clip.mp4"],
    ids=["image_data_uri", "video_data_uri", "remote_url"],
)
def test_resolve_media_location_passes_uris_through(location: str) -> None:
    """Real ``File.resolve()`` anchors a data URI under the workspace, so it must be skipped."""
    assert resolve_media_location(location) == location


def test_resolve_media_location_resolves_paths_through_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(file_module.File, "resolve", lambda _self: "/project/outputs/clip.mp4")

    assert resolve_media_location("{outputs}/clip.mp4") == "/project/outputs/clip.mp4"
