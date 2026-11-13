"""Tests that ``WorldLabsWorldGeneration._get_media_bytes`` resolves URL-bearing
artifacts via ``File`` instead of attempting ``UrlArtifact.to_bytes()`` (which
issues an HTTP GET against ``self.value`` and fails for project macro paths
like ``{outputs}/foo.png``).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from griptape.artifacts import ImageUrlArtifact
from griptape.artifacts.video_url_artifact import VideoUrlArtifact

from griptape_nodes_library.splat.world_labs_world_generation import WorldLabsWorldGeneration

CANNED_BYTES = b"BYTES-FROM-FILE"


@pytest.fixture
def node(monkeypatch: pytest.MonkeyPatch) -> WorldLabsWorldGeneration:
    instance = WorldLabsWorldGeneration.__new__(WorldLabsWorldGeneration)
    monkeypatch.setattr(instance, "_log", lambda *_args, **_kwargs: None, raising=False)
    return instance


@pytest.fixture
def captured_paths(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    paths: list[str] = []

    async def fake_string_to_bytes(self: WorldLabsWorldGeneration, value: str) -> bytes:
        paths.append(value)
        return CANNED_BYTES

    monkeypatch.setattr(WorldLabsWorldGeneration, "_string_to_bytes", fake_string_to_bytes)
    return paths


@pytest.mark.asyncio
async def test_image_url_artifact_with_macro_path_routes_through_file(
    node: WorldLabsWorldGeneration, captured_paths: list[str]
) -> None:
    artifact = ImageUrlArtifact("{outputs}/foo.png")

    result = await node._get_media_bytes(artifact)

    assert result == CANNED_BYTES
    assert captured_paths == ["{outputs}/foo.png"]


@pytest.mark.asyncio
async def test_image_url_artifact_with_filesystem_path_routes_through_file(
    node: WorldLabsWorldGeneration, captured_paths: list[str]
) -> None:
    artifact = ImageUrlArtifact("/abs/path/foo.png")

    result = await node._get_media_bytes(artifact)

    assert result == CANNED_BYTES
    assert captured_paths == ["/abs/path/foo.png"]


@pytest.mark.asyncio
async def test_image_url_artifact_with_http_url_routes_through_file(
    node: WorldLabsWorldGeneration, captured_paths: list[str]
) -> None:
    artifact = ImageUrlArtifact("https://example.com/foo.png")

    result = await node._get_media_bytes(artifact)

    assert result == CANNED_BYTES
    assert captured_paths == ["https://example.com/foo.png"]


@pytest.mark.asyncio
async def test_plain_string_input_routes_through_file(
    node: WorldLabsWorldGeneration, captured_paths: list[str]
) -> None:
    result = await node._get_media_bytes("{outputs}/foo.png")
    assert result == CANNED_BYTES
    assert captured_paths == ["{outputs}/foo.png"]


@pytest.mark.asyncio
async def test_raw_bytes_artifact_uses_to_bytes_directly(
    node: WorldLabsWorldGeneration, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Raw-bytes artifacts (no string ``.value``) must use ``to_bytes()``; we
    must NOT route them through ``File`` because their value is bytes, not a
    path/URL.
    """

    async def should_not_be_called(self: WorldLabsWorldGeneration, value: str) -> bytes:
        raise AssertionError("_string_to_bytes should not be called for raw-bytes artifacts")

    monkeypatch.setattr(WorldLabsWorldGeneration, "_string_to_bytes", should_not_be_called)

    artifact = SimpleNamespace(value=b"\x89PNG", to_bytes=lambda: b"RAW-BYTES")

    result = await node._get_media_bytes(artifact)
    assert result == b"RAW-BYTES"


@pytest.mark.asyncio
async def test_falsy_input_returns_none(node: WorldLabsWorldGeneration) -> None:
    assert await node._get_media_bytes(None) is None
    assert await node._get_media_bytes("") is None


@pytest.mark.asyncio
async def test_empty_string_value_artifact_returns_none(
    node: WorldLabsWorldGeneration, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An artifact whose ``.value`` is an empty string and which lacks
    ``to_bytes`` should return None rather than asking File to resolve nothing.
    """

    async def should_not_be_called(self: WorldLabsWorldGeneration, value: str) -> bytes:
        raise AssertionError("_string_to_bytes should not be called for empty values")

    monkeypatch.setattr(WorldLabsWorldGeneration, "_string_to_bytes", should_not_be_called)

    artifact = SimpleNamespace(value="")
    assert await node._get_media_bytes(artifact) is None


@pytest.mark.asyncio
async def test_unknown_input_type_returns_none(node: WorldLabsWorldGeneration) -> None:
    """An object with neither a string ``.value`` nor ``to_bytes`` should not
    crash; it should just return None.
    """

    class _Bare:
        pass

    assert await node._get_media_bytes(_Bare()) is None


@pytest.mark.asyncio
async def test_parse_result_lets_exceptions_reach_the_base_handler(
    node: WorldLabsWorldGeneration, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def boom(_world: dict, _generation_id: str) -> None:
        msg = "unexpected result shape"
        raise KeyError(msg)

    monkeypatch.setattr(node, "_handle_success", boom, raising=False)
    with pytest.raises(KeyError, match="unexpected result shape"):
        await node._parse_result({}, "gen-1")


def _editor_dict(artifact_type: str, value: str) -> dict:
    """The shape the editor's file picker sets on a media parameter."""
    return {"type": artifact_type, "value": value, "width": 512, "height": 256, "name": "upload.jpg", "meta": {}}


def test_image_set_inline_is_stored_as_an_artifact() -> None:
    node = WorldLabsWorldGeneration(name="world")

    node.set_parameter_value("image", _editor_dict("ImageUrlArtifact", "https://example.com/upload.jpg"))

    image = node.get_parameter_value("image")
    assert isinstance(image, ImageUrlArtifact)
    assert image.value == "https://example.com/upload.jpg"


def test_video_set_inline_is_stored_as_an_artifact() -> None:
    node = WorldLabsWorldGeneration(name="world")

    node.set_parameter_value("video", _editor_dict("VideoUrlArtifact", "https://example.com/upload.mp4"))

    video = node.get_parameter_value("video")
    assert isinstance(video, VideoUrlArtifact)
    assert video.value == "https://example.com/upload.mp4"


def test_images_set_inline_are_stored_as_artifacts() -> None:
    node = WorldLabsWorldGeneration(name="world")

    node.set_parameter_value("images", [_editor_dict("ImageUrlArtifact", "https://example.com/upload.jpg")])

    images = node.get_parameter_value("images")
    assert [type(image) for image in images] == [ImageUrlArtifact]
    assert images[0].value == "https://example.com/upload.jpg"
