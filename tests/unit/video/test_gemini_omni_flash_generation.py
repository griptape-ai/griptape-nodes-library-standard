"""Tests for how GeminiOmniFlashGeneration builds its request from image inputs."""

from __future__ import annotations

from typing import Any

import pytest
from griptape.artifacts.image_artifact import ImageArtifact
from griptape_nodes.files.file import FileLoadError
from griptape_nodes.retained_mode.events.os_events import FileIOFailureReason

from griptape_nodes_library.video import gemini_omni_flash_generation
from griptape_nodes_library.video.gemini_omni_flash_generation import GeminiOmniFlashGeneration

GOOD = "good.png"
MISSING = "missing.png"


class _FakeFile:
    def __init__(self, location: str) -> None:
        self._location = location

    async def aread_data_uri(self, *, fallback_mime: str) -> str:  # noqa: ARG002
        if self._location == MISSING:
            msg = f"No such file: {self._location}"
            raise FileLoadError(FileIOFailureReason.FILE_NOT_FOUND, msg)
        return "data:image/png;base64,R09PRA=="


def _node(monkeypatch: pytest.MonkeyPatch, values: dict[str, Any]) -> GeminiOmniFlashGeneration:
    monkeypatch.setattr(gemini_omni_flash_generation, "File", _FakeFile)
    node = GeminiOmniFlashGeneration.__new__(GeminiOmniFlashGeneration)
    node.name = "GeminiOmniFlash"  # type: ignore[misc]
    monkeypatch.setattr(node, "get_parameter_value", lambda name: values.get(name), raising=False)
    monkeypatch.setattr(node, "get_parameter_list_value", lambda name: values.get(name) or [], raising=False)
    return node


@pytest.mark.asyncio
async def test_all_references_loaded_is_reference_to_video(monkeypatch: pytest.MonkeyPatch) -> None:
    node = _node(monkeypatch, {"prompt": "p", "reference_images": [GOOD, GOOD]})

    payload = await node._build_payload()

    assert payload["generation_config"]["video_config"]["task"] == "reference_to_video"
    assert [item["type"] for item in payload["input"]] == ["image", "image", "text"]


@pytest.mark.asyncio
@pytest.mark.parametrize("references", [[GOOD, MISSING], [MISSING]])
async def test_reference_that_fails_to_load_fails_the_build(
    monkeypatch: pytest.MonkeyPatch, references: list[str]
) -> None:
    node = _node(monkeypatch, {"prompt": "p", "reference_images": references})

    with pytest.raises(ValueError, match=f"reference image {len(references)} of {len(references)}"):
        await node._build_payload()


@pytest.mark.asyncio
async def test_legacy_image_that_fails_to_load_fails_the_build(monkeypatch: pytest.MonkeyPatch) -> None:
    node = _node(monkeypatch, {"prompt": "p", "image": MISSING})

    with pytest.raises(ValueError, match="Couldn't load image"):
        await node._build_payload()


@pytest.mark.asyncio
async def test_no_images_is_text_to_video(monkeypatch: pytest.MonkeyPatch) -> None:
    node = _node(monkeypatch, {"prompt": "p", "reference_images": [None]})

    payload = await node._build_payload()

    assert payload["generation_config"]["video_config"]["task"] == "text_to_video"
    assert payload["input"] == "p"


@pytest.mark.asyncio
async def test_ignored_legacy_image_is_not_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    node = _node(monkeypatch, {"prompt": "p", "image": MISSING, "reference_images": [GOOD]})

    payload = await node._build_payload()

    assert payload["generation_config"]["video_config"]["task"] == "reference_to_video"


@pytest.mark.asyncio
async def test_image_artifact_reference_is_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    artifact = ImageArtifact(b"\x89PNG", format="jpeg", width=1, height=1)
    node = _node(monkeypatch, {"prompt": "p", "reference_images": [artifact]})

    payload = await node._build_payload()

    assert payload["input"][0] == {"type": "image", "data": artifact.base64, "mime_type": "image/jpeg"}
