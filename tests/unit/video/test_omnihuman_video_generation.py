from __future__ import annotations

from typing import Any

import pytest

from griptape_nodes_library.video.omnihuman_video_generation import OmnihumanVideoGeneration


def _node(monkeypatch: pytest.MonkeyPatch, model_id: str, **values: Any) -> OmnihumanVideoGeneration:
    node = OmnihumanVideoGeneration(name="OmniHuman")
    node.set_parameter_value("image_url", "https://public.example/source.jpg")
    node.set_parameter_value("audio_url", "https://public.example/audio.mp3")
    node.set_parameter_value("auto_detect_masks", False)
    for name, value in values.items():
        node.set_parameter_value(name, value)
    node.set_parameter_value("model_id", model_id)

    async def public_image_url() -> str:
        return "https://public.example/source.jpg"

    async def public_audio_url() -> str:
        return "https://public.example/audio.mp3"

    monkeypatch.setattr(node, "_resolve_public_image_url", public_image_url)
    monkeypatch.setattr(node._public_audio_url_parameter, "aget_public_url_for_parameter", public_audio_url)
    return node


@pytest.mark.asyncio
async def test_omnihuman_1_0_leaves_out_hidden_parameters(monkeypatch: pytest.MonkeyPatch) -> None:
    node = _node(monkeypatch, "omnihuman-1-0", prompt="dance", seed=42, fast_mode=True)

    payload = await node._build_payload()

    assert payload["req_key"] == "realman_avatar_picture_omni_cv"
    assert "prompt" not in payload
    assert "seed" not in payload
    assert "fast_mode" not in payload


@pytest.mark.asyncio
async def test_omnihuman_1_5_leaves_out_random_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    node = _node(monkeypatch, "omnihuman-1-5")

    payload = await node._build_payload()

    assert "seed" not in payload


@pytest.mark.asyncio
async def test_omnihuman_1_5_sends_set_parameters(monkeypatch: pytest.MonkeyPatch) -> None:
    node = _node(monkeypatch, "omnihuman-1-5", prompt="dance", seed=0, fast_mode=True)

    payload = await node._build_payload()

    assert payload["req_key"] == "realman_avatar_picture_omni15_cv"
    assert payload["prompt"] == "dance"
    assert payload["seed"] == 0
    assert payload["fast_mode"] is True
