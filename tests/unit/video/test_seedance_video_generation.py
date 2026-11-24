from __future__ import annotations

import pytest

from griptape_nodes_library.video.seedance_video_generation import SeedanceVideoGeneration


@pytest.mark.asyncio
async def test_build_payload_sends_settings_as_fields_not_prompt_flags() -> None:
    node = SeedanceVideoGeneration(name="Seedance")
    node.set_parameter_value("model_id", "seedance-1-5-pro-251215")
    node.set_parameter_value("prompt", "A fox runs through a forest")
    node.set_parameter_value("resolution", "1080p")
    node.set_parameter_value("ratio", "16:9")
    node.set_parameter_value("duration", 8)
    node.set_parameter_value("camerafixed", True)
    node.set_parameter_value("generate_audio", True)

    payload = await node._build_payload()

    assert payload["content"] == [{"type": "text", "text": "A fox runs through a forest"}]
    assert payload["resolution"] == "1080p"
    assert payload["ratio"] == "16:9"
    assert payload["duration"] == 8  # noqa: PLR2004
    assert payload["camera_fixed"] is True
    assert "camerafixed" not in payload
    assert payload["generate_audio"] is True
