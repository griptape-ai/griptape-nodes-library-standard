from __future__ import annotations

import re

import pytest

from griptape_nodes_library.video.seedance_video_generation import SeedanceVideoGeneration


def _node(model_id: str, resolution: str, duration: int) -> SeedanceVideoGeneration:
    node = SeedanceVideoGeneration(name="Seedance")
    node.set_parameter_value("model_id", model_id)
    node.set_parameter_value("prompt", "A fox runs through a forest")
    node.set_parameter_value("resolution", resolution)
    node.set_parameter_value("duration", duration)
    return node


def _value_error_messages(node: SeedanceVideoGeneration) -> list[str]:
    return [str(e) for e in node.validate_before_node_run() or [] if isinstance(e, ValueError)]


@pytest.mark.parametrize(
    ("model_id", "resolution", "duration", "expected"),
    [
        ("seedance-1-0-pro-250528", "720p", 7, "duration 5, 10 seconds, got 7"),
        ("seedance-1-0-pro-fast-251015", "1080p", 12, "duration 5, 10 seconds, got 12"),
        ("seedance-1-5-pro-251215", "1080p", 12, "resolution 480p, 720p, got '1080p'"),
    ],
)
def test_validate_rejects_unsupported_model_combinations(
    model_id: str, resolution: str, duration: int, expected: str
) -> None:
    messages = _value_error_messages(_node(model_id, resolution, duration))

    assert any(re.search(re.escape(expected), message) for message in messages), messages


@pytest.mark.parametrize(
    ("model_id", "resolution", "duration"),
    [
        ("seedance-1-0-pro-250528", "1080p", 10),
        ("seedance-1-0-pro-fast-251015", "480p", 5),
        ("seedance-1-5-pro-251215", "720p", 12),
        ("seedance-1-5-pro-251215", "480p", 4),
    ],
)
def test_validate_accepts_supported_model_combinations(model_id: str, resolution: str, duration: int) -> None:
    assert _value_error_messages(_node(model_id, resolution, duration)) == []


@pytest.mark.asyncio
async def test_build_payload_sends_settings_as_fields_not_prompt_flags() -> None:
    node = _node("seedance-1-5-pro-251215", "720p", 8)
    node.set_parameter_value("ratio", "16:9")
    node.set_parameter_value("camerafixed", True)
    node.set_parameter_value("generate_audio", True)

    payload = await node._build_payload()

    assert payload["content"] == [{"type": "text", "text": "A fox runs through a forest"}]
    assert payload["resolution"] == "720p"
    assert payload["ratio"] == "16:9"
    assert payload["duration"] == 8  # noqa: PLR2004
    assert payload["camerafixed"] is True
    assert payload["generate_audio"] is True
