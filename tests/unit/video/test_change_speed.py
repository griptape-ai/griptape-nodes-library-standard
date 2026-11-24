"""Tests for ChangeSpeed's audio tempo filter."""

from __future__ import annotations

import math

import pytest

from griptape_nodes_library.video.change_speed import ChangeSpeed


def _atempo_factors(audio_filter: str) -> list[float]:
    return [float(part.removeprefix("atempo=")) for part in audio_filter.split(",")]


@pytest.mark.parametrize("speed", [0.1, 0.25, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 10.0])
def test_the_audio_tempo_matches_the_speed(speed: float) -> None:
    node = ChangeSpeed(name="test_change_speed")

    factors = _atempo_factors(node._build_atempo_filter(speed))

    assert math.isclose(math.prod(factors), speed)
    assert all(ChangeSpeed.ATEMPO_MIN_SPEED <= factor <= ChangeSpeed.ATEMPO_MAX_SPEED for factor in factors)


def test_the_ffmpeg_command_carries_the_chained_audio_filter() -> None:
    node = ChangeSpeed(name="test_change_speed")

    cmd = node._build_ffmpeg_command("in.mp4", "out.mp4", input_frame_rate=24.0, speed=10.0, include_audio=True)

    audio_filter = cmd[cmd.index("-af") + 1]
    assert math.isclose(math.prod(_atempo_factors(audio_filter)), 10.0)
