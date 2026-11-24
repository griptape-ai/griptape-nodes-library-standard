"""Tests for ffmpeg_utils.seconds_to_ts."""

import pytest

from griptape_nodes_library.utils.ffmpeg_utils import seconds_to_ts


@pytest.mark.parametrize(
    ("sec", "expected"),
    [
        (0, "00:00:00.000"),
        (-1.5, "00:00:00.000"),
        (1.5, "00:00:01.500"),
        (1.9994, "00:00:01.999"),
        (1.9996, "00:00:02.000"),
        (59.9996, "00:01:00.000"),
        (3599.9996, "01:00:00.000"),
        (3725.25, "01:02:05.250"),
    ],
)
def test_seconds_to_ts(sec: float, expected: str) -> None:
    assert seconds_to_ts(sec) == expected
