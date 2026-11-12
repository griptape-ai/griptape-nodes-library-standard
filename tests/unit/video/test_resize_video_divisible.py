"""Tests for ResizeVideo's divisible by mode, run through FFmpeg."""

from __future__ import annotations

import itertools
import re
import subprocess

import pytest
from static_ffmpeg import run  # type: ignore[import-untyped]

from griptape_nodes_library.utils.divisible_size import divisible_dimensions
from griptape_nodes_library.video.resize_video import ResizeSettings, ResizeVideo


def _settings(divisor: int, rounding: str, *, preserve_aspect_ratio: bool, anchor_dimension: str) -> ResizeSettings:
    return ResizeSettings(
        resize_mode=ResizeVideo.RESIZE_MODE_DIVISIBLE,
        percentage=100,
        target_size=1000,
        target_width=1000,
        target_height=1000,
        fit_mode=ResizeVideo.FIT_MODE_FIT,
        background_color="#000000",
        scaling_algorithm="bicubic",
        lanczos_parameter=3.0,
        divisor=divisor,
        rounding=rounding,
        preserve_aspect_ratio=preserve_aspect_ratio,
        anchor_dimension=anchor_dimension,
    )


def _ffmpeg_output_size(ffmpeg_path: str, width: int, height: int, scale_expr: str) -> tuple[int, int]:
    cmd = [
        ffmpeg_path,
        "-f",
        "lavfi",
        "-i",
        f"color=s={width}x{height}:d=0.04",
        "-vf",
        scale_expr,
        "-frames:v",
        "1",
        "-f",
        "null",
        "-",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=True)  # noqa: S603
    match = re.search(r"Video: .*?, (\d+)x(\d+)", result.stderr.split("Output #0")[-1])
    assert match is not None, result.stderr
    return int(match[1]), int(match[2])


@pytest.mark.parametrize(
    ("size", "rounding", "preserve_aspect_ratio", "anchor_dimension"),
    list(
        itertools.product(
            [(4970, 3726), (1080, 1920), (96, 160)],
            ["round", "floor", "ceil"],
            [True, False],
            ["auto", "width", "height"],
        )
    ),
)
def test_ffmpeg_matches_the_shared_rounding(
    size: tuple[int, int], rounding: str, *, preserve_aspect_ratio: bool, anchor_dimension: str
) -> None:
    ffmpeg_path, _ = run.get_or_fetch_platform_executables_else_raise()
    node = ResizeVideo(name="test_resize_video")
    settings = _settings(64, rounding, preserve_aspect_ratio=preserve_aspect_ratio, anchor_dimension=anchor_dimension)

    actual = _ffmpeg_output_size(ffmpeg_path, *size, node._build_scale_expression(settings))

    expected = divisible_dimensions(
        *size, 64, rounding, preserve_aspect_ratio=preserve_aspect_ratio, anchor_dimension=anchor_dimension
    )
    assert actual == expected


@pytest.mark.parametrize("divisor", [3, 127])
def test_odd_divisor_is_rejected(divisor: int) -> None:
    node = ResizeVideo(name="test_resize_video")
    node.set_parameter_value("resize_mode", ResizeVideo.RESIZE_MODE_DIVISIBLE)
    node.set_parameter_value("divisor", divisor)

    exceptions = node.validate_before_node_run() or []

    assert any("'divisor'" in str(e) for e in exceptions)


def test_divisor_is_clamped_to_its_range() -> None:
    node = ResizeVideo(name="test_resize_video")

    node.set_parameter_value("divisor", 1)
    assert node.get_parameter_value("divisor") == ResizeVideo.MIN_DIVISOR
    node.set_parameter_value("divisor", 4096)
    assert node.get_parameter_value("divisor") == ResizeVideo.MAX_DIVISOR
