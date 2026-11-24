"""Tests for ConcatenateVideos running end to end through the save step."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from griptape.artifacts.video_url_artifact import VideoUrlArtifact
from griptape_nodes.files.file import File
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from static_ffmpeg import run  # type: ignore[import-untyped]

from griptape_nodes_library.video.concatenate_videos import ConcatenateVideos


def _make_clip(path: Path, size: str) -> Path:
    ffmpeg_path, _ = run.get_or_fetch_platform_executables_else_raise()
    cmd = [ffmpeg_path, "-y", "-f", "lavfi", "-i", f"testsrc=size={size}:rate=10:duration=1", "-pix_fmt", "yuv420p"]
    subprocess.run([*cmd, str(path)], check=True, capture_output=True)  # noqa: S603
    return path


def test_the_node_has_an_output_file_parameter() -> None:
    node = ConcatenateVideos(name="test_concatenate_videos")

    param = node.get_parameter_by_name("output_file")
    assert param is not None
    assert param.default_value == "concatenated.mp4"


def test_choosing_an_output_format_changes_the_output_file_extension() -> None:
    node = ConcatenateVideos(name="test_concatenate_videos")

    node.set_parameter_value("output_format", "webm")

    assert node.get_parameter_value("output_file") == "concatenated.webm"


@pytest.mark.usefixtures("open_workflow")
@pytest.mark.parametrize("second_size", ["64x48", "80x48"])
def test_two_clips_are_joined_and_saved(tmp_path: Path, second_size: str) -> None:
    GriptapeNodes.ConfigManager().workspace_path = tmp_path
    first = _make_clip(tmp_path / "first.mp4", "64x48")
    second = _make_clip(tmp_path / "second.mp4", second_size)
    node = ConcatenateVideos(name="test_concatenate_videos")

    node._process_concatenation([str(first), str(second)], "mp4", video_codec="libx264", audio_codec="aac")

    output = node.parameter_output_values["output"]
    assert isinstance(output, VideoUrlArtifact)
    saved = Path(File(output.value).resolve())
    assert saved.is_relative_to(tmp_path)
    assert saved.name.startswith("concatenated")
    assert saved.stat().st_size > 0
