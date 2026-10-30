"""Regression tests for ExtractFrames' ``input_video`` handling.

The engine stores media inputs as file paths and maps static server URLs back to paths. A node that
writes a preview URL into its own video input from ``after_value_set`` gets a path back, sees a
mismatch, and writes again until Python hits its recursion limit.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

from griptape_nodes_library.video.extract_frames import ExtractFrames


@pytest.fixture
def workspace_video() -> Iterator[Path]:
    path = Path(GriptapeNodes.ConfigManager().workspace_path) / "extract_frames_clip.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    yield path
    path.unlink(missing_ok=True)


def test_video_path_is_stored_and_fps_detected(workspace_video: Path) -> None:
    node = ExtractFrames(name="test_extract_frames")

    with patch.object(ExtractFrames, "_probe_video", return_value=(24.0, 2.0)) as probe:
        node.set_parameter_value("input_video", str(workspace_video))

    stored = node.get_parameter_value("input_video")
    assert str(getattr(stored, "value", stored)).endswith("/extract_frames_clip.mp4")
    assert node.get_parameter_value("_video_fps") == 24.0  # noqa: PLR2004
    probe.assert_called_once_with(str(workspace_video))


def test_clearing_video_skips_fps_probe() -> None:
    node = ExtractFrames(name="test_extract_frames")

    with patch.object(ExtractFrames, "_probe_video") as probe:
        node.set_parameter_value("input_video", "")

    assert not node.get_parameter_value("input_video")
    probe.assert_not_called()


def test_same_video_is_probed_once(workspace_video: Path) -> None:
    node = ExtractFrames(name="test_extract_frames")

    with patch.object(ExtractFrames, "_probe_video", return_value=(24.0, 2.0)) as probe:
        node.set_parameter_value("input_video", str(workspace_video))
        node.set_parameter_value("input_video", str(workspace_video))

    probe.assert_called_once()
