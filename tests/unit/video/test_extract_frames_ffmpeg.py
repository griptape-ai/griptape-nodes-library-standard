"""Unit tests for the ffmpeg commands Extract Frames and Extract Last Frame run.

ffmpeg is stubbed out; these check the arguments passed and the error raised on failure.
"""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING, Any

import pytest

from griptape_nodes_library.video import extract_frames
from griptape_nodes_library.video.extract_frames import ExtractFrames
from griptape_nodes_library.video.extract_last_frame import ExtractLastFrame

if TYPE_CHECKING:
    from pathlib import Path

BANNER = "ffmpeg version 9.0 Copyright (c) 2000-2026 the FFmpeg developers\n" + "  configuration: --enable-x\n" * 40


@pytest.fixture
def ffmpeg_calls(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    monkeypatch.setattr(
        extract_frames.run, "get_or_fetch_platform_executables_else_raise", lambda: ("ffmpeg", "ffprobe")
    )
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(extract_frames.subprocess, "run", fake_run)
    return calls


def test_extract_frames_uses_fps_mode_not_vsync(ffmpeg_calls: list[list[str]], tmp_path: Path) -> None:
    node = ExtractFrames(name="ef")
    node._extract_frames("in.mp4", [1, 3], tmp_path, "frames", 4, "png")

    [cmd] = ffmpeg_calls
    assert "-vsync" not in cmd
    assert cmd[cmd.index("-fps_mode") + 1] == "passthrough"
    assert "-hide_banner" in cmd


def test_extract_frames_error_keeps_the_end_of_stderr(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        extract_frames.run, "get_or_fetch_platform_executables_else_raise", lambda: ("ffmpeg", "ffprobe")
    )

    def failing_run(cmd: list[str], **_kwargs: Any) -> None:
        raise subprocess.CalledProcessError(8, cmd, "", BANNER + "Unrecognized option 'vsync'.\n")

    monkeypatch.setattr(extract_frames.subprocess, "run", failing_run)
    node = ExtractFrames(name="ef")

    with pytest.raises(RuntimeError, match="Unrecognized option 'vsync'"):
        node._extract_frames("in.mp4", [1], tmp_path, "frames", 4, "png")


def test_extract_last_frame_uses_fps_mode_not_vsync(monkeypatch: pytest.MonkeyPatch) -> None:
    node = ExtractLastFrame(name="elf")
    monkeypatch.setattr(node, "_get_ffmpeg_paths", lambda: ("ffmpeg", "ffprobe"))

    cmd = node._build_ffmpeg_command("in.mp4", "out.png", 24.0)

    assert "-vsync" not in cmd
    assert cmd[cmd.index("-fps_mode") + 1] == "passthrough"
