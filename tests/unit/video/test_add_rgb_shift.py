from __future__ import annotations

import shutil
import subprocess
from typing import TYPE_CHECKING

import pytest

from griptape_nodes_library.video.add_rgb_shift import AddRGBShift

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if path is None:
        pytest.skip("ffmpeg is not on PATH")
    return path


@pytest.fixture
def input_video(tmp_path: Path, ffmpeg: str) -> Path:
    path = tmp_path / "input.mp4"
    subprocess.run(  # noqa: S603
        [ffmpeg, "-v", "error", "-f", "lavfi", "-i", "testsrc=size=64x64:rate=24:duration=0.5", "-y", str(path)],
        check=True,
    )
    return path


@pytest.mark.parametrize("tear_enabled", [True, False])
@pytest.mark.parametrize("output_frame_rate", ["auto", "30"])
def test_ffmpeg_command_runs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    ffmpeg: str,
    input_video: Path,
    *,
    tear_enabled: bool,
    output_frame_rate: str,
) -> None:
    node = AddRGBShift(name="RGB Shift")
    node.set_parameter_value("output_frame_rate", output_frame_rate)
    monkeypatch.setattr(node, "_get_ffmpeg_paths", lambda: (ffmpeg, ""))
    output_path = tmp_path / "output.mp4"

    cmd = node._build_ffmpeg_command(
        str(input_video),
        str(output_path),
        24.0,
        tear_enabled=tear_enabled,
        tear_position=0.5,
        tear_offset=10,
        intensity=1.0,
    )
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603

    assert result.returncode == 0, result.stderr
    assert output_path.stat().st_size > 0
