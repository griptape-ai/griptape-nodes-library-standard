"""Nodes that probe their input video resolve project macro paths before calling ffprobe.

ffprobe can't open ``{outputs}/clip.mp4``. The LTX nodes swallow probe failures, so an unresolved
path silently skipped their input checks instead of failing. Data URIs must skip ``File.resolve()``,
which would treat them as relative file paths.
"""

from __future__ import annotations

import json
import subprocess
from typing import Any

import pytest
from griptape_nodes.files import file as file_module
from griptape_nodes.retained_mode.events.os_events import FileIOFailureReason

from griptape_nodes_library.video import ltx_video_retake as retake_module
from griptape_nodes_library.video import ltx_video_to_video_hdr as hdr_module
from griptape_nodes_library.video import topaz_video_upscale as topaz_module
from griptape_nodes_library.video.ltx_video_retake import LTXVideoRetake
from griptape_nodes_library.video.ltx_video_to_video_hdr import LTXVideoToVideoHDR
from griptape_nodes_library.video.topaz_video_upscale import TopazVideoUpscale

MACRO_PATH = "{outputs}/clip.mp4"
RESOLVED_PATH = "/resolved{outputs}/clip.mp4"
DATA_URI = "data:video/mp4;base64,AAAA"

_STREAM = {"width": 1920, "height": 1080, "duration": "4.0", "nb_read_frames": "96"}


@pytest.fixture
def captured_paths(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record every location handed to ``File`` and resolve each to ``/resolved<location>``."""
    paths: list[str] = []
    real_init = file_module.File.__init__

    def capture_init(self: file_module.File, path: Any, *args: Any, **kwargs: Any) -> None:
        paths.append(path)
        real_init(self, path, *args, **kwargs)

    def fake_resolve(self: file_module.File) -> str:  # noqa: ARG001
        return f"/resolved{paths[-1]}"

    monkeypatch.setattr(file_module.File, "__init__", capture_init)
    monkeypatch.setattr(file_module.File, "resolve", fake_resolve)
    return paths


@pytest.fixture
def ffprobe_calls(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Capture ffprobe commands and answer each with a single 1080p stream."""
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps({"streams": [_STREAM]}), stderr="")

    # Both LTX modules import the same static_ffmpeg ``run`` module.
    monkeypatch.setattr(
        retake_module.run, "get_or_fetch_platform_executables_else_raise", lambda: ("ffmpeg", "ffprobe")
    )
    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


@pytest.fixture
def failing_resolve(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_resolve(self: file_module.File) -> str:  # noqa: ARG001
        raise file_module.FileLoadError(FileIOFailureReason.MISSING_MACRO_VARIABLES, "no project loaded")

    monkeypatch.setattr(file_module.File, "resolve", fail_resolve)


# ---------------------------------------------------------------------------
# LTX Video Retake
# ---------------------------------------------------------------------------


def test_retake_probe_resolves_macro_path(captured_paths: list[str], ffprobe_calls: list[list[str]]) -> None:
    info = LTXVideoRetake(name="Retake")._get_video_stream_info(MACRO_PATH)

    assert ffprobe_calls[0][-1] == RESOLVED_PATH
    assert captured_paths == [MACRO_PATH]
    assert info == {"duration": 4.0, "width": 1920, "height": 1080}


def test_retake_probe_passes_data_uri_through(captured_paths: list[str], ffprobe_calls: list[list[str]]) -> None:
    LTXVideoRetake(name="Retake")._get_video_stream_info(DATA_URI)

    assert ffprobe_calls[0][-1] == DATA_URI
    assert captured_paths == []


@pytest.mark.usefixtures("failing_resolve")
def test_retake_probe_is_skipped_when_resolution_fails(ffprobe_calls: list[list[str]]) -> None:
    assert LTXVideoRetake(name="Retake")._get_video_stream_info(MACRO_PATH) is None
    assert ffprobe_calls == []


# ---------------------------------------------------------------------------
# LTX Video to Video HDR
# ---------------------------------------------------------------------------


def test_hdr_probe_resolves_macro_path(captured_paths: list[str], ffprobe_calls: list[list[str]]) -> None:
    assert hdr_module.run is retake_module.run

    probed = LTXVideoToVideoHDR(name="HDR")._probe_video(MACRO_PATH)

    assert ffprobe_calls[0][-1] == RESOLVED_PATH
    assert captured_paths == [MACRO_PATH]
    assert probed == (1920, 1080, 96)


def test_hdr_probe_passes_data_uri_through(captured_paths: list[str], ffprobe_calls: list[list[str]]) -> None:
    LTXVideoToVideoHDR(name="HDR")._probe_video(DATA_URI)

    assert ffprobe_calls[0][-1] == DATA_URI
    assert captured_paths == []


@pytest.mark.usefixtures("failing_resolve")
def test_hdr_probe_is_skipped_when_resolution_fails(ffprobe_calls: list[list[str]]) -> None:
    assert LTXVideoToVideoHDR(name="HDR")._probe_video(MACRO_PATH) is None
    assert ffprobe_calls == []


# ---------------------------------------------------------------------------
# Topaz Video Upscale
# ---------------------------------------------------------------------------


@pytest.fixture
def probed_locations(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    locations: list[str] = []

    def fake_extract(location: str) -> str:
        locations.append(location)
        return "metadata"

    monkeypatch.setattr(topaz_module, "extract_video_metadata_structured", fake_extract)
    return locations


def test_topaz_probe_resolves_macro_path(captured_paths: list[str], probed_locations: list[str]) -> None:
    TopazVideoUpscale(name="Topaz")._probe_source(MACRO_PATH)

    assert probed_locations == [RESOLVED_PATH]
    assert MACRO_PATH in captured_paths


def test_topaz_probe_passes_data_uri_through(captured_paths: list[str], probed_locations: list[str]) -> None:
    TopazVideoUpscale(name="Topaz")._probe_source(DATA_URI)

    assert probed_locations == [DATA_URI]
    assert captured_paths == []
