"""Add Overlay resolves project macro paths on the overlay input and sizes blend-mode overlays.

Project-saved artifacts carry macro paths like ``{outputs}/images/grain.png``. FFmpeg can't open
those, so the overlay has to go through ``File.resolve()`` the same way the base video does
(https://github.com/griptape-ai/griptape-nodes-library-standard/issues/700).

FFmpeg's ``blend`` filter, used by every mode except ``overlay``, also requires both inputs to
share size and pixel format, so the overlay is scaled to the base video using the ``sizing`` option.
"""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING, Any

import pytest
from griptape.artifacts import ImageArtifact, ImageUrlArtifact
from griptape.artifacts.video_url_artifact import VideoUrlArtifact
from griptape_nodes.files import file as file_module

from griptape_nodes_library.video.add_overlay import AddOverlay

if TYPE_CHECKING:
    from pathlib import Path

BASE_SIZE = (1920, 1080)


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
def node(monkeypatch: pytest.MonkeyPatch) -> AddOverlay:
    """An Add Overlay node that never shells out to probe the base video."""
    overlay_node = AddOverlay(name="Add Overlay")
    monkeypatch.setattr(overlay_node, "_get_ffmpeg_paths", lambda: ("ffmpeg", "ffprobe"))
    monkeypatch.setattr(overlay_node, "_detect_video_properties", lambda *_args: (30.0, BASE_SIZE, 5.0))
    return overlay_node


def _filter_complex(cmd: list[str]) -> str:
    return cmd[cmd.index("-filter_complex") + 1]


# ---------------------------------------------------------------------------
# Overlay location resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overlay_input",
    [
        ImageUrlArtifact("{outputs}/images/grain.png"),
        VideoUrlArtifact("{outputs}/images/grain.png"),
        {"type": "ImageUrlArtifact", "value": "{outputs}/images/grain.png"},
    ],
    ids=["image_url_artifact", "video_url_artifact", "serialized_dict"],
)
def test_overlay_macro_path_is_resolved_via_file(
    node: AddOverlay, overlay_input: Any, captured_paths: list[str]
) -> None:
    assert node._resolve_overlay_location(overlay_input) == "/resolved{outputs}/images/grain.png"
    assert captured_paths == ["{outputs}/images/grain.png"]


def test_resolved_overlay_is_the_first_ffmpeg_input(node: AddOverlay, captured_paths: list[str]) -> None:  # noqa: ARG001
    cmd = node._build_ffmpeg_command(
        "/base.mp4", "/out.mp4", 30.0, overlay_video=ImageUrlArtifact("{outputs}/images/grain.png")
    )

    assert cmd[1:5] == ["-i", "/resolved{outputs}/images/grain.png", "-i", "/base.mp4"]


@pytest.mark.parametrize(
    "overlay_input",
    [ImageUrlArtifact("data:image/png;base64,iVBORw0KGgo="), {"value": "data:image/png;base64,iVBORw0KGgo="}],
    ids=["image_url_artifact", "serialized_dict"],
)
def test_overlay_data_uri_reaches_ffmpeg_unchanged(node: AddOverlay, overlay_input: Any) -> None:
    """Uses the real ``File``, which would anchor a data URI under the workspace."""
    assert node._resolve_overlay_location(overlay_input) == "data:image/png;base64,iVBORw0KGgo="


@pytest.mark.parametrize(
    "overlay_input",
    [ImageArtifact(b"\x89PNG", format="png", width=1, height=1), {"type": "ImageUrlArtifact", "value": ""}],
    ids=["byte_artifact", "empty_dict_value"],
)
def test_overlay_without_a_location_raises(node: AddOverlay, overlay_input: Any) -> None:
    with pytest.raises(ValueError, match="'overlay_video' must reference a file or URL"):
        node._resolve_overlay_location(overlay_input)


# ---------------------------------------------------------------------------
# Blend-mode sizing
# ---------------------------------------------------------------------------


def test_scale_to_cover_scales_and_crops_overlay_to_base_size(node: AddOverlay, captured_paths: list[str]) -> None:  # noqa: ARG001
    cmd = node._build_ffmpeg_command(
        "/base.mp4",
        "/out.mp4",
        30.0,
        overlay_video=ImageUrlArtifact("/grain.png"),
        blend_mode="screen",
        sizing="Scale to cover",
    )

    filter_complex = _filter_complex(cmd)
    assert "scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080" in filter_complex
    assert "[1]setsar=1,format=gbrp[bg]" in filter_complex
    assert "format=gbrp[fg]" in filter_complex
    assert "[bg][fg]blend=all_mode=screen" in filter_complex


def test_scale_to_fit_stretches_overlay_to_base_size(node: AddOverlay, captured_paths: list[str]) -> None:  # noqa: ARG001
    cmd = node._build_ffmpeg_command(
        "/base.mp4",
        "/out.mp4",
        30.0,
        overlay_video=ImageUrlArtifact("/grain.png"),
        blend_mode="screen",
        sizing="Scale to fit",
    )

    filter_complex = _filter_complex(cmd)
    assert "scale=1920:1080," in filter_complex
    assert "crop=" not in filter_complex


def test_overlay_mode_keeps_overlay_at_original_size(node: AddOverlay, captured_paths: list[str]) -> None:  # noqa: ARG001
    cmd = node._build_ffmpeg_command(
        "/base.mp4",
        "/out.mp4",
        30.0,
        overlay_video=ImageUrlArtifact("/logo.png"),
        blend_mode="overlay",
        sizing="Scale to cover",
    )

    filter_complex = _filter_complex(cmd)
    assert "scale=" not in filter_complex
    assert "overlay=(W-w)/2:(H-h)/2" in filter_complex


def test_unknown_sizing_raises(node: AddOverlay) -> None:
    with pytest.raises(ValueError, match="Unknown sizing option"):
        node._get_sizing_filter("Scale to taste", 1920, 1080)


# ---------------------------------------------------------------------------
# Real FFmpeg run
# ---------------------------------------------------------------------------


def _ffmpeg_path() -> str:
    from static_ffmpeg import run  # type: ignore[import-untyped]  # noqa: PLC0415

    ffmpeg_path, _ = run.get_or_fetch_platform_executables_else_raise()
    return ffmpeg_path


def _make_input(path: Path, lavfi_source: str, *, frames: int | None = None) -> Path:
    cmd = [_ffmpeg_path(), "-y", "-f", "lavfi", "-i", lavfi_source]
    if frames is not None:
        cmd += ["-frames:v", str(frames)]
    cmd.append(str(path))
    subprocess.run(cmd, capture_output=True, check=True, timeout=120)  # noqa: S603
    return path


@pytest.mark.parametrize("blend_mode", [mode for mode in AddOverlay.BLEND_MODES if mode != "overlay"])
@pytest.mark.parametrize("sizing", AddOverlay.SIZING_OPTIONS)
def test_blend_modes_render_when_overlay_and_base_differ_in_size(tmp_path: Path, blend_mode: str, sizing: str) -> None:
    """Before sizing was applied, FFmpeg refused these with "do not match the corresponding second input"."""
    base = _make_input(tmp_path / "base.mp4", "testsrc=size=320x240:duration=1:rate=10")
    overlay = _make_input(tmp_path / "grain.png", "color=c=red@0.5:s=200x200,format=rgba", frames=1)
    output = tmp_path / "out.mp4"

    overlay_node = AddOverlay(name="Add Overlay")
    cmd = overlay_node._build_ffmpeg_command(
        str(base),
        str(output),
        10.0,
        overlay_video=ImageUrlArtifact(str(overlay)),
        blend_mode=blend_mode,
        amount=0.5,
        sizing=sizing,
    )

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)  # noqa: S603
    assert result.returncode == 0, result.stderr[-2000:]
    assert output.stat().st_size > 0
