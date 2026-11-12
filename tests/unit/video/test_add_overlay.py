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
    assert "[1]setsar=1,format=gbrp,split[bg_blend][bg_keep]" in filter_complex
    assert "[bg_blend][fg_color]blend=all_mode=screen" in filter_complex


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


# ---------------------------------------------------------------------------
# Alpha: blend modes respect it, a matte supplies it, premultiplied input is unpremultiplied
# ---------------------------------------------------------------------------

SIZE = "64x64"


def _build(node: AddOverlay, base: Path, output: Path, overlay: Path, **kwargs: Any) -> list[str]:
    params: dict[str, Any] = {"blend_mode": "overlay", "amount": 1.0, "sizing": "Scale to fit"}
    params.update(kwargs)
    if "matte_video" in params:
        params["matte_video"] = ImageUrlArtifact(str(params["matte_video"]))
    return node._build_ffmpeg_command(
        str(base), str(output), 10.0, overlay_video=ImageUrlArtifact(str(overlay)), **params
    )


def _render_pixel(cmd: list[str], output: Path, tmp_path: Path, xy: tuple[int, int] = (32, 32)) -> tuple[int, int, int]:
    from PIL import Image  # noqa: PLC0415

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)  # noqa: S603
    assert result.returncode == 0, result.stderr[-2000:]
    frame = tmp_path / "frame.png"
    subprocess.run(  # noqa: S603
        [_ffmpeg_path(), "-y", "-i", str(output), "-frames:v", "1", "-update", "1", str(frame)],
        capture_output=True,
        check=True,
        timeout=120,
    )
    pixel = Image.open(frame).convert("RGB").getpixel(xy)
    assert isinstance(pixel, tuple)
    return pixel[0], pixel[1], pixel[2]


@pytest.fixture
def gray_base(tmp_path: Path) -> Path:
    return _make_input(tmp_path / "base.mp4", f"color=c=0x646464:s={SIZE}:duration=1:rate=10")


@pytest.fixture
def black_base(tmp_path: Path) -> Path:
    return _make_input(tmp_path / "base.mp4", f"color=c=black:s={SIZE}:duration=1:rate=10")


def _assert_close(actual: tuple[int, int, int], expected: int, tolerance: int = 6) -> None:
    assert all(abs(channel - expected) <= tolerance for channel in actual), f"{actual} is not near {expected}"


@pytest.mark.parametrize("blend_mode", [mode for mode in AddOverlay.BLEND_MODES if mode != "overlay"])
def test_blend_modes_leave_base_untouched_under_a_transparent_overlay(
    tmp_path: Path, gray_base: Path, blend_mode: str
) -> None:
    overlay = _make_input(tmp_path / "clear.png", f"color=c=white@0.0:s={SIZE},format=rgba", frames=1)
    output = tmp_path / "out.mp4"

    cmd = _build(AddOverlay(name="Add Overlay"), gray_base, output, overlay, blend_mode=blend_mode)

    _assert_close(_render_pixel(cmd, output, tmp_path), 100)


def test_screen_blend_mixes_by_overlay_alpha(tmp_path: Path, gray_base: Path) -> None:
    # White at half alpha screened over gray 100 is white; half-mixed with the base gives about 178.
    overlay = _make_input(tmp_path / "half.png", f"color=c=white@0.5:s={SIZE},format=rgba", frames=1)
    output = tmp_path / "out.mp4"

    cmd = _build(AddOverlay(name="Add Overlay"), gray_base, output, overlay, blend_mode="screen")

    _assert_close(_render_pixel(cmd, output, tmp_path), 178)


def test_premultiplied_alpha_is_unpremultiplied_before_compositing(tmp_path: Path, black_base: Path) -> None:
    # RGB 128 at alpha 128. Premultiplied means the colour is already scaled: 128 over black stays about 128.
    # Read as straight, the same pixel composites to about 64.
    overlay = _make_input(tmp_path / "pm.png", f"color=c=0x808080@0.5:s={SIZE},format=rgba", frames=1)
    node = AddOverlay(name="Add Overlay")

    straight_out = tmp_path / "straight.mp4"
    straight = _render_pixel(_build(node, black_base, straight_out, overlay), straight_out, tmp_path)
    premultiplied_out = tmp_path / "premultiplied.mp4"
    premultiplied = _render_pixel(
        _build(node, black_base, premultiplied_out, overlay, alpha_convention="premultiplied"),
        premultiplied_out,
        tmp_path,
    )

    _assert_close(straight, 64)
    _assert_close(premultiplied, 128)


@pytest.mark.parametrize("blend_mode", ["overlay", "screen"])
def test_black_matte_hides_and_white_matte_shows_the_overlay(tmp_path: Path, black_base: Path, blend_mode: str) -> None:
    overlay = _make_input(tmp_path / "white.png", f"color=c=white:s={SIZE}", frames=1)
    black_matte = _make_input(tmp_path / "black_matte.png", f"color=c=black:s={SIZE}", frames=1)
    white_matte = _make_input(tmp_path / "white_matte.png", f"color=c=white:s={SIZE}", frames=1)
    node = AddOverlay(name="Add Overlay")

    hidden_out = tmp_path / "hidden.mp4"
    hidden = _render_pixel(
        _build(node, black_base, hidden_out, overlay, blend_mode=blend_mode, matte_video=black_matte),
        hidden_out,
        tmp_path,
    )
    shown_out = tmp_path / "shown.mp4"
    shown = _render_pixel(
        _build(node, black_base, shown_out, overlay, blend_mode=blend_mode, matte_video=white_matte),
        shown_out,
        tmp_path,
    )

    _assert_close(hidden, 0)
    _assert_close(shown, 255, tolerance=10)


def test_matte_is_scaled_to_the_overlay_size(tmp_path: Path, black_base: Path) -> None:
    overlay = _make_input(tmp_path / "white.png", f"color=c=white:s={SIZE}", frames=1)
    small_matte = _make_input(tmp_path / "matte.png", "color=c=white:s=8x8", frames=1)
    output = tmp_path / "out.mp4"

    cmd = _build(AddOverlay(name="Add Overlay"), black_base, output, overlay, matte_video=small_matte)

    _assert_close(_render_pixel(cmd, output, tmp_path), 255, tolerance=10)


def test_matte_is_the_third_ffmpeg_input_and_absent_without_one(node: AddOverlay, captured_paths: list[str]) -> None:  # noqa: ARG001
    with_matte = node._build_ffmpeg_command(
        "/base.mp4",
        "/out.mp4",
        30.0,
        overlay_video=ImageUrlArtifact("/logo.png"),
        matte_video=ImageUrlArtifact("/matte.png"),
    )
    without_matte = node._build_ffmpeg_command(
        "/base.mp4", "/out.mp4", 30.0, overlay_video=ImageUrlArtifact("/logo.png")
    )

    inputs = [with_matte[i + 1] for i, token in enumerate(with_matte) if token == "-i"]
    assert inputs == ["/resolved/logo.png", "/base.mp4", "/resolved/matte.png"]
    assert "[2]" in _filter_complex(with_matte)
    assert with_matte.count("-i") == 3
    assert without_matte.count("-i") == 2
    assert "[2]" not in _filter_complex(without_matte)


def test_matte_without_a_location_raises(node: AddOverlay) -> None:
    with pytest.raises(ValueError, match="matte_video"):
        node._build_ffmpeg_command(
            "/base.mp4",
            "/out.mp4",
            30.0,
            overlay_video=ImageUrlArtifact("/logo.png"),
            matte_video=ImageArtifact(b"\x00", format="png", width=1, height=1),
        )


def test_unknown_alpha_convention_raises(node: AddOverlay) -> None:
    with pytest.raises(ValueError, match="Unknown alpha convention"):
        node._build_ffmpeg_command(
            "/base.mp4",
            "/out.mp4",
            30.0,
            overlay_video=ImageUrlArtifact("/logo.png"),
            alpha_convention="sideways",
        )


# ---------------------------------------------------------------------------
# Edge cleanup: choke and edge softness
# ---------------------------------------------------------------------------


def test_edge_cleanup_adds_nothing_to_the_graph_by_default(node: AddOverlay, captured_paths: list[str]) -> None:  # noqa: ARG001
    cmd = node._build_ffmpeg_command("/base.mp4", "/out.mp4", 30.0, overlay_video=ImageUrlArtifact("/logo.png"))

    filter_complex = _filter_complex(cmd)
    assert "erosion" not in filter_complex
    assert "gblur" not in filter_complex


def test_choke_erodes_the_alpha_once_per_pixel(node: AddOverlay, captured_paths: list[str]) -> None:  # noqa: ARG001
    cmd = node._build_ffmpeg_command(
        "/base.mp4", "/out.mp4", 30.0, overlay_video=ImageUrlArtifact("/logo.png"), choke=3, edge_softness=1.5
    )

    filter_complex = _filter_complex(cmd)
    assert "alphaextract,erosion,erosion,erosion,gblur=sigma=1.5[edge_alpha]" in filter_complex


def test_choke_pulls_the_edge_in_and_leaves_the_centre(tmp_path: Path, black_base: Path) -> None:
    # An opaque white square at x=16..47. One pixel in from its left edge is solid white without a choke
    # and becomes base black once the alpha is eroded by two pixels. The centre stays white.
    overlay = _make_input(
        tmp_path / "square.png",
        f"color=c=black@0.0:s={SIZE},format=rgba,drawbox=x=16:y=16:w=32:h=32:color=white@1.0:t=fill:replace=1",
        frames=1,
    )
    node = AddOverlay(name="Add Overlay")

    plain_out = tmp_path / "plain.mp4"
    plain = _render_pixel(_build(node, black_base, plain_out, overlay), plain_out, tmp_path, xy=(17, 32))
    choked_out = tmp_path / "choked.mp4"
    choked_edge = _render_pixel(
        _build(node, black_base, choked_out, overlay, choke=2), choked_out, tmp_path, xy=(17, 32)
    )
    choked_centre = _render_pixel(_build(node, black_base, choked_out, overlay, choke=2), choked_out, tmp_path)

    _assert_close(plain, 255, tolerance=10)
    _assert_close(choked_edge, 0, tolerance=10)
    _assert_close(choked_centre, 255, tolerance=10)


@pytest.mark.parametrize("blend_mode", ["overlay", "screen"])
def test_choke_and_softness_render_in_every_mode(tmp_path: Path, gray_base: Path, blend_mode: str) -> None:
    overlay = _make_input(tmp_path / "half.png", f"color=c=white@0.5:s={SIZE},format=rgba", frames=1)
    output = tmp_path / "out.mp4"

    cmd = _build(
        AddOverlay(name="Add Overlay"), gray_base, output, overlay, blend_mode=blend_mode, choke=2, edge_softness=1.0
    )

    _render_pixel(cmd, output, tmp_path)


# ---------------------------------------------------------------------------
# A still image on either input: the video sets the length
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("location", "expected"),
    [
        ("/a/b/logo.PNG", True),
        ("/a/b/photo.jpeg", True),
        ("http://localhost:8124/static/a.webp?x=1", True),
        ("data:image/png;base64,AAAA", True),
        ("data:video/mp4;base64,AAAA", False),
        ("/a/b/clip.mp4", False),
        ("/a/b/clip.mov", False),
        ("http://localhost:8124/static/clip.mp4?x=1.png", False),
    ],
)
def test_still_images_are_recognised_by_location(node: AddOverlay, location: str, expected: bool) -> None:  # noqa: FBT001
    assert node._is_still_image(location) is expected


def test_an_image_is_accepted_as_the_base_input(node: AddOverlay) -> None:
    node.parameter_values["video"] = ImageUrlArtifact("/base.png")

    assert node._validate_video_input() is None


def test_a_missing_base_input_is_still_rejected(node: AddOverlay) -> None:
    node.parameter_values["video"] = None

    assert node._validate_video_input()


def test_video_overlay_over_image_base_takes_the_overlay_length(tmp_path: Path) -> None:
    base = _make_input(tmp_path / "base.png", f"color=c=blue:s={SIZE}", frames=1)
    overlay = _make_input(tmp_path / "overlay.mp4", f"color=c=red:s={SIZE}:duration=1:rate=10")
    output = tmp_path / "out.mp4"

    cmd = _build(AddOverlay(name="Add Overlay"), base, output, overlay)
    pixel = _render_pixel(cmd, output, tmp_path)

    probe = subprocess.run(  # noqa: S603
        [
            AddOverlay(name="Add Overlay")._get_ffmpeg_paths()[1],
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames,r_frame_rate",
            "-of",
            "csv=p=0",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    frame_rate, frame_count = probe.stdout.strip().split(",")
    assert frame_rate == "10/1"
    assert int(frame_count) == 10
    assert pixel[0] > 200
    assert pixel[2] < 60


def test_image_overlay_over_video_base_runs_for_the_video_length(tmp_path: Path) -> None:
    base = _make_input(tmp_path / "base.mp4", f"color=c=blue:s={SIZE}:duration=1:rate=10")
    overlay = _make_input(tmp_path / "overlay.png", f"color=c=red:s={SIZE}", frames=1)
    output = tmp_path / "out.mp4"

    cmd = _build(AddOverlay(name="Add Overlay"), base, output, overlay)
    pixel = _render_pixel(cmd, output, tmp_path)

    assert pixel[0] > 200
    assert pixel[2] < 60


def test_two_images_raise_because_the_result_has_no_length(tmp_path: Path) -> None:
    base = _make_input(tmp_path / "base.png", f"color=c=blue:s={SIZE}", frames=1)
    overlay = _make_input(tmp_path / "overlay.png", f"color=c=red:s={SIZE}", frames=1)

    with pytest.raises(ValueError, match="At least one"):
        _build(AddOverlay(name="Add Overlay"), base, tmp_path / "out.mp4", overlay)


def test_image_base_does_not_loop_a_video_base(node: AddOverlay, captured_paths: list[str]) -> None:  # noqa: ARG001
    cmd = node._build_ffmpeg_command("/base.mp4", "/out.mp4", 30.0, overlay_video=ImageUrlArtifact("/logo.png"))

    assert "-loop" not in cmd


def test_amount_defaults_to_full_strength() -> None:
    assert AddOverlay(name="Add Overlay").get_parameter_value("amount") == 1.0
