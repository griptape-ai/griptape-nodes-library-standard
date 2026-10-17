"""Tests for the pre-flight checks proxy nodes run before submitting a request."""

from __future__ import annotations

import base64
import json
import subprocess
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from griptape.artifacts import ImageArtifact, ImageUrlArtifact
from griptape.artifacts.video_url_artifact import VideoUrlArtifact
from griptape_nodes.exe_types.core_types import NodeError
from PIL import Image

from griptape_nodes_library.proxy import preflight
from griptape_nodes_library.proxy.preflight import (
    ImageInfo,
    check_aspect_ratio,
    check_aspect_ratio_range,
    check_count,
    check_dimensions,
    check_duration,
    check_mime_type,
    check_size,
    input_subject,
    local_image_info,
    local_size_bytes,
    local_video_duration,
    local_video_info,
)

SUBJECT = "Image 2 in 'reference images'"
DOCS = "https://example.com/limits"


def _png_bytes(width: int, height: int) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (width, height)).save(buffer, format="PNG")
    return buffer.getvalue()


def _data_uri(data: bytes, mime_type: str = "image/png") -> str:
    return f"data:{mime_type};base64,{base64.b64encode(data).decode()}"


class TestInputSubject:
    def test_names_a_position_in_a_list(self) -> None:
        assert input_subject("reference images", noun="image", position=2) == SUBJECT

    def test_names_a_single_input(self) -> None:
        assert input_subject("start frame", noun="image") == "The image in 'start frame'"


class TestLocalSizeBytes:
    @pytest.mark.parametrize("length", [1, 2, 3, 100, 1001])
    def test_measures_a_data_uri_exactly(self, length: int) -> None:
        assert local_size_bytes(_data_uri(b"x" * length), kind="image") == length

    def test_ignores_line_breaks_in_a_data_uri(self) -> None:
        wrapped = f"data:image/png;base64,{base64.encodebytes(b'x' * 1000).decode()}"
        assert local_size_bytes(wrapped, kind="image") == 1000

    def test_measures_a_local_file(self, tmp_path: Path) -> None:
        path = tmp_path / "image.png"
        path.write_bytes(b"x" * 1234)
        assert local_size_bytes(ImageUrlArtifact(value=str(path)), kind="image") == 1234

    def test_skips_a_remote_url(self) -> None:
        assert local_size_bytes("https://example.com/image.png", kind="image") is None

    def test_skips_a_path_the_os_refuses(self) -> None:
        assert local_size_bytes("A" * 3000, kind="image") is None

    def test_skips_a_missing_file(self, tmp_path: Path) -> None:
        assert local_size_bytes(str(tmp_path / "missing.png"), kind="image") is None

    def test_skips_an_empty_value(self) -> None:
        assert local_size_bytes(None, kind="image") is None


class TestLocalImageInfo:
    def test_reads_a_data_uri(self) -> None:
        assert local_image_info(_data_uri(_png_bytes(40, 30))) == ImageInfo(40, 30, "image/png")

    def test_reads_a_local_file(self, tmp_path: Path) -> None:
        path = tmp_path / "image.jpg"
        Image.new("RGB", (64, 48)).save(path, format="JPEG")
        assert local_image_info(str(path)) == ImageInfo(64, 48, "image/jpeg")

    def test_reads_an_image_artifact(self) -> None:
        artifact = ImageArtifact(value=_png_bytes(8, 4), format="png", width=8, height=4)
        assert local_image_info(artifact) == ImageInfo(8, 4, "image/png")

    def test_reports_a_multi_picture_jpeg_as_jpeg(self) -> None:
        buffer = BytesIO()
        Image.new("RGB", (8, 8)).save(buffer, format="MPO", save_all=True, append_images=[Image.new("RGB", (8, 8))])
        assert local_image_info(_data_uri(buffer.getvalue(), "image/jpeg")) == ImageInfo(8, 8, "image/jpeg")

    def test_skips_an_image_too_large_to_open(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 10)
        assert local_image_info(_data_uri(_png_bytes(40, 30))) is None

    def test_skips_a_remote_url(self) -> None:
        assert local_image_info("https://example.com/image.png") is None

    def test_skips_bytes_that_are_not_an_image(self) -> None:
        assert local_image_info(_data_uri(b"not an image")) is None


class TestLocalVideoDuration:
    @pytest.fixture(autouse=True)
    def no_ffmpeg_download(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fetch(*_args: Any, **_kwargs: Any) -> None:
            pytest.fail("pre-flight must never fetch ffmpeg")

        monkeypatch.setattr(preflight.static_ffmpeg_run, "get_or_fetch_platform_executables_else_raise", fetch)

    @pytest.fixture
    def clip(self, tmp_path: Path) -> Path:
        path = tmp_path / "clip.mp4"
        path.write_bytes(b"video")
        return path

    @pytest.fixture
    def ffprobe_runs(self, monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
        """Pretend ffprobe is installed and answer every run with a 9.5 second duration."""
        runs: list[dict[str, Any]] = []

        def fake_run(cmd: list[str], **kwargs: Any) -> SimpleNamespace:
            runs.append({"cmd": cmd, **kwargs})
            return SimpleNamespace(stdout=json.dumps({"format": {"duration": "9.5"}}))

        monkeypatch.setattr(preflight, "_installed_ffprobe", lambda: "/usr/bin/ffprobe")
        monkeypatch.setattr(preflight.subprocess, "run", fake_run)
        return runs

    def test_probes_a_local_file_with_a_short_timeout(self, clip: Path, ffprobe_runs: list[dict[str, Any]]) -> None:
        assert local_video_duration(VideoUrlArtifact(value=str(clip))) == 9.5
        assert ffprobe_runs[0]["cmd"][0] == "/usr/bin/ffprobe"
        assert ffprobe_runs[0]["cmd"][-1] == str(clip)
        assert ffprobe_runs[0]["timeout"] == preflight.FFPROBE_TIMEOUT_SECONDS

    def test_skips_when_ffprobe_times_out(self, clip: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        def slow_run(cmd: list[str], **kwargs: Any) -> SimpleNamespace:
            raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])

        monkeypatch.setattr(preflight, "_installed_ffprobe", lambda: "/usr/bin/ffprobe")
        monkeypatch.setattr(preflight.subprocess, "run", slow_run)

        assert local_video_duration(str(clip)) is None

    def test_skips_when_the_probe_fails(self, clip: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        def failing_run(cmd: list[str], **_kwargs: Any) -> SimpleNamespace:
            raise subprocess.CalledProcessError(1, cmd, stderr="Invalid data found")

        monkeypatch.setattr(preflight, "_installed_ffprobe", lambda: "/usr/bin/ffprobe")
        monkeypatch.setattr(preflight.subprocess, "run", failing_run)

        assert local_video_duration(str(clip)) is None

    def test_skips_when_ffprobe_is_not_installed(self, clip: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(preflight, "_installed_ffprobe", lambda: None)
        monkeypatch.setattr(preflight.subprocess, "run", lambda *_a, **_k: pytest.fail("ffprobe must not run"))

        assert local_video_duration(str(clip)) is None

    @pytest.mark.parametrize("value", ["https://example.com/clip.mp4", "data:video/mp4;base64,AAAA"])
    def test_does_not_probe_a_remote_or_inline_video(self, value: str, ffprobe_runs: list[dict[str, Any]]) -> None:
        assert local_video_duration(value) is None
        assert ffprobe_runs == []


class TestLocalVideoInfo:
    @pytest.fixture
    def clip(self, tmp_path: Path) -> Path:
        path = tmp_path / "clip.mp4"
        path.write_bytes(b"video")
        return path

    def _answer(self, monkeypatch: pytest.MonkeyPatch, stream: dict[str, Any]) -> None:
        monkeypatch.setattr(preflight, "_installed_ffprobe", lambda: "/usr/bin/ffprobe")
        monkeypatch.setattr(
            preflight.subprocess,
            "run",
            lambda *_a, **_k: SimpleNamespace(stdout=json.dumps({"streams": [stream]})),
        )

    def test_reads_the_frame_size(self, clip: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self._answer(monkeypatch, {"width": 1920, "height": 1080})
        assert local_video_info(str(clip)) == ImageInfo(1920, 1080, None)

    @pytest.mark.parametrize(
        "stream",
        [
            {"width": 1920, "height": 1080, "tags": {"rotate": "90"}},
            {"width": 1920, "height": 1080, "side_data_list": [{"rotation": -90}]},
        ],
        ids=["rotate-tag", "display-matrix"],
    )
    def test_reports_a_rotated_clip_as_displayed(
        self, clip: Path, monkeypatch: pytest.MonkeyPatch, stream: dict[str, Any]
    ) -> None:
        self._answer(monkeypatch, stream)
        assert local_video_info(str(clip)) == ImageInfo(1080, 1920, None)

    def test_skips_when_ffprobe_is_not_installed(self, clip: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(preflight, "_installed_ffprobe", lambda: None)
        assert local_video_info(str(clip)) is None

    def test_skips_a_remote_video(self) -> None:
        assert local_video_info("https://example.com/clip.mp4") is None


class TestInstalledFfprobe:
    @pytest.fixture(autouse=True)
    def no_ffmpeg_download(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fetch(*_args: Any, **_kwargs: Any) -> None:
            pytest.fail("pre-flight must never fetch ffmpeg")

        monkeypatch.setattr(preflight.static_ffmpeg_run, "get_or_fetch_platform_executables_else_raise", fetch)

    def _platform_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        monkeypatch.setattr(preflight.static_ffmpeg_run, "get_platform_dir", lambda: str(tmp_path))
        (tmp_path / "ffprobe").write_bytes(b"")
        return tmp_path

    def test_uses_a_finished_static_ffmpeg_install(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        platform_dir = self._platform_dir(tmp_path, monkeypatch)
        (platform_dir / "installed.crumb").write_text("installed")
        assert preflight._installed_ffprobe() == str(platform_dir / "ffprobe")

    def test_ignores_an_unfinished_static_ffmpeg_install(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self._platform_dir(tmp_path, monkeypatch)
        monkeypatch.setattr(preflight.shutil, "which", lambda _name: None)
        assert preflight._installed_ffprobe() is None

    def test_falls_back_to_ffprobe_on_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self._platform_dir(tmp_path, monkeypatch)
        monkeypatch.setattr(preflight.shutil, "which", lambda _name: "/opt/bin/ffprobe")
        assert preflight._installed_ffprobe() == "/opt/bin/ffprobe"


class TestCheckCount:
    def test_allows_the_maximum(self) -> None:
        assert check_count(3, maximum=3, parameter="reference images") is None

    def test_rejects_more_than_the_maximum(self) -> None:
        error = check_count(5, maximum=3, parameter="reference images", noun="images", docs_url=DOCS)
        assert isinstance(error, NodeError)
        assert str(error) == "5 images are connected to 'reference images' and the limit is 3. Remove 2."
        assert error.links is not None
        assert error.links[0].url == DOCS


class TestCheckSize:
    def test_allows_the_maximum(self) -> None:
        assert check_size(7 * 1024 * 1024, maximum_bytes=7 * 1024 * 1024, subject=SUBJECT) is None

    def test_skips_an_unknown_size(self) -> None:
        assert check_size(None, maximum_bytes=1, subject=SUBJECT) is None

    def test_rejects_a_larger_file(self) -> None:
        error = check_size(int(8.5 * 1024 * 1024), maximum_bytes=7 * 1024 * 1024, subject=SUBJECT)
        assert error is not None
        assert str(error) == f"{SUBJECT} is 8.5 MB and the limit is 7 MB. Use a smaller file."
        assert not error.links

    def test_reports_a_size_just_over_the_limit_as_over_it(self) -> None:
        error = check_size(7 * 1024 * 1024 + 1024, maximum_bytes=7 * 1024 * 1024, subject=SUBJECT)
        assert str(error).startswith(f"{SUBJECT} is 7.001 MB and the limit is 7 MB.")

    def test_reports_a_size_over_a_non_round_limit_as_over_it(self) -> None:
        error = check_size(10_000_001, maximum_bytes=10_000_000, subject=SUBJECT)
        assert str(error).startswith(f"{SUBJECT} is 9.54 MB and the limit is 9.53674 MB.")

    def test_never_shows_the_size_and_limit_as_the_same_number(self) -> None:
        error = check_size(200_000_001, maximum_bytes=200_000_000, subject=SUBJECT)
        assert str(error).startswith(f"{SUBJECT} is 190.735 MB and the limit is 190.734 MB.")

    def test_uses_the_given_fix(self) -> None:
        error = check_size(2, maximum_bytes=1, subject=SUBJECT, fix="Turn on 'Auto Image Resize'.")
        assert error is not None
        assert str(error).endswith("Turn on 'Auto Image Resize'.")


class TestCheckDimensions:
    def test_allows_an_image_within_every_limit(self) -> None:
        info = ImageInfo(1280, 720, "image/png")
        assert check_dimensions(info, subject=SUBJECT, min_side=200, max_side=4096, max_megapixels=16) is None

    def test_skips_an_unknown_image(self) -> None:
        assert check_dimensions(None, subject=SUBJECT, min_side=10_000) is None

    def test_rejects_a_short_side(self) -> None:
        error = check_dimensions(ImageInfo(1000, 150, None), subject=SUBJECT, min_side=200)
        assert str(error) == f"{SUBJECT} is 1000x150 and each side must be at least 200 pixels. Use a larger one."

    def test_rejects_a_long_side(self) -> None:
        error = check_dimensions(ImageInfo(300, 5000, None), subject=SUBJECT, max_side=4096)
        assert str(error) == f"{SUBJECT} is 300x5000 and each side must be at most 4096 pixels. Resize it smaller."

    def test_rejects_too_many_pixels(self) -> None:
        error = check_dimensions(ImageInfo(4000, 4000, None), subject=SUBJECT, max_megapixels=8)
        assert str(error) == (
            f"{SUBJECT} is 4000x4000 (16.0 megapixels) and the limit is 8 megapixels. Resize it smaller."
        )

    def test_reports_megapixels_just_over_the_limit_as_over_them(self) -> None:
        error = check_dimensions(ImageInfo(2829, 2829, None), subject=SUBJECT, max_megapixels=8)
        assert "(8.003 megapixels)" in str(error)


class TestCheckAspectRatio:
    @pytest.mark.parametrize(("width", "height"), [(1920, 1080), (1366, 768), (720, 1280)])
    def test_allows_a_named_ratio(self, width: int, height: int) -> None:
        info = ImageInfo(width, height, None)
        assert check_aspect_ratio(info, subject=SUBJECT, allowed=("16:9", "9:16")) is None

    def test_rejects_an_unnamed_ratio(self) -> None:
        error = check_aspect_ratio(ImageInfo(1000, 1000, None), subject=SUBJECT, allowed=("16:9", "9:16"))
        assert str(error) == f"{SUBJECT} is 1000x1000, which isn't a supported aspect ratio. Crop it to 16:9 or 9:16."

    def test_skips_an_unknown_image(self) -> None:
        assert check_aspect_ratio(None, subject=SUBJECT, allowed=("1:1",)) is None


class TestCheckAspectRatioRange:
    @pytest.mark.parametrize(("width", "height"), [(300, 900), (900, 300), (1000, 1000), (1920, 1080)])
    def test_allows_a_shape_in_range_including_the_ends(self, width: int, height: int) -> None:
        info = ImageInfo(width, height, None)
        assert check_aspect_ratio_range(info, subject=SUBJECT, widest="3:1", tallest="1:3") is None

    def test_rejects_a_shape_that_is_too_wide(self) -> None:
        error = check_aspect_ratio_range(ImageInfo(1000, 300, None), subject=SUBJECT, widest="3:1", tallest="1:3")
        assert str(error) == (
            f"{SUBJECT} is 1000x300, which is too wide. The aspect ratio must be between 1:3 and 3:1. Crop it."
        )

    def test_rejects_a_shape_that_is_too_tall(self) -> None:
        error = check_aspect_ratio_range(ImageInfo(250, 1000, None), subject=SUBJECT, widest="3:1", tallest="1:3")
        assert str(error) == (
            f"{SUBJECT} is 250x1000, which is too tall. The aspect ratio must be between 1:3 and 3:1. Crop it."
        )

    def test_names_the_bound_that_failed_for_a_portrait_only_range(self) -> None:
        error = check_aspect_ratio_range(ImageInfo(1000, 1000, None), subject=SUBJECT, widest="3:4", tallest="9:16")
        assert "which is too wide" in str(error)

    def test_skips_an_unknown_image(self) -> None:
        assert check_aspect_ratio_range(None, subject=SUBJECT, widest="3:1", tallest="1:3") is None


class TestCheckMimeType:
    def test_allows_a_listed_type(self) -> None:
        assert check_mime_type("image/png", subject=SUBJECT, allowed=("image/jpeg", "image/png")) is None

    def test_skips_an_unknown_type(self) -> None:
        assert check_mime_type(None, subject=SUBJECT, allowed=("image/png",)) is None

    def test_rejects_an_unlisted_type(self) -> None:
        error = check_mime_type("image/webp", subject=SUBJECT, allowed=("image/jpeg", "image/png"))
        assert str(error) == f"{SUBJECT} is a WEBP file. Convert it to JPEG or PNG."


class TestCheckDuration:
    def test_allows_a_duration_in_range(self) -> None:
        assert check_duration(5.0, subject="The video in 'Video'", minimum=1, maximum=8.7) is None

    def test_skips_an_unknown_duration(self) -> None:
        assert check_duration(None, subject="The video in 'Video'", maximum=1) is None

    def test_rejects_a_long_video(self) -> None:
        error = check_duration(12.345, subject="The video in 'Video'", maximum=8.7)
        assert str(error) == (
            "The video in 'Video' is 12.35 seconds long and the limit is 8.7. Use a Trim Video node to shorten it."
        )

    def test_rejects_a_short_video(self) -> None:
        error = check_duration(0.5, subject="The video in 'Video'", minimum=2)
        assert str(error) == "The video in 'Video' is 0.50 seconds long and the minimum is 2. Use a longer clip."

    def test_reports_a_duration_just_over_the_limit_as_over_it(self) -> None:
        error = check_duration(8.701, subject="The video in 'Video'", maximum=8.7)
        assert str(error).startswith("The video in 'Video' is 8.701 seconds long and the limit is 8.7.")

    def test_reports_a_duration_under_a_non_round_minimum_as_under_it(self) -> None:
        error = check_duration(1.9455, subject="The video in 'Video'", minimum=1.946)
        assert str(error).startswith("The video in 'Video' is 1.9455 seconds long and the minimum is 1.946.")

    def test_uses_the_given_fix(self) -> None:
        error = check_duration(20, subject="The video in 'Video'", maximum=10, fix="Pick a shorter clip.")
        assert str(error).endswith("Pick a shorter clip.")
