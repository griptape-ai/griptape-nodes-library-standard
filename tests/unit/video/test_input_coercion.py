"""Tests for ``_coerce_*_url_or_data_uri`` helpers across video nodes.

These helpers normalize various image/video/audio inputs (plain strings,
artifact objects, serialized artifact dicts) into a single string that is
safe to hand to ``File(...).aread_data_uri()``. The helpers MUST pass through
HTTP(S) URLs, ``data:`` URIs, project macro paths like ``{inputs}/foo.png``,
and plain filesystem paths without wrapping them as raw base64. Wrapping is
only applied to artifact ``.base64`` payloads (which are genuinely raw bytes).
"""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest
from griptape.artifacts import ImageUrlArtifact
from griptape.artifacts.video_url_artifact import VideoUrlArtifact

from griptape_nodes_library.three_d.tripo_image_to_3d_generation import TripoImageTo3DGeneration
from griptape_nodes_library.video.kling_image_to_video_generation import KlingImageToVideoGeneration
from griptape_nodes_library.video.kling_omni_video_generation import KlingOmniVideoGeneration
from griptape_nodes_library.video.ltx_audio_to_video_generation import LTXAudioToVideoGeneration
from griptape_nodes_library.video.ltx_image_to_video_generation import LTXImageToVideoGeneration
from griptape_nodes_library.video.ltx_video_retake import LTXVideoRetake
from griptape_nodes_library.video.minimax_hailuo_video_generation import MinimaxHailuoVideoGeneration
from griptape_nodes_library.video.omnihuman_subject_detection import OmnihumanSubjectDetection
from griptape_nodes_library.video.omnihuman_subject_recognition import OmnihumanSubjectRecognition
from griptape_nodes_library.video.omnihuman_video_generation import OmnihumanVideoGeneration
from griptape_nodes_library.video.seedance_2_0_video_generation import Seedance20VideoGeneration
from griptape_nodes_library.video.seedance_video_generation import SeedanceVideoGeneration
from griptape_nodes_library.video.wan_animate_generation import WanAnimateGeneration
from griptape_nodes_library.video.wan_image_to_video_generation import WanImageToVideoGeneration
from griptape_nodes_library.video.wan_reference_to_video_generation import WanReferenceToVideoGeneration
from griptape_nodes_library.video.wan_text_to_video_generation import WanTextToVideoGeneration

# ---------------------------------------------------------------------------
# Helper registries
# ---------------------------------------------------------------------------

# Every node class whose ``_coerce_image_url_or_data_uri`` was modified in this
# branch. Seedance 2.0 has dict-handling behavior that differs from the others
# and is covered separately.
IMAGE_HELPERS_GENERIC: list[tuple[str, Callable[[Any], str | None]]] = [
    ("KlingImageToVideoGeneration", KlingImageToVideoGeneration._coerce_image_url_or_data_uri),
    ("KlingOmniVideoGeneration", KlingOmniVideoGeneration._coerce_image_url_or_data_uri),
    ("LTXAudioToVideoGeneration", LTXAudioToVideoGeneration._coerce_image_url_or_data_uri),
    ("LTXImageToVideoGeneration", LTXImageToVideoGeneration._coerce_image_url_or_data_uri),
    ("MinimaxHailuoVideoGeneration", MinimaxHailuoVideoGeneration._coerce_image_url_or_data_uri),
    ("OmnihumanSubjectDetection", OmnihumanSubjectDetection._coerce_image_url_or_data_uri),
    ("OmnihumanSubjectRecognition", OmnihumanSubjectRecognition._coerce_image_url_or_data_uri),
    ("OmnihumanVideoGeneration", OmnihumanVideoGeneration._coerce_image_url_or_data_uri),
    ("SeedanceVideoGeneration", SeedanceVideoGeneration._coerce_image_url_or_data_uri),
    ("TripoImageTo3DGeneration", TripoImageTo3DGeneration._coerce_image_url_or_data_uri),
    ("WanAnimateGeneration", WanAnimateGeneration._coerce_image_url_or_data_uri),
]

VIDEO_HELPERS: list[tuple[str, Callable[[Any], str | None]]] = [
    ("LTXVideoRetake", LTXVideoRetake._coerce_video_url_or_data_uri),
    ("WanAnimateGeneration", WanAnimateGeneration._coerce_video_url_or_data_uri),
    ("WanReferenceToVideoGeneration", WanReferenceToVideoGeneration._coerce_video_url_or_data_uri),
]

AUDIO_HELPERS: list[tuple[str, Callable[[Any], str | None]]] = [
    ("LTXAudioToVideoGeneration", LTXAudioToVideoGeneration._coerce_audio_url_or_data_uri),
    ("OmnihumanVideoGeneration", OmnihumanVideoGeneration._coerce_audio_url_or_data_uri),
    ("WanImageToVideoGeneration", WanImageToVideoGeneration._coerce_audio_url_or_data_uri),
    ("WanReferenceToVideoGeneration", WanReferenceToVideoGeneration._coerce_audio_url_or_data_uri),
    ("WanTextToVideoGeneration", WanTextToVideoGeneration._coerce_audio_url_or_data_uri),
]


def _ids(helpers: list[tuple[str, Callable[[Any], str | None]]]) -> list[str]:
    return [name for name, _ in helpers]


# ---------------------------------------------------------------------------
# IMAGE helpers (excluding Seedance 2.0 which has special dict handling)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("_name", "helper"), IMAGE_HELPERS_GENERIC, ids=_ids(IMAGE_HELPERS_GENERIC))
class TestImageHelpers:
    def test_none_returns_none(self, _name: str, helper: Callable[[Any], str | None]) -> None:
        assert helper(None) is None

    @pytest.mark.parametrize("value", ["", "   ", "\t\n"])
    def test_empty_string_returns_none(self, _name: str, helper: Callable[[Any], str | None], value: str) -> None:
        assert helper(value) is None

    @pytest.mark.parametrize(
        "value",
        [
            "http://example.com/foo.png",
            "https://example.com/foo.png",
            "data:image/png;base64,QUJD",
            "data:image/jpeg;base64,QUJD",
        ],
    )
    def test_known_uri_strings_pass_through(self, _name: str, helper: Callable[[Any], str | None], value: str) -> None:
        assert helper(value) == value

    def test_strings_are_stripped(self, _name: str, helper: Callable[[Any], str | None]) -> None:
        assert helper("  https://example.com/foo.png  ") == "https://example.com/foo.png"

    @pytest.mark.parametrize(
        "value",
        [
            "{inputs}/foo.png",
            "{outputs}/color_bars.png",
            "{workspace}/scene/frame.jpeg",
        ],
    )
    def test_macro_paths_pass_through_unwrapped(
        self, _name: str, helper: Callable[[Any], str | None], value: str
    ) -> None:
        # Critical fix: macro paths must NOT be wrapped as raw base64 since
        # File() can resolve them downstream.
        assert helper(value) == value

    @pytest.mark.parametrize(
        "value",
        [
            "/tmp/foo.png",
            "/abs/path/to/image.jpg",
            "relative/path/foo.webp",
            "image.png",
        ],
    )
    def test_filesystem_paths_pass_through_unwrapped(
        self, _name: str, helper: Callable[[Any], str | None], value: str
    ) -> None:
        assert helper(value) == value

    def test_raw_string_is_not_wrapped_as_base64(self, _name: str, helper: Callable[[Any], str | None]) -> None:
        # Even a string that "looks like" base64 is no longer auto-wrapped:
        # if the caller wanted base64 data it would be on .base64 of an artifact.
        assert helper("QUJDREVG") == "QUJDREVG"

    def test_image_url_artifact_with_http_url(self, _name: str, helper: Callable[[Any], str | None]) -> None:
        artifact = ImageUrlArtifact("https://example.com/foo.png")
        assert helper(artifact) == "https://example.com/foo.png"

    def test_image_url_artifact_with_macro_path(self, _name: str, helper: Callable[[Any], str | None]) -> None:
        # The artifact .value branch previously gated on URL/data-URI prefix,
        # which caused macro-path artifacts to fall through to None.
        artifact = ImageUrlArtifact("{inputs}/foo.png")
        assert helper(artifact) == "{inputs}/foo.png"

    def test_image_url_artifact_with_filesystem_path(self, _name: str, helper: Callable[[Any], str | None]) -> None:
        artifact = ImageUrlArtifact("/abs/path/foo.png")
        assert helper(artifact) == "/abs/path/foo.png"

    def test_artifact_value_is_stripped(self, _name: str, helper: Callable[[Any], str | None]) -> None:
        artifact = SimpleNamespace(value="  {inputs}/foo.png  ", base64=None)
        assert helper(artifact) == "{inputs}/foo.png"

    def test_artifact_base64_raw_is_wrapped_as_data_uri(self, _name: str, helper: Callable[[Any], str | None]) -> None:
        artifact = SimpleNamespace(value=None, base64="RAW_IMAGE_B64")
        assert helper(artifact) == "data:image/png;base64,RAW_IMAGE_B64"

    def test_artifact_base64_already_data_uri_passes_through(
        self, _name: str, helper: Callable[[Any], str | None]
    ) -> None:
        artifact = SimpleNamespace(value=None, base64="data:image/jpeg;base64,RAW")
        assert helper(artifact) == "data:image/jpeg;base64,RAW"

    def test_artifact_value_takes_precedence_over_base64(self, _name: str, helper: Callable[[Any], str | None]) -> None:
        artifact = SimpleNamespace(value="https://example.com/foo.png", base64="UNUSED")
        assert helper(artifact) == "https://example.com/foo.png"

    def test_empty_artifact_returns_none(self, _name: str, helper: Callable[[Any], str | None]) -> None:
        artifact = SimpleNamespace(value="", base64="")
        assert helper(artifact) is None

    def test_artifact_with_no_known_attrs_returns_none(self, _name: str, helper: Callable[[Any], str | None]) -> None:
        # Plain object that exposes neither .value nor .base64.
        assert helper(SimpleNamespace()) is None

    def test_artifact_with_non_string_value_falls_back_to_base64(
        self, _name: str, helper: Callable[[Any], str | None]
    ) -> None:
        artifact = SimpleNamespace(value=123, base64="RAW")
        assert helper(artifact) == "data:image/png;base64,RAW"


# ---------------------------------------------------------------------------
# VIDEO helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("_name", "helper"), VIDEO_HELPERS, ids=_ids(VIDEO_HELPERS))
class TestVideoHelpers:
    def test_none_returns_none(self, _name: str, helper: Callable[[Any], str | None]) -> None:
        assert helper(None) is None

    @pytest.mark.parametrize("value", ["", "   "])
    def test_empty_string_returns_none(self, _name: str, helper: Callable[[Any], str | None], value: str) -> None:
        assert helper(value) is None

    @pytest.mark.parametrize(
        "value",
        [
            "http://example.com/foo.mp4",
            "https://example.com/foo.mp4",
            "data:video/mp4;base64,QUJD",
            "data:video/webm;base64,QUJD",
        ],
    )
    def test_known_uri_strings_pass_through(self, _name: str, helper: Callable[[Any], str | None], value: str) -> None:
        assert helper(value) == value

    @pytest.mark.parametrize(
        "value",
        ["{inputs}/foo.mp4", "{outputs}/render.mp4", "/tmp/clip.mov", "relative/clip.mp4"],
    )
    def test_macro_paths_and_filesystem_paths_pass_through_unwrapped(
        self, _name: str, helper: Callable[[Any], str | None], value: str
    ) -> None:
        assert helper(value) == value

    def test_video_url_artifact_with_http_url(self, _name: str, helper: Callable[[Any], str | None]) -> None:
        artifact = VideoUrlArtifact("https://example.com/foo.mp4")
        assert helper(artifact) == "https://example.com/foo.mp4"

    def test_video_url_artifact_with_macro_path(self, _name: str, helper: Callable[[Any], str | None]) -> None:
        artifact = VideoUrlArtifact("{inputs}/foo.mp4")
        assert helper(artifact) == "{inputs}/foo.mp4"

    def test_artifact_base64_raw_is_wrapped_as_video_data_uri(
        self, _name: str, helper: Callable[[Any], str | None]
    ) -> None:
        artifact = SimpleNamespace(value=None, base64="RAW_VIDEO_B64")
        assert helper(artifact) == "data:video/mp4;base64,RAW_VIDEO_B64"

    def test_artifact_base64_already_data_uri_passes_through(
        self, _name: str, helper: Callable[[Any], str | None]
    ) -> None:
        artifact = SimpleNamespace(value=None, base64="data:video/webm;base64,RAW")
        assert helper(artifact) == "data:video/webm;base64,RAW"

    def test_strings_are_stripped(self, _name: str, helper: Callable[[Any], str | None]) -> None:
        assert helper("  https://example.com/foo.mp4  ") == "https://example.com/foo.mp4"

    def test_artifact_with_no_known_attrs_returns_none(self, _name: str, helper: Callable[[Any], str | None]) -> None:
        assert helper(SimpleNamespace()) is None

    def test_dict_with_value_key(self, _name: str, helper: Callable[[Any], str | None]) -> None:
        # Serialized VideoUrlArtifact (dict form) arrives when a video is
        # uploaded directly to the node rather than fed via Load Video.
        assert helper({"value": "https://example.com/foo.mp4"}) == "https://example.com/foo.mp4"

    def test_dict_with_value_key_is_stripped(self, _name: str, helper: Callable[[Any], str | None]) -> None:
        assert helper({"value": "  https://example.com/foo.mp4  "}) == "https://example.com/foo.mp4"

    def test_dict_with_url_key_only(self, _name: str, helper: Callable[[Any], str | None]) -> None:
        assert helper({"url": "https://example.com/foo.mp4"}) == "https://example.com/foo.mp4"

    def test_dict_value_takes_precedence_over_url(self, _name: str, helper: Callable[[Any], str | None]) -> None:
        result = helper({"value": "https://example.com/value.mp4", "url": "https://example.com/url.mp4"})
        assert result == "https://example.com/value.mp4"

    def test_dict_with_neither_key_returns_none(self, _name: str, helper: Callable[[Any], str | None]) -> None:
        assert helper({"type": "VideoUrlArtifact"}) is None

    def test_dict_with_empty_value_and_url_returns_none(self, _name: str, helper: Callable[[Any], str | None]) -> None:
        assert helper({"value": "", "url": "   "}) is None


# ---------------------------------------------------------------------------
# AUDIO helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("_name", "helper"), AUDIO_HELPERS, ids=_ids(AUDIO_HELPERS))
class TestAudioHelpers:
    def test_none_returns_none(self, _name: str, helper: Callable[[Any], str | None]) -> None:
        assert helper(None) is None

    @pytest.mark.parametrize("value", ["", "   "])
    def test_empty_string_returns_none(self, _name: str, helper: Callable[[Any], str | None], value: str) -> None:
        assert helper(value) is None

    @pytest.mark.parametrize(
        "value",
        [
            "http://example.com/foo.mp3",
            "https://example.com/foo.wav",
            "data:audio/mpeg;base64,QUJD",
            "data:audio/wav;base64,QUJD",
        ],
    )
    def test_known_uri_strings_pass_through(self, _name: str, helper: Callable[[Any], str | None], value: str) -> None:
        assert helper(value) == value

    @pytest.mark.parametrize(
        "value",
        ["{inputs}/foo.mp3", "{outputs}/voice.wav", "/tmp/clip.mp3", "relative/clip.wav"],
    )
    def test_macro_paths_and_filesystem_paths_pass_through_unwrapped(
        self, _name: str, helper: Callable[[Any], str | None], value: str
    ) -> None:
        assert helper(value) == value

    def test_audio_url_artifact_with_macro_path(self, _name: str, helper: Callable[[Any], str | None]) -> None:
        artifact = SimpleNamespace(value="{inputs}/voice.mp3", base64=None)
        assert helper(artifact) == "{inputs}/voice.mp3"

    def test_artifact_base64_raw_is_wrapped_as_audio_data_uri(
        self, _name: str, helper: Callable[[Any], str | None]
    ) -> None:
        artifact = SimpleNamespace(value=None, base64="RAW_AUDIO_B64")
        assert helper(artifact) == "data:audio/mpeg;base64,RAW_AUDIO_B64"

    def test_artifact_base64_already_data_uri_passes_through(
        self, _name: str, helper: Callable[[Any], str | None]
    ) -> None:
        artifact = SimpleNamespace(value=None, base64="data:audio/wav;base64,RAW")
        assert helper(artifact) == "data:audio/wav;base64,RAW"

    def test_strings_are_stripped(self, _name: str, helper: Callable[[Any], str | None]) -> None:
        assert helper("  https://example.com/foo.mp3  ") == "https://example.com/foo.mp3"

    def test_artifact_value_with_path_passes_through(self, _name: str, helper: Callable[[Any], str | None]) -> None:
        # Plain filesystem path on .value (as Load Audio emits) must survive.
        artifact = SimpleNamespace(value="/abs/path/voice.wav", base64=None)
        assert helper(artifact) == "/abs/path/voice.wav"


# ---------------------------------------------------------------------------
# Seedance 2.0 image helper: dict path
# ---------------------------------------------------------------------------


class TestSeedance20ImageHelperDictPath:
    """Seedance 2.0 also accepts serialized artifact dicts as inputs."""

    helper = staticmethod(Seedance20VideoGeneration._coerce_image_url_or_data_uri)

    def test_dict_with_image_url_artifact_type_passes_value_through(self) -> None:
        result = self.helper({"type": "ImageUrlArtifact", "value": "https://example.com/foo.png"})
        assert result == "https://example.com/foo.png"

    def test_dict_with_image_url_artifact_type_passes_macro_path_through(self) -> None:
        result = self.helper({"type": "ImageUrlArtifact", "value": "{inputs}/foo.png"})
        assert result == "{inputs}/foo.png"

    def test_dict_with_image_url_artifact_type_strips_whitespace(self) -> None:
        result = self.helper({"type": "ImageUrlArtifact", "value": "  {inputs}/foo.png  "})
        assert result == "{inputs}/foo.png"

    def test_dict_with_url_value_passes_through_regardless_of_type(self) -> None:
        result = self.helper({"type": "SomethingElse", "value": "https://example.com/foo.png"})
        assert result == "https://example.com/foo.png"

    def test_dict_with_data_uri_value_passes_through(self) -> None:
        result = self.helper({"type": "ImageArtifact", "value": "data:image/png;base64,RAW"})
        assert result == "data:image/png;base64,RAW"

    def test_dict_with_image_artifact_wraps_with_format(self) -> None:
        result = self.helper({"type": "ImageArtifact", "value": "RAW", "format": "jpeg"})
        assert result == "data:image/jpeg;base64,RAW"

    def test_dict_with_image_artifact_defaults_to_png_format(self) -> None:
        result = self.helper({"type": "ImageArtifact", "value": "RAW"})
        assert result == "data:image/png;base64,RAW"

    def test_dict_with_image_artifact_lowercases_format(self) -> None:
        result = self.helper({"type": "ImageArtifact", "value": "RAW", "format": "JPEG"})
        assert result == "data:image/jpeg;base64,RAW"

    def test_dict_with_unknown_type_and_macro_path_passes_through(self) -> None:
        # This is the key fix for serialized artifacts with macro-path values.
        # Previously the dict branch only handled ImageUrlArtifact / known
        # data-URI prefixes / ImageArtifact, so an unknown type with a macro
        # path returned None.
        result = self.helper({"type": "RandomShape", "value": "{inputs}/foo.png"})
        assert result == "{inputs}/foo.png"

    def test_dict_with_unknown_type_and_filesystem_path_passes_through(self) -> None:
        result = self.helper({"type": "RandomShape", "value": "/tmp/foo.png"})
        assert result == "/tmp/foo.png"

    def test_dict_with_no_type_and_macro_path_passes_through(self) -> None:
        result = self.helper({"value": "{inputs}/foo.png"})
        assert result == "{inputs}/foo.png"

    def test_dict_with_empty_value_returns_none(self) -> None:
        assert self.helper({"type": "ImageArtifact", "value": ""}) is None
        assert self.helper({"type": "ImageArtifact", "value": "   "}) is None

    def test_dict_with_non_string_value_returns_none(self) -> None:
        assert self.helper({"type": "ImageArtifact", "value": 123}) is None
        assert self.helper({"type": "ImageArtifact", "value": None}) is None

    def test_dict_with_no_value_key_returns_none(self) -> None:
        assert self.helper({"type": "ImageArtifact"}) is None

    def test_object_with_to_dict_routes_through_dict_path(self) -> None:
        class Fake:
            def to_dict(self) -> dict:
                return {"type": "ImageUrlArtifact", "value": "{inputs}/foo.png"}

        assert self.helper(Fake()) == "{inputs}/foo.png"

    def test_object_to_dict_returning_non_dict_falls_back_to_attrs(self) -> None:
        class Fake:
            def to_dict(self) -> str:
                return "not-a-dict"

            value = "{inputs}/fallback.png"
            base64 = None

        assert self.helper(Fake()) == "{inputs}/fallback.png"

    def test_object_with_failing_to_dict_returns_none_safely(self) -> None:
        # The outer try/except swallows the exception; we just need to assert
        # the helper does not propagate it. (The .value fallback only runs when
        # to_dict either is missing or returns a dict that does not coerce.)
        class Fake:
            def to_dict(self) -> dict:
                raise RuntimeError("boom")

        assert self.helper(Fake()) is None

    def test_plain_string_macro_path_passes_through(self) -> None:
        # Confirms the non-dict, non-artifact string branch matches the
        # generic helper behavior.
        assert self.helper("{inputs}/foo.png") == "{inputs}/foo.png"


# ---------------------------------------------------------------------------
# LTXVideoRetake: _extract_video_url delegates to _coerce_video_url_or_data_uri
# ---------------------------------------------------------------------------


class TestLTXVideoRetakeExtractVideoUrl:
    """``_extract_video_url`` is a thin wrapper that previously str()'d unknown
    inputs (which would mangle artifact objects). It now delegates to the new
    ``_coerce_video_url_or_data_uri`` helper.
    """

    def test_extracts_url_from_video_url_artifact(self) -> None:
        node = LTXVideoRetake.__new__(LTXVideoRetake)  # bypass __init__
        artifact = VideoUrlArtifact("https://example.com/foo.mp4")
        assert node._extract_video_url(artifact) == "https://example.com/foo.mp4"

    def test_extracts_macro_path_from_video_url_artifact(self) -> None:
        node = LTXVideoRetake.__new__(LTXVideoRetake)
        artifact = VideoUrlArtifact("{inputs}/foo.mp4")
        assert node._extract_video_url(artifact) == "{inputs}/foo.mp4"

    def test_extracts_plain_string_macro_path(self) -> None:
        node = LTXVideoRetake.__new__(LTXVideoRetake)
        assert node._extract_video_url("{inputs}/foo.mp4") == "{inputs}/foo.mp4"

    def test_extracts_plain_string_url(self) -> None:
        node = LTXVideoRetake.__new__(LTXVideoRetake)
        assert node._extract_video_url("https://example.com/foo.mp4") == "https://example.com/foo.mp4"

    def test_returns_none_for_none(self) -> None:
        node = LTXVideoRetake.__new__(LTXVideoRetake)
        assert node._extract_video_url(None) is None

    def test_returns_none_for_empty_string(self) -> None:
        node = LTXVideoRetake.__new__(LTXVideoRetake)
        assert node._extract_video_url("") is None

    def test_returns_none_for_whitespace_string(self) -> None:
        node = LTXVideoRetake.__new__(LTXVideoRetake)
        assert node._extract_video_url("   ") is None

    def test_extracts_value_from_generic_artifact_with_value_attr(self) -> None:
        node = LTXVideoRetake.__new__(LTXVideoRetake)
        artifact = SimpleNamespace(value="{inputs}/foo.mp4", base64=None)
        assert node._extract_video_url(artifact) == "{inputs}/foo.mp4"

    def test_wraps_raw_base64_from_video_artifact(self) -> None:
        node = LTXVideoRetake.__new__(LTXVideoRetake)
        artifact = SimpleNamespace(value=None, base64="RAW_VIDEO_B64")
        assert node._extract_video_url(artifact) == "data:video/mp4;base64,RAW_VIDEO_B64"


# ---------------------------------------------------------------------------
# LTXVideoRetake: _prepare_video_data_uri_async passes macro paths to File()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestLTXVideoRetakePrepareDataUriAsync:
    """Verifies that the helper output threads correctly into ``File()`` so
    macro paths and filesystem paths are resolved by File rather than wrapped
    as raw base64 (the original bug).
    """

    async def test_macro_path_string_is_handed_to_file(self, monkeypatch: pytest.MonkeyPatch) -> None:
        node = LTXVideoRetake.__new__(LTXVideoRetake)
        captured: dict[str, str] = {}

        from griptape_nodes.files import file as file_module

        original_init = file_module.File.__init__

        def capture_init(self: file_module.File, path: str, *args: Any, **kwargs: Any) -> None:
            captured["path"] = path
            original_init(self, path, *args, **kwargs)

        async def fake_aread(self: file_module.File, fallback_mime: str = "application/octet-stream") -> str:
            return "data:video/mp4;base64,RESOLVED"

        monkeypatch.setattr(file_module.File, "__init__", capture_init)
        monkeypatch.setattr(file_module.File, "aread_data_uri", fake_aread)

        result = await node._prepare_video_data_uri_async("{inputs}/clip.mp4")
        assert result == "data:video/mp4;base64,RESOLVED"
        assert captured["path"] == "{inputs}/clip.mp4"

    async def test_filesystem_path_string_is_handed_to_file(self, monkeypatch: pytest.MonkeyPatch) -> None:
        node = LTXVideoRetake.__new__(LTXVideoRetake)
        captured: dict[str, str] = {}

        from griptape_nodes.files import file as file_module

        original_init = file_module.File.__init__

        def capture_init(self: file_module.File, path: str, *args: Any, **kwargs: Any) -> None:
            captured["path"] = path
            original_init(self, path, *args, **kwargs)

        async def fake_aread(self: file_module.File, fallback_mime: str = "application/octet-stream") -> str:
            return "data:video/mp4;base64,RESOLVED"

        monkeypatch.setattr(file_module.File, "__init__", capture_init)
        monkeypatch.setattr(file_module.File, "aread_data_uri", fake_aread)

        result = await node._prepare_video_data_uri_async("/tmp/clip.mp4")
        assert result == "data:video/mp4;base64,RESOLVED"
        assert captured["path"] == "/tmp/clip.mp4"

    async def test_video_url_artifact_with_macro_path_is_handed_to_file(self, monkeypatch: pytest.MonkeyPatch) -> None:
        node = LTXVideoRetake.__new__(LTXVideoRetake)
        captured: dict[str, str] = {}

        from griptape_nodes.files import file as file_module

        original_init = file_module.File.__init__

        def capture_init(self: file_module.File, path: str, *args: Any, **kwargs: Any) -> None:
            captured["path"] = path
            original_init(self, path, *args, **kwargs)

        async def fake_aread(self: file_module.File, fallback_mime: str = "application/octet-stream") -> str:
            return "data:video/mp4;base64,RESOLVED"

        monkeypatch.setattr(file_module.File, "__init__", capture_init)
        monkeypatch.setattr(file_module.File, "aread_data_uri", fake_aread)

        result = await node._prepare_video_data_uri_async(VideoUrlArtifact("{inputs}/clip.mp4"))
        assert result == "data:video/mp4;base64,RESOLVED"
        assert captured["path"] == "{inputs}/clip.mp4"

    async def test_data_uri_input_is_returned_without_calling_file(self, monkeypatch: pytest.MonkeyPatch) -> None:
        node = LTXVideoRetake.__new__(LTXVideoRetake)

        from griptape_nodes.files import file as file_module

        async def fail_aread(self: file_module.File, fallback_mime: str = "application/octet-stream") -> str:
            raise AssertionError("File.aread_data_uri must not be called for data: URIs")

        monkeypatch.setattr(file_module.File, "aread_data_uri", fail_aread)

        result = await node._prepare_video_data_uri_async("data:video/mp4;base64,ALREADY_BASE64")
        assert result == "data:video/mp4;base64,ALREADY_BASE64"

    async def test_returns_none_for_falsy_input(self) -> None:
        node = LTXVideoRetake.__new__(LTXVideoRetake)
        assert await node._prepare_video_data_uri_async(None) is None
        assert await node._prepare_video_data_uri_async("") is None
        assert await node._prepare_video_data_uri_async(0) is None

    async def test_returns_none_when_helper_resolves_to_none(self) -> None:
        node = LTXVideoRetake.__new__(LTXVideoRetake)
        # A non-falsy input that the coercer cannot resolve to a string.
        assert await node._prepare_video_data_uri_async(SimpleNamespace()) is None

    async def test_file_load_error_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        node = LTXVideoRetake.__new__(LTXVideoRetake)
        node.name = "LTXVideoRetake"

        from griptape_nodes.files import file as file_module
        from griptape_nodes.retained_mode.events.os_events import FileIOFailureReason

        async def fail_aread(self: file_module.File, fallback_mime: str = "application/octet-stream") -> str:
            raise file_module.FileLoadError(FileIOFailureReason.FILE_NOT_FOUND, "nope")

        monkeypatch.setattr(file_module.File, "aread_data_uri", fail_aread)

        assert await node._prepare_video_data_uri_async("{inputs}/missing.mp4") is None


# ---------------------------------------------------------------------------
# WAN response: _extract_result_video_url / _extract_video_url
# ---------------------------------------------------------------------------


class TestWanReferenceExtractResultVideoUrl:
    """The WAN proxy nests successful generations under ``output.video_url``;
    older or flatter responses may put the URL at the top level. The helper
    must check the nested location first and fall back to the top-level key,
    or successful generations silently produce no output (charging credits
    for videos the user cannot see).
    """

    helper = staticmethod(WanReferenceToVideoGeneration._extract_result_video_url)

    def test_extracts_nested_output_video_url(self) -> None:
        result = self.helper({"output": {"video_url": "https://example.com/foo.mp4"}})
        assert result == "https://example.com/foo.mp4"

    def test_extracts_top_level_video_url(self) -> None:
        result = self.helper({"video_url": "https://example.com/foo.mp4"})
        assert result == "https://example.com/foo.mp4"

    def test_nested_takes_precedence_over_top_level(self) -> None:
        result = self.helper(
            {
                "output": {"video_url": "https://example.com/nested.mp4"},
                "video_url": "https://example.com/top.mp4",
            }
        )
        assert result == "https://example.com/nested.mp4"

    def test_falls_back_to_top_level_when_nested_missing(self) -> None:
        result = self.helper({"output": {}, "video_url": "https://example.com/top.mp4"})
        assert result == "https://example.com/top.mp4"

    def test_falls_back_to_top_level_when_nested_value_invalid(self) -> None:
        # Non-http nested value should not be returned; the helper should keep
        # looking and use the valid top-level URL instead.
        result = self.helper(
            {
                "output": {"video_url": "not-a-url"},
                "video_url": "https://example.com/top.mp4",
            }
        )
        assert result == "https://example.com/top.mp4"

    def test_returns_none_for_none(self) -> None:
        assert self.helper(None) is None

    def test_returns_none_for_empty_dict(self) -> None:
        assert self.helper({}) is None

    def test_returns_none_when_neither_present(self) -> None:
        assert self.helper({"task_status": "SUCCEEDED"}) is None

    def test_returns_none_when_output_is_not_dict(self) -> None:
        assert self.helper({"output": "https://example.com/foo.mp4"}) is None

    def test_returns_none_when_url_is_non_http(self) -> None:
        assert self.helper({"output": {"video_url": "ftp://example.com/foo.mp4"}}) is None
        assert self.helper({"video_url": "ftp://example.com/foo.mp4"}) is None


class TestWanAnimateExtractVideoUrl:
    """``WanAnimateGeneration`` calls the same WAN proxy as the reference node,
    so it must also handle ``output.video_url``. The historical
    ``results.video_url`` shape and a plain top-level ``video_url`` are kept
    as fallbacks so existing responses continue to work.
    """

    helper = staticmethod(WanAnimateGeneration._extract_video_url)

    def test_extracts_nested_output_video_url(self) -> None:
        result = self.helper({"output": {"video_url": "https://example.com/foo.mp4"}})
        assert result == "https://example.com/foo.mp4"

    def test_extracts_results_video_url(self) -> None:
        result = self.helper({"results": {"video_url": "https://example.com/foo.mp4"}})
        assert result == "https://example.com/foo.mp4"

    def test_extracts_top_level_video_url(self) -> None:
        result = self.helper({"video_url": "https://example.com/foo.mp4"})
        assert result == "https://example.com/foo.mp4"

    def test_output_takes_precedence_over_results_and_top_level(self) -> None:
        result = self.helper(
            {
                "output": {"video_url": "https://example.com/output.mp4"},
                "results": {"video_url": "https://example.com/results.mp4"},
                "video_url": "https://example.com/top.mp4",
            }
        )
        assert result == "https://example.com/output.mp4"

    def test_returns_none_for_none(self) -> None:
        assert self.helper(None) is None

    def test_returns_none_when_nothing_present(self) -> None:
        assert self.helper({"task_status": "SUCCEEDED"}) is None
