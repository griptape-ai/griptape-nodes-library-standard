from __future__ import annotations

import pytest
from griptape.artifacts.video_url_artifact import VideoUrlArtifact
from griptape_nodes.exe_types.param_components.artifact_url.public_artifact_url_parameter import (
    PublicArtifactUrlParameter,
)

import griptape_nodes_library.video.runway_ml_character_performance as runway_module
from griptape_nodes_library.video.runway_ml_character_performance import RunwayMLCharacterPerformance


@pytest.fixture(autouse=True)
def stub_public_artifact_bucket_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        PublicArtifactUrlParameter, "_get_bucket_id", staticmethod(lambda *_args, **_kwargs: "test-bucket")
    )


def _stub_duration(monkeypatch: pytest.MonkeyPatch, seconds: float) -> None:
    def fake_get_video_duration_sync(_url: str) -> float:
        return seconds

    monkeypatch.setattr(runway_module, "get_video_duration_sync", fake_get_video_duration_sync)


def _build_node() -> RunwayMLCharacterPerformance:
    return RunwayMLCharacterPerformance(name="RunwayCharPerf")


@pytest.mark.parametrize("seconds", [3.0, 10.5, 30.0])
def test_validate_reference_duration_accepts_in_range(monkeypatch: pytest.MonkeyPatch, seconds: float) -> None:
    _stub_duration(monkeypatch, seconds=seconds)
    node = _build_node()
    error = node._validate_reference_duration(VideoUrlArtifact("https://example.com/ref.mp4"))
    assert error is None


@pytest.mark.parametrize("seconds", [0.5, 1.9])
def test_validate_reference_duration_rejects_too_short(monkeypatch: pytest.MonkeyPatch, seconds: float) -> None:
    _stub_duration(monkeypatch, seconds=seconds)
    node = _build_node()
    error = node._validate_reference_duration(VideoUrlArtifact("https://example.com/ref.mp4"))
    assert isinstance(error, ValueError)
    assert "reference_video must be between" in str(error)


def test_validate_reference_duration_rejects_too_long(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_duration(monkeypatch, seconds=45.0)
    node = _build_node()
    error = node._validate_reference_duration(VideoUrlArtifact("https://example.com/ref.mp4"))
    assert isinstance(error, ValueError)
    assert "reference_video must be between" in str(error)


def test_validate_reference_duration_skips_when_probe_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """ffprobe failures are best-effort; the validator should swallow them and let
    server-side validation catch out-of-range cases."""

    def boom(_url: str) -> float:
        msg = "ffprobe not available"
        raise ValueError(msg)

    monkeypatch.setattr(runway_module, "get_video_duration_sync", boom)
    node = _build_node()
    error = node._validate_reference_duration(VideoUrlArtifact("https://example.com/ref.mp4"))
    assert error is None


def test_validate_reference_duration_skips_when_value_missing() -> None:
    node = _build_node()

    class _Missing:
        value = None

    assert node._validate_reference_duration(_Missing()) is None
