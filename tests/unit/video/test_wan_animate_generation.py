from __future__ import annotations

import pytest
from griptape_nodes.exe_types.param_components.artifact_url.public_artifact_url_parameter import (
    PublicArtifactUrlParameter,
)

import griptape_nodes_library.video.wan_animate_generation as wan_animate_module
from griptape_nodes_library.video.wan_animate_generation import WanAnimateGeneration


@pytest.fixture(autouse=True)
def stub_public_artifact_bucket_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        PublicArtifactUrlParameter, "_get_bucket_id", staticmethod(lambda *_args, **_kwargs: "test-bucket")
    )


def _stub_video_duration(monkeypatch: pytest.MonkeyPatch, seconds: float = 4.0) -> None:
    async def fake_get_video_duration(_url: str) -> float:
        return seconds

    monkeypatch.setattr(wan_animate_module, "get_video_duration", fake_get_video_duration)


@pytest.mark.asyncio
async def test_build_payload_uploads_inputs_to_public_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    """The DashScope Animate endpoint forwards ``image_url`` / ``video_url``
    straight through; sending ``data:`` URIs trips
    ``InvalidVideo.FileFormat``. ``_build_payload`` must hand the provider
    public HTTP URLs produced by the ``PublicArtifactUrlParameter`` helpers.
    """
    node = WanAnimateGeneration(name="WanAnimate")
    node.set_parameter_value("image_url", "{inputs}/source.jpg")
    node.set_parameter_value("video_url", "{inputs}/reference.mp4")
    _stub_video_duration(monkeypatch, seconds=4.2)

    monkeypatch.setattr(
        node._public_image_url_parameter,
        "get_public_url_for_parameter",
        lambda: "https://public.example/source.jpg",
    )
    monkeypatch.setattr(
        node._public_video_url_parameter,
        "get_public_url_for_parameter",
        lambda: "https://public.example/reference.mp4",
    )

    payload = await node._build_payload()

    assert payload == {
        "input": {
            "image_url": "https://public.example/source.jpg",
            "video_url": "https://public.example/reference.mp4",
        },
        "parameters": {
            # 4.2s is rounded up to 5 by ``math.ceil``.
            "mode": "wan-std",
            "duration": 5,
        },
    }


@pytest.mark.asyncio
async def test_build_payload_does_not_send_data_uri(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression guard against re-introducing the ``prepare_media_data_uri``
    flow: the payload must contain HTTP(S) URLs, never ``data:`` URIs.
    """
    node = WanAnimateGeneration(name="WanAnimate")
    node.set_parameter_value("image_url", "{inputs}/source.jpg")
    node.set_parameter_value("video_url", "{inputs}/reference.mp4")
    _stub_video_duration(monkeypatch)

    monkeypatch.setattr(
        node._public_image_url_parameter,
        "get_public_url_for_parameter",
        lambda: "https://public.example/source.jpg",
    )
    monkeypatch.setattr(
        node._public_video_url_parameter,
        "get_public_url_for_parameter",
        lambda: "https://public.example/reference.mp4",
    )

    payload = await node._build_payload()

    for key in ("image_url", "video_url"):
        assert payload["input"][key].startswith(("http://", "https://"))
        assert not payload["input"][key].startswith("data:")


@pytest.mark.asyncio
async def test_build_payload_raises_when_image_upload_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    node = WanAnimateGeneration(name="WanAnimate")
    node.set_parameter_value("image_url", "{inputs}/source.jpg")
    node.set_parameter_value("video_url", "{inputs}/reference.mp4")
    _stub_video_duration(monkeypatch)

    monkeypatch.setattr(node._public_image_url_parameter, "get_public_url_for_parameter", lambda: "")
    monkeypatch.setattr(
        node._public_video_url_parameter,
        "get_public_url_for_parameter",
        lambda: "https://public.example/reference.mp4",
    )

    with pytest.raises(ValueError, match="Failed to upload input image"):
        await node._build_payload()


@pytest.mark.asyncio
async def test_build_payload_raises_when_video_upload_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    node = WanAnimateGeneration(name="WanAnimate")
    node.set_parameter_value("image_url", "{inputs}/source.jpg")
    node.set_parameter_value("video_url", "{inputs}/reference.mp4")
    _stub_video_duration(monkeypatch)

    monkeypatch.setattr(
        node._public_image_url_parameter,
        "get_public_url_for_parameter",
        lambda: "https://public.example/source.jpg",
    )
    monkeypatch.setattr(node._public_video_url_parameter, "get_public_url_for_parameter", lambda: "")

    with pytest.raises(ValueError, match="Failed to upload input video"):
        await node._build_payload()


@pytest.mark.asyncio
async def test_build_payload_passes_through_existing_public_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    """``PublicArtifactUrlParameter`` short-circuits when the parameter is
    already a public HTTP URL (no upload, no cleanup). Verify the payload still
    carries those URLs verbatim.
    """
    node = WanAnimateGeneration(name="WanAnimate")
    node.set_parameter_value("image_url", "https://cdn.example/already-public.jpg")
    node.set_parameter_value("video_url", "https://cdn.example/already-public.mp4")
    node.set_parameter_value("mode", "wan-pro")
    _stub_video_duration(monkeypatch, seconds=10.0)

    monkeypatch.setattr(
        node._public_image_url_parameter,
        "get_public_url_for_parameter",
        lambda: "https://cdn.example/already-public.jpg",
    )
    monkeypatch.setattr(
        node._public_video_url_parameter,
        "get_public_url_for_parameter",
        lambda: "https://cdn.example/already-public.mp4",
    )

    payload = await node._build_payload()

    assert payload["input"] == {
        "image_url": "https://cdn.example/already-public.jpg",
        "video_url": "https://cdn.example/already-public.mp4",
    }
    assert payload["parameters"]["mode"] == "wan-pro"
    assert payload["parameters"]["duration"] == 10


@pytest.mark.asyncio
async def test_build_payload_normalizes_serialized_artifact_dicts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Direct uploads through the media-upload badge land as serialized
    ``VideoUrlArtifact`` dicts rather than artifact instances. The node must
    coerce the dict to its inner ``value`` so ``File()`` and the upload
    helper can consume it; otherwise ``File`` parses the dict's repr as a
    macro path and reports "missing required variables".
    """
    node = WanAnimateGeneration(name="WanAnimate")
    node.set_parameter_value(
        "image_url",
        {
            "value": "/abs/path/source.png",
            "name": "source.png",
            "type": "ImageUrlArtifact",
        },
    )
    node.set_parameter_value(
        "video_url",
        {
            "value": "/abs/path/reference.mp4",
            "width": 1280,
            "height": 720,
            "duration": 6.041992,
            "name": "reference.mp4",
            "type": "VideoUrlArtifact",
        },
    )

    seen_duration_url: list[str] = []

    async def fake_get_video_duration(url: str) -> float:
        seen_duration_url.append(url)
        return 6.041992

    monkeypatch.setattr(wan_animate_module, "get_video_duration", fake_get_video_duration)
    monkeypatch.setattr(
        node._public_image_url_parameter,
        "get_public_url_for_parameter",
        lambda: "https://public.example/source.png",
    )
    monkeypatch.setattr(
        node._public_video_url_parameter,
        "get_public_url_for_parameter",
        lambda: "https://public.example/reference.mp4",
    )

    payload = await node._build_payload()

    # The dict was unwrapped to its inner string before duration probing.
    assert seen_duration_url == ["/abs/path/reference.mp4"]
    # Both parameters are now plain strings the upload helpers can consume.
    assert node.get_parameter_value("image_url") == "/abs/path/source.png"
    assert node.get_parameter_value("video_url") == "/abs/path/reference.mp4"
    assert payload["input"] == {
        "image_url": "https://public.example/source.png",
        "video_url": "https://public.example/reference.mp4",
    }
    assert payload["parameters"]["duration"] == 7  # math.ceil(6.041992)


@pytest.mark.asyncio
async def test_build_payload_raises_when_video_dict_has_no_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """A serialized-dict shape with no usable inner string should fail with
    a clear "Video URL must be provided" error rather than a misleading
    macro-resolution error from ``File()``.
    """
    node = WanAnimateGeneration(name="WanAnimate")
    node.set_parameter_value("image_url", "https://public.example/source.png")
    node.set_parameter_value("video_url", {"type": "VideoUrlArtifact"})
    _stub_video_duration(monkeypatch)

    monkeypatch.setattr(
        node._public_image_url_parameter,
        "get_public_url_for_parameter",
        lambda: "https://public.example/source.png",
    )

    with pytest.raises(ValueError, match="Video URL must be provided"):
        await node._build_payload()


class TestExtractVideoUrl:
    """DashScope's documented WAN Animate response nests the result URL under
    ``output.results.video_url``. Older/sibling response shapes expose it at
    other locations; the extractor must keep working across all of them so
    successful generations don't silently produce "no video URL was found".
    """

    helper = staticmethod(WanAnimateGeneration._extract_video_url)

    def test_dashscope_documented_shape(self) -> None:
        # https://help.aliyun.com/zh/model-studio/wan2-2-animate-mix-api
        response = {
            "request_id": "a67f8716-18ef-447c-a286-xxxxxx",
            "output": {
                "task_id": "0385dc79-5ff8-4d82-bcb6-xxxxxx",
                "task_status": "SUCCEEDED",
                "submit_time": "2025-09-18 15:32:00.105",
                "scheduled_time": "2025-09-18 15:32:15.066",
                "end_time": "2025-09-18 15:34:41.898",
                "results": {"video_url": "https://dashscope.example/result.mp4"},
            },
            "usage": {"video_duration": 5.2, "video_ratio": "standard"},
        }
        assert self.helper(response) == "https://dashscope.example/result.mp4"

    def test_falls_back_to_output_video_url(self) -> None:
        # Earlier WAN Animate response variants put video_url directly on output.
        assert self.helper({"output": {"video_url": "https://example.com/foo.mp4"}}) == "https://example.com/foo.mp4"

    def test_falls_back_to_top_level_results_video_url(self) -> None:
        assert self.helper({"results": {"video_url": "https://example.com/foo.mp4"}}) == "https://example.com/foo.mp4"

    def test_falls_back_to_top_level_video_url(self) -> None:
        assert self.helper({"video_url": "https://example.com/foo.mp4"}) == "https://example.com/foo.mp4"

    def test_nested_results_takes_precedence(self) -> None:
        # When DashScope returns the documented shape ``output.results.video_url``
        # we must use that and not any stale value at older locations.
        response = {
            "output": {
                "results": {"video_url": "https://example.com/nested.mp4"},
                "video_url": "https://example.com/old.mp4",
            },
            "video_url": "https://example.com/top.mp4",
        }
        assert self.helper(response) == "https://example.com/nested.mp4"

    def test_returns_none_for_none(self) -> None:
        assert self.helper(None) is None

    def test_returns_none_for_empty_dict(self) -> None:
        assert self.helper({}) is None

    def test_returns_none_for_non_http_url(self) -> None:
        assert self.helper({"output": {"results": {"video_url": "ftp://example.com/foo.mp4"}}}) is None

    def test_returns_none_when_results_is_not_dict(self) -> None:
        # The Wan I2V endpoint puts results at the top level as a dict; if a
        # caller hands us a list shape we must not crash.
        assert self.helper({"output": {"results": ["not-a-dict"]}}) is None


class TestExtractStatus:
    """DashScope nests ``task_status`` under ``output``. ``_extract_status``
    must look there first; otherwise FAILED tasks fall through to the
    "no video URL was found" branch instead of surfacing the provider error.
    """

    helper = staticmethod(WanAnimateGeneration._extract_status)

    def test_extracts_nested_output_task_status(self) -> None:
        assert self.helper({"output": {"task_status": "SUCCEEDED"}}) == "SUCCEEDED"

    def test_extracts_failed_status(self) -> None:
        assert self.helper({"output": {"task_status": "FAILED"}}) == "FAILED"

    def test_falls_back_to_top_level(self) -> None:
        assert self.helper({"task_status": "SUCCEEDED"}) == "SUCCEEDED"

    def test_returns_none_for_missing(self) -> None:
        assert self.helper({"output": {"foo": "bar"}}) is None
        assert self.helper(None) is None
