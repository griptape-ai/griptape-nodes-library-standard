from __future__ import annotations

import pytest

from griptape_nodes_library.utils.ffmpeg_utils import (
    ColorDetails,
    Dimensions,
    FileDetails,
    FrameDetails,
    VideoMetadata,
)
from griptape_nodes_library.video.topaz_video_upscale import (
    MAX_STARLIGHT_FRAMES,
    MODEL_MAPPING,
    STARLIGHT_1080P_MAX_PIXELS,
    STARLIGHT_MAX_OUTPUT_PIXELS,
    ResizeMode,
    TopazVideoUpscale,
)


def _metadata(
    *,
    width: int = 960,
    height: int = 540,
    nb_frames: int | None = 300,
    duration: float | None = 10.0,
    frame_rate: float = 30.0,
    file_size: int | None = 1234,
) -> VideoMetadata:
    return VideoMetadata(
        file_details=FileDetails(
            codec_name="h264",
            codec_type="video",
            optional_duration=duration,
            optional_file_size=file_size,
        ),
        dimensions=Dimensions(
            width=width,
            height=height,
            aspect_ratio_decimal=width / height,
            aspect_ratio_string="16:9",
        ),
        color_details=ColorDetails(),
        frame_details=FrameDetails(
            r_frame_rate=f"{int(frame_rate)}/1",
            avg_frame_rate=f"{int(frame_rate)}/1",
            time_base="1/30000",
            frame_rate=frame_rate,
            optional_nb_frames=nb_frames,
        ),
    )


def _node(name: str = "TopazVideoUpscale") -> TopazVideoUpscale:
    return TopazVideoUpscale(name=name)


def _stub_probe(monkeypatch: pytest.MonkeyPatch, metadata: VideoMetadata) -> None:
    monkeypatch.setattr(
        TopazVideoUpscale,
        "_probe_source",
        lambda self, video_input: metadata,  # noqa: ARG005
    )


PUBLIC_URL = "https://acct.blob.core.windows.net/bucket/clip.mp4?sig=abc"


def _stub_public_url(node: TopazVideoUpscale, monkeypatch: pytest.MonkeyPatch, value: str | None = PUBLIC_URL) -> None:
    """Stand in for the upload to Griptape Cloud storage.

    The real call presigns a URL over the network; the node only cares that it
    gets one back to hand to Topaz as `source.external`.
    """
    monkeypatch.setattr(
        node._public_video_url_parameter,
        "get_public_url_for_parameter",
        lambda: value,
    )


# -- output resolution -------------------------------------------------------


def test_percentage_200_doubles_the_source() -> None:
    node = _node()
    node.set_parameter_value("resize_mode", ResizeMode.PERCENTAGE)
    node.set_parameter_value("percentage", 200)

    assert node._output_resolution(960, 540) == (1920, 1080)


def test_percentage_400_quadruples_the_source() -> None:
    node = _node()
    node.set_parameter_value("resize_mode", ResizeMode.PERCENTAGE)
    node.set_parameter_value("percentage", 400)

    assert node._output_resolution(960, 540) == (3840, 2160)


def test_width_and_height_uses_the_typed_dimensions() -> None:
    node = _node()
    node.set_parameter_value("resize_mode", ResizeMode.WIDTH_HEIGHT)
    node.set_parameter_value("target_width", 2560)
    node.set_parameter_value("target_height", 1440)

    assert node._output_resolution(960, 540) == (2560, 1440)


def test_width_only_preserves_aspect_ratio() -> None:
    node = _node()
    node.set_parameter_value("resize_mode", ResizeMode.WIDTH)
    node.set_parameter_value("target_size", 1920)

    assert node._output_resolution(960, 540) == (1920, 1080)


def test_height_only_preserves_aspect_ratio() -> None:
    node = _node()
    node.set_parameter_value("resize_mode", ResizeMode.HEIGHT)
    node.set_parameter_value("target_size", 1080)

    assert node._output_resolution(960, 540) == (1920, 1080)


def test_odd_dimensions_are_rounded_down_to_even() -> None:
    # Odd dimensions break yuv420 encoding, so they must never reach the provider.
    node = _node()
    node.set_parameter_value("resize_mode", ResizeMode.WIDTH_HEIGHT)
    node.set_parameter_value("target_width", 1921)
    node.set_parameter_value("target_height", 1081)

    assert node._output_resolution(960, 540) == (1920, 1080)


def test_odd_source_still_yields_even_output() -> None:
    node = _node()
    node.set_parameter_value("resize_mode", ResizeMode.PERCENTAGE)
    node.set_parameter_value("percentage", 200)

    width, height = node._output_resolution(961, 541)

    assert width % 2 == 0
    assert height % 2 == 0


def test_output_exceeding_the_hard_cap_raises() -> None:
    node = _node()
    node.set_parameter_value("resize_mode", ResizeMode.PERCENTAGE)
    node.set_parameter_value("percentage", 200)

    with pytest.raises(ValueError, match="hard limit"):
        # Doubling an already-4K source vastly exceeds the 3840x2160 cap.
        node._output_resolution(3840, 2160)


def _force_values(node: TopazVideoUpscale, monkeypatch: pytest.MonkeyPatch, **values: object) -> None:
    """Bypass the parameters' own validation to reach _output_resolution's guards.

    `min_val=2` clamps a zero width and the `Options` trait rejects an off-list
    mode, so `set_parameter_value` can never deliver either. The guards still
    exist to keep a bad value from reaching Topaz; this is how they get exercised.
    """
    original = node.get_parameter_value
    monkeypatch.setattr(
        node,
        "get_parameter_value",
        lambda name, *a, **kw: values.get(name, original(name, *a, **kw)),
    )


def test_width_and_height_without_dimensions_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    node = _node()
    _force_values(node, monkeypatch, resize_mode=ResizeMode.WIDTH_HEIGHT, target_width=0)

    with pytest.raises(ValueError, match="positive target width and height"):
        node._output_resolution(960, 540)


def test_unknown_resize_mode_raises_rather_than_defaulting(monkeypatch: pytest.MonkeyPatch) -> None:
    # The wildcard case must surface an unexpected value, not silently pick percentage.
    node = _node()
    _force_values(node, monkeypatch, resize_mode="8x")

    with pytest.raises(ValueError, match="Unknown resize mode"):
        node._output_resolution(960, 540)


# -- frame count -------------------------------------------------------------


def test_frame_count_prefers_nb_frames() -> None:
    assert TopazVideoUpscale._frame_count(_metadata(nb_frames=297)) == 297


def test_frame_count_falls_back_to_duration_times_rate() -> None:
    # ffprobe omits nb_frames for plenty of ordinary MP4s, and the proxy requires
    # frameCount with no default -- the fallback is what keeps those from 400ing.
    metadata = _metadata(nb_frames=None, duration=10.0, frame_rate=29.97)

    assert TopazVideoUpscale._frame_count(metadata) == 300  # ceil(299.7)


def test_frame_count_is_zero_when_nothing_is_derivable() -> None:
    metadata = _metadata(nb_frames=None, duration=None, frame_rate=0.0)

    assert TopazVideoUpscale._frame_count(metadata) == 0


# -- payload -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_build_payload_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    node = _node()
    node.set_parameter_value("video", "{inputs}/clip.mp4")
    node.set_parameter_value("resize_mode", ResizeMode.PERCENTAGE)
    node.set_parameter_value("percentage", 200)
    _stub_probe(monkeypatch, _metadata())
    _stub_public_url(node, monkeypatch)

    payload = await node._build_payload()

    assert payload["source"] == {
        "container": "mp4",
        "frameCount": 300,
        "frameRate": 30.0,
        "resolution": {"width": 960, "height": 540},
        "duration": 10.0,
        "size": 1234,
        "external": {"provider": "web-url", "presignedUrl": PUBLIC_URL},
    }
    assert payload["output"] == {"resolution": {"width": 1920, "height": 1080}}


@pytest.mark.asyncio
async def test_build_payload_sends_no_video_data(monkeypatch: pytest.MonkeyPatch) -> None:
    # The whole point of source.external: the video never travels in the body.
    node = _node()
    node.set_parameter_value("video", "{inputs}/clip.mp4")
    _stub_probe(monkeypatch, _metadata())
    _stub_public_url(node, monkeypatch)

    payload = await node._build_payload()

    assert "video" not in payload


@pytest.mark.asyncio
async def test_build_payload_wraps_the_url_in_a_web_url_descriptor(monkeypatch: pytest.MonkeyPatch) -> None:
    # The proxy rejects a bare string, and Topaz requires `provider` to be one of
    # r2/s3/web-url. `web-url` is the honest label for an Azure Blob SAS URL.
    node = _node()
    node.set_parameter_value("video", "{inputs}/clip.mp4")
    _stub_probe(monkeypatch, _metadata())
    _stub_public_url(node, monkeypatch)

    payload = await node._build_payload()

    assert payload["source"]["external"] == {"provider": "web-url", "presignedUrl": PUBLIC_URL}


@pytest.mark.asyncio
async def test_build_payload_omits_filters(monkeypatch: pytest.MonkeyPatch) -> None:
    # The proxy synthesizes filters from the routed model id. Sending our own only
    # risks the "filters[].model must match the requested model" rejection.
    node = _node()
    node.set_parameter_value("video", "{inputs}/clip.mp4")
    _stub_probe(monkeypatch, _metadata())
    _stub_public_url(node, monkeypatch)

    payload = await node._build_payload()

    assert "filters" not in payload


@pytest.mark.asyncio
async def test_build_payload_omits_absent_optional_source_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    node = _node()
    node.set_parameter_value("video", "{inputs}/clip.mp4")
    _stub_probe(monkeypatch, _metadata(duration=None, file_size=None))
    _stub_public_url(node, monkeypatch)

    payload = await node._build_payload()

    assert "duration" not in payload["source"]
    assert "size" not in payload["source"]
    assert payload["source"]["frameCount"] == 300


@pytest.mark.asyncio
async def test_build_payload_requires_a_video() -> None:
    node = _node()

    with pytest.raises(ValueError, match="requires an input video"):
        await node._build_payload()


@pytest.mark.asyncio
async def test_build_payload_rejects_underivable_frame_count(monkeypatch: pytest.MonkeyPatch) -> None:
    node = _node()
    node.set_parameter_value("video", "{inputs}/clip.mp4")
    _stub_probe(monkeypatch, _metadata(nb_frames=None, duration=None, frame_rate=0.0))
    _stub_public_url(node, monkeypatch)

    with pytest.raises(ValueError, match="could not determine the frame count"):
        await node._build_payload()


@pytest.mark.asyncio
async def test_build_payload_rejects_clips_over_the_frame_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    node = _node()
    node.set_parameter_value("video", "{inputs}/clip.mp4")
    _stub_probe(monkeypatch, _metadata(nb_frames=MAX_STARLIGHT_FRAMES + 1))
    _stub_public_url(node, monkeypatch)

    with pytest.raises(ValueError, match="over Starlight's"):
        await node._build_payload()


@pytest.mark.asyncio
async def test_rejected_input_is_never_uploaded(monkeypatch: pytest.MonkeyPatch) -> None:
    # Uploading costs a round trip and cloud storage, so the cheap local checks
    # have to run first. A clip over the frame cap must be rejected before it is
    # ever sent anywhere.
    node = _node()
    node.set_parameter_value("video", "{inputs}/clip.mp4")
    _stub_probe(monkeypatch, _metadata(nb_frames=MAX_STARLIGHT_FRAMES + 1))

    calls: list[None] = []

    def _record() -> str:
        calls.append(None)
        return PUBLIC_URL

    monkeypatch.setattr(node._public_video_url_parameter, "get_public_url_for_parameter", _record)

    with pytest.raises(ValueError, match="over Starlight's"):
        await node._build_payload()

    assert calls == []


@pytest.mark.asyncio
async def test_build_payload_raises_when_no_public_url_is_produced(monkeypatch: pytest.MonkeyPatch) -> None:
    node = _node()
    node.set_parameter_value("video", "{inputs}/clip.mp4")
    _stub_probe(monkeypatch, _metadata())
    _stub_public_url(node, monkeypatch, value=None)

    with pytest.raises(ValueError, match="could not produce a public URL"):
        await node._build_payload()


# -- model routing -----------------------------------------------------------


def test_model_id_is_the_full_prefixed_route() -> None:
    # The URL id keeps the "topaz-video-" prefix and the version dot; only the
    # body's filters[].model uses the bare code, which we do not send.
    node = _node()
    node.set_parameter_value("model", "Starlight Precise 2.5")

    assert node._get_api_model_id() == "topaz-video-slp-2.5"


def test_default_model_is_the_newer_starlight() -> None:
    assert _node()._get_api_model_id() == "topaz-video-slp-2.6"


def test_every_mapped_model_is_prefixed() -> None:
    for model_id in MODEL_MAPPING.values():
        assert model_id.startswith("topaz-video-")


# -- result parsing -----------------------------------------------------------


@pytest.mark.asyncio
async def test_parse_result_downloads_from_the_documented_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    # `download.url` is Topaz's documented shape for a completed generation.
    node = _node()
    captured: dict = {}

    async def fake_download_and_save(self, download_url, *_args, **_kwargs) -> None:  # noqa: ARG001
        captured["url"] = download_url

    monkeypatch.setattr(TopazVideoUpscale, "_download_and_save", fake_download_and_save)

    await node._parse_result(
        {"status": "complete", "download": {"url": "https://topaz.example/out.mp4"}}, "gen-1"
    )

    assert captured["url"] == "https://topaz.example/out.mp4"


# -- UI reactions ------------------------------------------------------------


def test_resize_mode_reveals_the_matching_fields() -> None:
    node = _node()
    target_size = node.get_parameter_by_name("target_size")
    target_width = node.get_parameter_by_name("target_width")
    target_height = node.get_parameter_by_name("target_height")
    percentage = node.get_parameter_by_name("percentage")
    assert target_size is not None
    assert target_width is not None
    assert target_height is not None
    assert percentage is not None

    # Default mode is percentage.
    assert target_size.ui_options.get("hide") is True
    assert target_width.ui_options.get("hide") is True
    assert target_height.ui_options.get("hide") is True
    assert percentage.ui_options.get("hide") is not True

    node.set_parameter_value("resize_mode", ResizeMode.WIDTH)
    assert target_size.ui_options.get("hide") is not True
    assert target_width.ui_options.get("hide") is True
    assert target_height.ui_options.get("hide") is True
    assert percentage.ui_options.get("hide") is True

    node.set_parameter_value("resize_mode", ResizeMode.HEIGHT)
    assert target_size.ui_options.get("hide") is not True
    assert target_width.ui_options.get("hide") is True
    assert target_height.ui_options.get("hide") is True
    assert percentage.ui_options.get("hide") is True

    node.set_parameter_value("resize_mode", ResizeMode.WIDTH_HEIGHT)
    assert target_size.ui_options.get("hide") is True
    assert target_width.ui_options.get("hide") is not True
    assert target_height.ui_options.get("hide") is not True
    assert percentage.ui_options.get("hide") is True

    node.set_parameter_value("resize_mode", ResizeMode.PERCENTAGE)
    assert target_size.ui_options.get("hide") is True
    assert target_width.ui_options.get("hide") is True
    assert target_height.ui_options.get("hide") is True
    assert percentage.ui_options.get("hide") is not True


def test_model_carries_a_cost_badge() -> None:
    # Starlight is metered per frame; the issue asks for that to be visible.
    badge = _node().get_parameter_by_name("model").get_badge()

    assert badge is not None
    assert badge.variant == "warning"


def test_width_and_height_near_4k_warns_about_the_expensive_tier() -> None:
    node = _node()
    node.set_parameter_value("resize_mode", ResizeMode.WIDTH_HEIGHT)
    node.set_parameter_value("target_width", 3200)
    node.set_parameter_value("target_height", 1800)

    badge = node.get_parameter_by_name("resize_mode").get_badge()

    assert badge is not None
    assert badge.variant == "warning"
    assert "4K" in (badge.title or "")


def test_width_and_height_1080p_reports_the_cheaper_tier() -> None:
    node = _node()
    node.set_parameter_value("resize_mode", ResizeMode.WIDTH_HEIGHT)
    node.set_parameter_value("target_width", 1920)
    node.set_parameter_value("target_height", 1080)

    badge = node.get_parameter_by_name("resize_mode").get_badge()

    assert badge is not None
    assert badge.variant == "info"


def test_tier_boundary_is_pixel_area_not_height() -> None:
    # 1080x1920 portrait is the same pixel count as 1920x1080, so it stays in the
    # cheap tier; 1921x1080 crosses it.
    node = _node()
    node.set_parameter_value("resize_mode", ResizeMode.WIDTH_HEIGHT)

    node.set_parameter_value("target_width", 1080)
    node.set_parameter_value("target_height", 1920)
    assert node.get_parameter_by_name("resize_mode").get_badge().variant == "info"

    node.set_parameter_value("target_width", 1922)
    assert 1922 * 1920 > STARLIGHT_1080P_MAX_PIXELS
    assert node.get_parameter_by_name("resize_mode").get_badge().variant == "warning"


def test_width_and_height_over_hard_cap_shows_error_badge(monkeypatch: pytest.MonkeyPatch) -> None:
    # target_width/target_height are themselves capped at 3840/2160, so their product
    # can never exceed STARLIGHT_MAX_OUTPUT_PIXELS through normal UI values -- force it
    # to exercise the defensive badge branch anyway.
    node = _node()
    node.set_parameter_value("resize_mode", ResizeMode.WIDTH_HEIGHT)
    _force_values(node, monkeypatch, target_width=3840, target_height=3840)

    assert 3840 * 3840 > STARLIGHT_MAX_OUTPUT_PIXELS
    node._update_tier_badge()
    badge = node.get_parameter_by_name("resize_mode").get_badge()

    assert badge is not None
    assert badge.variant == "error"


def test_width_only_shows_a_source_dependent_note_badge() -> None:
    node = _node()
    node.set_parameter_value("resize_mode", ResizeMode.WIDTH)
    node.set_parameter_value("target_size", 1920)

    badge = node.get_parameter_by_name("resize_mode").get_badge()

    assert badge is not None
    assert badge.variant == "note"


def test_height_only_shows_a_source_dependent_note_badge() -> None:
    node = _node()
    node.set_parameter_value("resize_mode", ResizeMode.HEIGHT)
    node.set_parameter_value("target_size", 1080)

    badge = node.get_parameter_by_name("resize_mode").get_badge()

    assert badge is not None
    assert badge.variant == "note"
