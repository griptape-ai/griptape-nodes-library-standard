from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

from griptape_nodes_library.utils.ffmpeg_utils import (
    ColorDetails,
    Dimensions,
    FileDetails,
    FrameDetails,
    VideoMetadata,
)
from griptape_nodes_library.video.topaz_video_upscale import (
    ASTRA_FILTER_PARAMS,
    MAX_ASTRA_FRAMES,
    MAX_ASTRA_FRAMES_WITH_PROMPT,
    MAX_STARLIGHT_FRAMES,
    MODEL_FAMILIES,
    MODEL_MAPPING,
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


ASTRA_MODEL = "Astra 2"
STARLIGHT_MODEL = "Starlight Precise 2.6"


def _astra_node(name: str = "TopazVideoUpscale") -> TopazVideoUpscale:
    """A node switched to Astra through the normal setter, so the UI reactions fire."""
    node = _node(name)
    node.set_parameter_value("model", ASTRA_MODEL)
    return node


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
    # The output ceiling belongs to Starlight only, so pin the model explicitly.
    node = _node()
    node.set_parameter_value("model", STARLIGHT_MODEL)
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
async def test_starlight_payload_has_only_source_and_output(monkeypatch: pytest.MonkeyPatch) -> None:
    # Starlight sends nothing but source and output. Asserting the exact key set
    # (rather than `"filters" not in payload`) catches any unexpected top-level key,
    # not just that one.
    node = _node()
    node.set_parameter_value("video", "{inputs}/clip.mp4")
    _stub_probe(monkeypatch, _metadata())
    _stub_public_url(node, monkeypatch)

    payload = await node._build_payload()

    assert set(payload) == {"source", "output"}


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


def test_payload_build_error_triggers_failure_exception() -> None:
    # _handle_failure_exception is what raises when the failure output has no
    # outgoing connection -- without it a ValueError guard sets was_successful=False
    # and quietly dead-ends the flow instead of crashing with immediate feedback.
    node = _node()
    err = ValueError("boom")
    with patch.object(node, "_handle_failure_exception") as mock_failure:
        node._handle_payload_build_error(err)
    mock_failure.assert_called_once_with(err)


# -- container derivation -----------------------------------------------------


@pytest.mark.asyncio
async def test_container_is_derived_from_the_extension(monkeypatch: pytest.MonkeyPatch) -> None:
    node = _node()
    node.set_parameter_value("video", "{inputs}/clip.mov")
    _stub_probe(monkeypatch, _metadata())
    _stub_public_url(node, monkeypatch)

    payload = await node._build_payload()

    assert payload["source"]["container"] == "mov"


@pytest.mark.asyncio
async def test_container_defaults_map_mp4_variants(monkeypatch: pytest.MonkeyPatch) -> None:
    node = _node()
    node.set_parameter_value("video", "{inputs}/clip.m4v")
    _stub_probe(monkeypatch, _metadata())
    _stub_public_url(node, monkeypatch)

    payload = await node._build_payload()

    assert payload["source"]["container"] == "mp4"


@pytest.mark.asyncio
async def test_container_derived_from_data_uri_mime_type(monkeypatch: pytest.MonkeyPatch) -> None:
    node = _node()
    node.set_parameter_value("video", "data:video/quicktime;base64,AAAA")
    _stub_probe(monkeypatch, _metadata())
    _stub_public_url(node, monkeypatch)

    payload = await node._build_payload()

    assert payload["source"]["container"] == "mov"


@pytest.mark.asyncio
async def test_extensionless_url_falls_back_to_mp4(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # A signed URL that strips the filename gives no container signal at all. Guessing
    # mp4 keeps such a URL working; only a *recognizable but unsupported* token raises.
    node = _node()
    node.set_parameter_value("video", "https://acct.blob.core.windows.net/bucket/rawkey?sig=abc")
    _stub_probe(monkeypatch, _metadata())
    _stub_public_url(node, monkeypatch)

    with caplog.at_level(logging.WARNING, logger="griptape_nodes"):
        payload = await node._build_payload()

    assert payload["source"]["container"] == "mp4"
    assert "assuming mp4" in caplog.text


@pytest.mark.asyncio
async def test_data_uri_without_mime_subtype_falls_back_to_mp4(monkeypatch: pytest.MonkeyPatch) -> None:
    node = _node()
    node.set_parameter_value("video", "data:;base64,AAAA")
    _stub_probe(monkeypatch, _metadata())
    _stub_public_url(node, monkeypatch)

    payload = await node._build_payload()

    assert payload["source"]["container"] == "mp4"


@pytest.mark.asyncio
async def test_unsupported_container_raises_before_uploading(monkeypatch: pytest.MonkeyPatch) -> None:
    node = _node()
    node.set_parameter_value("video", "{inputs}/clip.webm")
    _stub_probe(monkeypatch, _metadata())

    calls: list[None] = []

    def _record() -> str:
        calls.append(None)
        return PUBLIC_URL

    monkeypatch.setattr(node._public_video_url_parameter, "get_public_url_for_parameter", _record)

    with pytest.raises(ValueError, match="only accepts mp4, mov, or mkv"):
        await node._build_payload()

    assert calls == []


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


def test_astra_routes_to_its_own_proxy_id() -> None:
    node = _node()
    node.set_parameter_value("model", ASTRA_MODEL)

    assert node._get_api_model_id() == "topaz-video-ast-2"


def test_every_model_has_a_registered_family() -> None:
    # MODEL_FAMILIES is a second table beside MODEL_MAPPING, so drift between them is
    # the failure mode. _family() raises on a miss; this catches it at test time.
    assert MODEL_FAMILIES.keys() == MODEL_MAPPING.keys()


# -- result parsing -----------------------------------------------------------


@pytest.mark.asyncio
async def test_parse_result_downloads_from_the_documented_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    # `download.url` is Topaz's documented shape for a completed generation.
    node = _node()
    captured: dict = {}

    async def fake_download_and_save(self, download_url, *_args, **_kwargs) -> None:  # noqa: ARG001
        captured["url"] = download_url

    monkeypatch.setattr(TopazVideoUpscale, "_download_and_save", fake_download_and_save)

    await node._parse_result({"status": "complete", "download": {"url": "https://topaz.example/out.mp4"}}, "gen-1")

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


def test_model_carries_no_badge() -> None:
    # Cost guidance lives outside the node; the model picker states no rates.
    model_param = _node().get_parameter_by_name("model")
    assert model_param is not None

    assert model_param.get_badge() is None


def test_an_output_within_the_limit_carries_no_badge() -> None:
    node = _node()
    node.set_parameter_value("resize_mode", ResizeMode.WIDTH_HEIGHT)
    node.set_parameter_value("target_width", 3200)
    node.set_parameter_value("target_height", 1800)

    resize_mode_param = node.get_parameter_by_name("resize_mode")
    assert resize_mode_param is not None

    assert resize_mode_param.get_badge() is None


def test_width_and_height_over_hard_cap_shows_error_badge(monkeypatch: pytest.MonkeyPatch) -> None:
    # target_width/target_height are themselves capped at 3840/2160, so their product
    # can never exceed STARLIGHT_MAX_OUTPUT_PIXELS through normal UI values -- force it
    # to exercise the defensive badge branch anyway.
    node = _node()
    node.set_parameter_value("resize_mode", ResizeMode.WIDTH_HEIGHT)
    _force_values(node, monkeypatch, target_width=3840, target_height=3840)

    assert 3840 * 3840 > STARLIGHT_MAX_OUTPUT_PIXELS
    node._update_limit_badge()
    resize_mode_param = node.get_parameter_by_name("resize_mode")
    assert resize_mode_param is not None
    badge = resize_mode_param.get_badge()

    assert badge is not None
    assert badge.variant == "error"


def test_returning_under_the_cap_clears_the_error_badge(monkeypatch: pytest.MonkeyPatch) -> None:
    # A badge that survives the edit that fixed it is worse than no badge at all.
    node = _node()
    node.set_parameter_value("resize_mode", ResizeMode.WIDTH_HEIGHT)
    _force_values(node, monkeypatch, target_width=3840, target_height=3840)
    node._update_limit_badge()

    _force_values(node, monkeypatch, target_width=3840, target_height=2160)
    node._update_limit_badge()

    resize_mode_param = node.get_parameter_by_name("resize_mode")
    assert resize_mode_param is not None
    assert resize_mode_param.get_badge() is None


@pytest.mark.parametrize("mode", [ResizeMode.WIDTH, ResizeMode.HEIGHT, ResizeMode.PERCENTAGE])
def test_source_dependent_modes_carry_no_badge(mode: ResizeMode) -> None:
    # Their output size follows from the source, which isn't probed until the node
    # runs, so there is nothing to check against here.
    node = _node()
    node.set_parameter_value("resize_mode", mode)

    resize_mode_param = node.get_parameter_by_name("resize_mode")
    assert resize_mode_param is not None
    assert resize_mode_param.get_badge() is None


# -- Astra: filters ----------------------------------------------------------


async def _astra_payload(monkeypatch: pytest.MonkeyPatch, node: TopazVideoUpscale) -> dict:
    node.set_parameter_value("video", "{inputs}/clip.mp4")
    _stub_probe(monkeypatch, _metadata())
    _stub_public_url(node, monkeypatch)
    return await node._build_payload()


@pytest.mark.asyncio
async def test_astra_payload_carries_the_creative_filters(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = await _astra_payload(monkeypatch, _astra_node())

    assert payload["filters"] == [{"creativity": 0.5, "sharp": 0.5, "realism": 0.5}]


@pytest.mark.asyncio
async def test_astra_filters_omit_the_model_key(monkeypatch: pytest.MonkeyPatch) -> None:
    # The proxy stamps the routed model code onto filters[0] when no entry carries one.
    # Sending our own would risk the "filters[].model must match" rejection and would
    # force this node to know that the code is the api id minus its "topaz-video-" prefix.
    payload = await _astra_payload(monkeypatch, _astra_node())

    assert "model" not in payload["filters"][0]


@pytest.mark.asyncio
async def test_astra_filters_are_never_an_empty_list(monkeypatch: pytest.MonkeyPatch) -> None:
    # The proxy rejects `filters: []` outright ("must be a non-empty array when provided"),
    # so the key is either absent or populated -- never present and empty.
    payload = await _astra_payload(monkeypatch, _astra_node())

    assert payload["filters"]


@pytest.mark.asyncio
async def test_astra_creative_values_reach_the_filter_verbatim(monkeypatch: pytest.MonkeyPatch) -> None:
    node = _astra_node()
    node.set_parameter_value("creativity", 0.9)
    node.set_parameter_value("sharp", 0.1)
    node.set_parameter_value("realism", 0.75)

    payload = await _astra_payload(monkeypatch, node)

    assert payload["filters"][0]["creativity"] == 0.9
    assert payload["filters"][0]["sharp"] == 0.1
    assert payload["filters"][0]["realism"] == 0.75


@pytest.mark.asyncio
async def test_astra_creative_values_are_sent_as_floats(monkeypatch: pytest.MonkeyPatch) -> None:
    # A value arriving over a connection or a workflow round-trip can be an int, and
    # `json.dumps(1)` emits `1` against a field Topaz documents as a decimal.
    node = _astra_node()
    node.set_parameter_value("creativity", 1)
    node.set_parameter_value("sharp", 0)

    payload = await _astra_payload(monkeypatch, node)

    assert isinstance(payload["filters"][0]["creativity"], float)
    assert isinstance(payload["filters"][0]["sharp"], float)


@pytest.mark.asyncio
async def test_astra_filters_include_a_nonempty_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    node = _astra_node()
    node.set_parameter_value("prompt", "a bouncing ball")

    payload = await _astra_payload(monkeypatch, node)

    assert payload["filters"][0]["prompt"] == "a bouncing ball"


@pytest.mark.asyncio
async def test_astra_filters_omit_a_whitespace_only_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    # The proxy decides whether a job is "prompted" -- and which frame cap to bill
    # against -- from the truthiness of this key, so a blank one must not be sent.
    node = _astra_node()
    node.set_parameter_value("prompt", "   \n ")

    payload = await _astra_payload(monkeypatch, node)

    assert "prompt" not in payload["filters"][0]


# -- Astra: frame caps -------------------------------------------------------


@pytest.mark.asyncio
async def test_astra_accepts_up_to_the_unprompted_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    node = _astra_node()
    node.set_parameter_value("video", "{inputs}/clip.mp4")
    _stub_probe(monkeypatch, _metadata(nb_frames=MAX_ASTRA_FRAMES))
    _stub_public_url(node, monkeypatch)

    payload = await node._build_payload()

    assert payload["source"]["frameCount"] == MAX_ASTRA_FRAMES


@pytest.mark.asyncio
async def test_astra_rejects_one_frame_over_the_unprompted_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    node = _astra_node()
    node.set_parameter_value("video", "{inputs}/clip.mp4")
    _stub_probe(monkeypatch, _metadata(nb_frames=MAX_ASTRA_FRAMES + 1))
    _stub_public_url(node, monkeypatch)

    with pytest.raises(ValueError, match="over Astra's"):
        await node._build_payload()


@pytest.mark.asyncio
async def test_a_prompt_drops_astras_cap_to_450(monkeypatch: pytest.MonkeyPatch) -> None:
    # 500 frames is fine unprompted and rejected once a prompt is set -- the whole
    # point of enforcing this client-side, since the proxy only clamps what it bills.
    node = _astra_node()
    node.set_parameter_value("video", "{inputs}/clip.mp4")
    _stub_probe(monkeypatch, _metadata(nb_frames=MAX_ASTRA_FRAMES_WITH_PROMPT + 50))
    _stub_public_url(node, monkeypatch)

    assert (await node._build_payload())["source"]["frameCount"] == MAX_ASTRA_FRAMES_WITH_PROMPT + 50

    node.set_parameter_value("prompt", "a bouncing ball")

    with pytest.raises(ValueError, match="clearing the prompt"):
        await node._build_payload()


@pytest.mark.asyncio
async def test_a_whitespace_prompt_does_not_lower_the_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    # The cap enforced here has to agree with the cap the proxy bills against, and the
    # proxy's test is truthiness of filters[].prompt -- which a blank string fails.
    node = _astra_node()
    node.set_parameter_value("prompt", "   ")
    node.set_parameter_value("video", "{inputs}/clip.mp4")
    _stub_probe(monkeypatch, _metadata(nb_frames=MAX_ASTRA_FRAMES_WITH_PROMPT + 50))
    _stub_public_url(node, monkeypatch)

    payload = await node._build_payload()

    assert payload["source"]["frameCount"] == MAX_ASTRA_FRAMES_WITH_PROMPT + 50


@pytest.mark.asyncio
async def test_astra_prompt_cap_rejects_before_uploading(monkeypatch: pytest.MonkeyPatch) -> None:
    # Same discipline as the Starlight case: the cheap local check must run before the
    # upload, or a rejected job still costs a round trip and cloud storage.
    node = _astra_node()
    node.set_parameter_value("prompt", "a bouncing ball")
    node.set_parameter_value("video", "{inputs}/clip.mp4")
    _stub_probe(monkeypatch, _metadata(nb_frames=MAX_ASTRA_FRAMES_WITH_PROMPT + 1))

    calls: list[None] = []

    def _record() -> str:
        calls.append(None)
        return PUBLIC_URL

    monkeypatch.setattr(node._public_video_url_parameter, "get_public_url_for_parameter", _record)

    with pytest.raises(ValueError, match="over Astra's"):
        await node._build_payload()

    assert calls == []


@pytest.mark.asyncio
async def test_starlights_frame_cap_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    # Astra's prompt-dependent cap must not leak into Starlight, which has no prompt
    # at all and keeps its flat 9000.
    node = _node()
    node.set_parameter_value("video", "{inputs}/clip.mp4")
    _stub_probe(monkeypatch, _metadata(nb_frames=MAX_STARLIGHT_FRAMES))
    _stub_public_url(node, monkeypatch)

    payload = await node._build_payload()

    assert payload["source"]["frameCount"] == MAX_STARLIGHT_FRAMES


# -- Astra: output resolution ------------------------------------------------


def test_astra_has_no_hard_output_cap() -> None:
    # Topaz documents no output ceiling for Astra and the proxy enforces none, so the
    # node must not invent one and reject a job Topaz may well accept.
    node = _astra_node()
    node.set_parameter_value("resize_mode", ResizeMode.PERCENTAGE)
    node.set_parameter_value("percentage", 200)

    assert node._output_resolution(3840, 2160) == (7680, 4320)


def test_starlight_still_enforces_the_hard_output_cap() -> None:
    node = _astra_node()
    node.set_parameter_value("resize_mode", ResizeMode.PERCENTAGE)
    node.set_parameter_value("percentage", 200)
    node.set_parameter_value("model", STARLIGHT_MODEL)

    with pytest.raises(ValueError, match="hard limit"):
        node._output_resolution(3840, 2160)


# -- Astra: badges -----------------------------------------------------------


def test_neither_family_puts_a_badge_on_the_model_picker() -> None:
    for node in (_node(), _astra_node()):
        param = node.get_parameter_by_name("model")
        assert param is not None
        assert param.get_badge() is None


def test_astra_has_no_ceiling_so_an_oversized_output_raises_no_badge(monkeypatch: pytest.MonkeyPatch) -> None:
    # The shape that gives Starlight a red "exceeds the limit" badge is unremarkable
    # for Astra, which Topaz publishes no output ceiling for.
    node = _astra_node()
    node.set_parameter_value("resize_mode", ResizeMode.WIDTH_HEIGHT)
    _force_values(node, monkeypatch, target_width=3840, target_height=3840)
    node._update_limit_badge()

    resize_mode_param = node.get_parameter_by_name("resize_mode")
    assert resize_mode_param is not None
    assert resize_mode_param.get_badge() is None


def test_switching_to_astra_clears_starlights_limit_badge(monkeypatch: pytest.MonkeyPatch) -> None:
    # The ceiling belongs to the family, so the badge has to follow a model change and
    # not just a size change.
    node = _node()
    node.set_parameter_value("resize_mode", ResizeMode.WIDTH_HEIGHT)
    _force_values(node, monkeypatch, target_width=3840, target_height=3840)
    node._update_limit_badge()

    resize_mode_param = node.get_parameter_by_name("resize_mode")
    assert resize_mode_param is not None
    assert resize_mode_param.get_badge() is not None

    node.set_parameter_value("model", ASTRA_MODEL)
    assert resize_mode_param.get_badge() is None


def _prompt_badge_variant(node: TopazVideoUpscale) -> str | None:
    param = node.get_parameter_by_name("prompt")
    assert param is not None
    badge = param.get_badge()
    assert badge is not None
    return badge.variant


def test_the_prompt_badge_states_the_cap_before_one_is_typed() -> None:
    # The source frame count is not knowable in the editor, so the badge states the
    # cap rather than testing against it.
    assert _prompt_badge_variant(_astra_node()) == "note"


def test_typing_a_prompt_escalates_the_badge_to_a_warning() -> None:
    node = _astra_node()
    node.set_parameter_value("prompt", "a bouncing ball")

    assert _prompt_badge_variant(node) == "warning"


def test_clearing_the_prompt_returns_the_badge_to_a_note() -> None:
    node = _astra_node()
    node.set_parameter_value("prompt", "a bouncing ball")
    node.set_parameter_value("prompt", "")

    assert _prompt_badge_variant(node) == "note"


# -- Astra: parameter visibility ---------------------------------------------


def _hidden(node: TopazVideoUpscale, name: str) -> bool:
    param = node.get_parameter_by_name(name)
    assert param is not None
    return param.ui_options.get("hide") is True


def test_the_creative_controls_are_hidden_on_a_fresh_starlight_node() -> None:
    # Not just tidiness: a visible prompt box under Starlight would silently do
    # nothing, because Starlight sends no filters at all.
    node = _node()

    for name in ASTRA_FILTER_PARAMS:
        assert _hidden(node, name), f"{name} should be hidden for Starlight"


def test_selecting_astra_reveals_the_creative_controls() -> None:
    node = _astra_node()

    for name in ASTRA_FILTER_PARAMS:
        assert not _hidden(node, name), f"{name} should be visible for Astra"


def test_switching_back_to_starlight_hides_them_again() -> None:
    node = _astra_node()
    node.set_parameter_value("model", STARLIGHT_MODEL)

    for name in ASTRA_FILTER_PARAMS:
        assert _hidden(node, name), f"{name} should be hidden again for Starlight"


def test_selecting_astra_does_not_disturb_the_resize_fields() -> None:
    # The two visibility rules are independent; switching models must not reset the
    # resize UI the user has already set up.
    node = _node()
    node.set_parameter_value("resize_mode", ResizeMode.WIDTH_HEIGHT)
    node.set_parameter_value("model", ASTRA_MODEL)

    assert not _hidden(node, "target_width")
    assert not _hidden(node, "target_height")
    assert _hidden(node, "percentage")
    assert _hidden(node, "target_size")
