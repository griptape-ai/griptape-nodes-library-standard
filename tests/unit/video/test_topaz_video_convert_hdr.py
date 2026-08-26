from __future__ import annotations

from unittest.mock import patch

import pytest

from griptape_nodes_library.proxy.proxy_api_key_providers import _NODE_PROVIDER_CONFIGS
from griptape_nodes_library.utils.ffmpeg_utils import (
    ColorDetails,
    Dimensions,
    FileDetails,
    FrameDetails,
    VideoMetadata,
)
from griptape_nodes_library.video.topaz_video_convert_hdr import (
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_FORMAT,
    MAX_HYPERION_FRAMES,
    MODEL_MAPPING,
    OUTPUT_ENCODINGS,
    OutputFormat,
    TopazVideoConvertHdr,
)


def _metadata(
    *,
    width: int = 1920,
    height: int = 1080,
    nb_frames: int | None = 300,
    duration: float | None = 10.0,
    frame_rate: float = 30.0,
    file_size: int | None = 1234,
    color_transfer: str | None = None,
    color_primaries: str | None = None,
    field_order: str | None = None,
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
        color_details=ColorDetails(
            optional_color_transfer=color_transfer,
            optional_color_primaries=color_primaries,
            optional_field_order=field_order,
        ),
        frame_details=FrameDetails(
            r_frame_rate=f"{int(frame_rate)}/1",
            avg_frame_rate=f"{int(frame_rate)}/1",
            time_base="1/30000",
            frame_rate=frame_rate,
            optional_nb_frames=nb_frames,
        ),
    )


def _node(name: str = "TopazVideoConvertHdr") -> TopazVideoConvertHdr:
    return TopazVideoConvertHdr(name=name)


def _stub_probe(monkeypatch: pytest.MonkeyPatch, metadata: VideoMetadata) -> None:
    monkeypatch.setattr(
        TopazVideoConvertHdr,
        "_probe_source",
        lambda self, video_input: metadata,  # noqa: ARG005
    )


PUBLIC_URL = "https://acct.blob.core.windows.net/bucket/clip.mp4?sig=abc"


def _stub_public_url(
    node: TopazVideoConvertHdr, monkeypatch: pytest.MonkeyPatch, value: str | None = PUBLIC_URL
) -> None:
    """Stand in for the upload to Griptape Cloud storage.

    The real call presigns a URL over the network; the node only cares that it
    gets one back to hand to Topaz as `source.external`.
    """
    monkeypatch.setattr(
        node._public_video_url_parameter,
        "get_public_url_for_parameter",
        lambda: value,
    )


# -- model routing -----------------------------------------------------------


def test_default_model_routes_to_hyperion_2_5() -> None:
    node = _node()

    assert node.get_parameter_value("model") == DEFAULT_MODEL
    assert node._get_api_model_id() == "topaz-video-hyp-2.5"


def test_hyperion_2_is_selectable() -> None:
    node = _node()
    node.set_parameter_value("model", "Hyperion 2")

    assert node._get_api_model_id() == "topaz-video-hyp-2"


def test_unknown_model_falls_back_to_the_default() -> None:
    # The Options trait makes this unreachable from the UI, but a workflow file
    # carrying a stale name must not route to an empty model id.
    node = _node()
    node.set_parameter_value("model", "Hyperion 9")

    assert node._get_api_model_id() == MODEL_MAPPING[DEFAULT_MODEL]


def test_every_model_is_declared_in_the_byok_registry() -> None:
    # Missing from _NODE_PROVIDER_CONFIGS the node silently loses its BYOK
    # parameters -- no error, just a node that can only use the Griptape key.
    assert _NODE_PROVIDER_CONFIGS["TopazVideoConvertHdr"].provider_id == "topaz"


# -- output format mapping ---------------------------------------------------


def test_every_output_format_has_an_encoding() -> None:
    # OutputFormat and OUTPUT_ENCODINGS are two tables that must not drift;
    # _encoding() raises on a miss, and this catches it at test time.
    assert set(OUTPUT_ENCODINGS) == set(OutputFormat)


def test_default_format_is_h265_main10_in_mp4() -> None:
    node = _node()
    encoding = node._encoding()

    assert node.get_parameter_value("output_format") == DEFAULT_OUTPUT_FORMAT
    assert (encoding.video_encoder, encoding.video_profile, encoding.container) == ("H265", "Main10", "mp4")


def test_prores_maps_to_a_mov() -> None:
    node = _node()
    node.set_parameter_value("output_format", OutputFormat.PRORES_422_HQ)
    encoding = node._encoding()

    assert (encoding.video_encoder, encoding.video_profile, encoding.container) == ("ProRes", "422 HQ", "mov")


def test_unknown_output_format_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # The Options trait silently discards an off-list value, so this is only
    # reachable through a stale workflow file -- but encoding to the wrong
    # container is worse than a loud failure, so the guard has to hold.
    node = _node()
    monkeypatch.setattr(node, "get_parameter_value", lambda _name: "AV1 (webm)")

    with pytest.raises(ValueError, match="unknown output format"):
        node._encoding()


def test_the_options_trait_discards_an_off_list_format() -> None:
    node = _node()
    node.set_parameter_value("output_format", "AV1 (webm)")

    assert node.get_parameter_value("output_format") == DEFAULT_OUTPUT_FORMAT


def test_output_filename_tracks_the_container() -> None:
    node = _node()
    assert node.get_parameter_value("output_file") == "topaz_video_convert_hdr.mp4"

    node.set_parameter_value("output_format", OutputFormat.PRORES_422_HQ)
    assert node.get_parameter_value("output_file") == "topaz_video_convert_hdr.mov"

    node.set_parameter_value("output_format", OutputFormat.H265_MAIN10)
    assert node.get_parameter_value("output_file") == "topaz_video_convert_hdr.mp4"


def test_a_user_typed_filename_is_never_clobbered() -> None:
    node = _node()
    node.set_parameter_value("output_file", "my_graded_master.mp4")
    node.set_parameter_value("output_format", OutputFormat.PRORES_422_HQ)

    assert node.get_parameter_value("output_file") == "my_graded_master.mp4"


# -- output resolution -------------------------------------------------------


def test_output_resolution_echoes_the_source() -> None:
    # Hyperion converts rather than scales, so there is no resize to apply.
    node = _node()

    assert node._output_resolution(1920, 1080) == (1920, 1080)
    assert node._output_resolution(3840, 2160) == (3840, 2160)


def test_output_resolution_is_rounded_to_even() -> None:
    node = _node()

    assert node._output_resolution(1921, 1081) == (1920, 1080)


def test_output_resolution_rejects_a_source_over_the_encoder_limit() -> None:
    node = _node()

    with pytest.raises(ValueError, match="over the 8192-pixel limit for H265"):
        node._output_resolution(10000, 4000)


def test_prores_accepts_a_source_h265_cannot_hold() -> None:
    node = _node()
    node.set_parameter_value("output_format", OutputFormat.PRORES_422_HQ)

    assert node._output_resolution(10000, 4000) == (10000, 4000)


# -- payload -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_build_payload_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    node = _node()
    node.set_parameter_value("video", "{inputs}/clip.mp4")
    _stub_probe(monkeypatch, _metadata())
    _stub_public_url(node, monkeypatch)

    payload = await node._build_payload()

    assert payload["source"] == {
        "container": "mp4",
        "frameCount": 300,
        "frameRate": 30.0,
        "resolution": {"width": 1920, "height": 1080},
        "duration": 10.0,
        "size": 1234,
        "external": {"provider": "web-url", "presignedUrl": PUBLIC_URL},
    }
    assert payload["output"] == {
        "resolution": {"width": 1920, "height": 1080},
        "videoEncoder": "H265",
        "videoProfile": "Main10",
        "container": "mp4",
    }


@pytest.mark.asyncio
async def test_build_payload_carries_the_prores_triple(monkeypatch: pytest.MonkeyPatch) -> None:
    node = _node()
    node.set_parameter_value("video", "{inputs}/clip.mp4")
    node.set_parameter_value("output_format", OutputFormat.PRORES_422_HQ)
    _stub_probe(monkeypatch, _metadata())
    _stub_public_url(node, monkeypatch)

    payload = await node._build_payload()

    assert payload["output"]["videoEncoder"] == "ProRes"
    assert payload["output"]["videoProfile"] == "422 HQ"
    assert payload["output"]["container"] == "mov"


@pytest.mark.asyncio
async def test_build_payload_omits_filters(monkeypatch: pytest.MonkeyPatch) -> None:
    # Hyperion takes no creative controls, and the proxy stamps the routed model
    # code onto the filters it synthesizes itself.
    node = _node()
    node.set_parameter_value("video", "{inputs}/clip.mp4")
    _stub_probe(monkeypatch, _metadata())
    _stub_public_url(node, monkeypatch)

    payload = await node._build_payload()

    assert "filters" not in payload
    assert "model" not in payload


@pytest.mark.asyncio
async def test_build_payload_sends_no_video_data(monkeypatch: pytest.MonkeyPatch) -> None:
    # The whole point of source.external: the video never travels in the body.
    node = _node()
    node.set_parameter_value("video", "{inputs}/clip.mp4")
    _stub_probe(monkeypatch, _metadata())
    _stub_public_url(node, monkeypatch)

    payload = await node._build_payload()

    assert "video" not in payload
    assert payload["source"]["external"]["presignedUrl"] == PUBLIC_URL


@pytest.mark.asyncio
async def test_build_payload_omits_absent_optional_source_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    node = _node()
    node.set_parameter_value("video", "{inputs}/clip.mp4")
    _stub_probe(monkeypatch, _metadata(duration=None, file_size=None))
    _stub_public_url(node, monkeypatch)

    payload = await node._build_payload()

    assert "duration" not in payload["source"]
    assert "size" not in payload["source"]


@pytest.mark.asyncio
async def test_build_payload_derives_frame_count_from_duration(monkeypatch: pytest.MonkeyPatch) -> None:
    node = _node()
    node.set_parameter_value("video", "{inputs}/clip.mp4")
    _stub_probe(monkeypatch, _metadata(nb_frames=None, duration=4.5, frame_rate=24.0))
    _stub_public_url(node, monkeypatch)

    payload = await node._build_payload()

    assert payload["source"]["frameCount"] == 108


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
    _stub_probe(monkeypatch, _metadata(nb_frames=MAX_HYPERION_FRAMES + 1))
    _stub_public_url(node, monkeypatch)

    with pytest.raises(ValueError, match="over the 9,000-frame job limit"):
        await node._build_payload()


@pytest.mark.asyncio
async def test_rejected_input_is_never_uploaded(monkeypatch: pytest.MonkeyPatch) -> None:
    # Uploading costs a round trip and cloud storage, so the cheap local checks
    # have to run first. A clip over the frame cap must be rejected before it is
    # ever sent anywhere.
    node = _node()
    node.set_parameter_value("video", "{inputs}/clip.mp4")
    _stub_probe(monkeypatch, _metadata(nb_frames=MAX_HYPERION_FRAMES + 1))

    calls: list[None] = []

    def _record() -> str:
        calls.append(None)
        return PUBLIC_URL

    monkeypatch.setattr(node._public_video_url_parameter, "get_public_url_for_parameter", _record)

    with pytest.raises(ValueError, match="over the 9,000-frame job limit"):
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


@pytest.mark.asyncio
async def test_build_payload_derives_the_container_from_a_mov_input(monkeypatch: pytest.MonkeyPatch) -> None:
    node = _node()
    node.set_parameter_value("video", "{inputs}/clip.mov")
    _stub_probe(monkeypatch, _metadata())
    _stub_public_url(node, monkeypatch)

    payload = await node._build_payload()

    assert payload["source"]["container"] == "mov"


# -- source advisories -------------------------------------------------------


def test_no_advisories_for_ordinary_progressive_sdr() -> None:
    node = _node()

    assert (
        node._source_advisories(_metadata(color_transfer="bt709", color_primaries="bt709", field_order="progressive"))
        == []
    )


def test_an_unprobed_source_raises_no_advisories() -> None:
    # ffprobe leaves all three fields unset on plenty of ordinary files; an unset
    # value must not read as HDR or as interlaced.
    node = _node()

    assert node._source_advisories(_metadata()) == []


def test_pq_transfer_is_flagged_as_already_hdr() -> None:
    node = _node()

    advisories = node._source_advisories(_metadata(color_transfer="smpte2084"))

    assert len(advisories) == 1
    assert "already looks like HDR" in advisories[0]


def test_bt2020_primaries_are_flagged_as_already_hdr() -> None:
    node = _node()

    advisories = node._source_advisories(_metadata(color_primaries="bt2020"))

    assert len(advisories) == 1
    assert "already looks like HDR" in advisories[0]


def test_interlaced_footage_is_flagged() -> None:
    node = _node()

    advisories = node._source_advisories(_metadata(field_order="tt"))

    assert len(advisories) == 1
    assert "interlaced" in advisories[0]


def test_unknown_field_order_is_not_flagged_as_interlaced() -> None:
    # ffprobe reports "unknown" when it could not determine field order, not when it
    # detected interlacing -- it must not trip the interlaced advisory.
    node = _node()

    assert node._source_advisories(_metadata(field_order="unknown")) == []


def test_bt2020_bit_depth_transfer_tags_are_not_flagged_as_hdr() -> None:
    # ffprobe's "bt2020-10"/"bt2020-12" label bit depth on wide-gamut content and use
    # the same OETF as BT.709 -- they are not HDR transfer characteristics.
    node = _node()

    assert node._source_advisories(_metadata(color_transfer="bt2020-10")) == []
    assert node._source_advisories(_metadata(color_transfer="bt2020-12")) == []


def test_both_advisories_can_fire_at_once() -> None:
    node = _node()

    assert len(node._source_advisories(_metadata(color_transfer="smpte2084", field_order="bff"))) == 2


@pytest.mark.asyncio
async def test_an_already_hdr_source_is_still_converted(monkeypatch: pytest.MonkeyPatch) -> None:
    # The advisory warns; it must not block. A file mislabelled BT.2020 is real,
    # and so is deliberately re-running a conversion.
    node = _node()
    node.set_parameter_value("video", "{inputs}/clip.mp4")
    _stub_probe(monkeypatch, _metadata(color_transfer="smpte2084"))
    _stub_public_url(node, monkeypatch)

    payload = await node._build_payload()

    assert payload["source"]["frameCount"] == 300
    assert node._advisories


# -- result parsing -----------------------------------------------------------


@pytest.mark.asyncio
async def test_parse_result_downloads_from_the_documented_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    # `download.url` is Topaz's documented shape for a completed generation.
    node = _node()
    captured: dict = {}

    async def fake_download_and_save(self, download_url, *_args, **_kwargs) -> None:  # noqa: ARG001
        captured["url"] = download_url

    monkeypatch.setattr(TopazVideoConvertHdr, "_download_and_save", fake_download_and_save)

    await node._parse_result({"status": "complete", "download": {"url": "https://topaz.example/out.mp4"}}, "gen-1")

    assert captured["url"] == "https://topaz.example/out.mp4"


@pytest.mark.asyncio
async def test_parse_result_takes_the_binary_branch_for_raw_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    node = _node()
    captured: dict = {}

    async def fake_binary(self, video_bytes: bytes) -> None:  # noqa: ARG001
        captured["bytes"] = video_bytes

    monkeypatch.setattr(TopazVideoConvertHdr, "_handle_binary_video_response", fake_binary)

    await node._parse_result({"raw_bytes": b"\x00\x01"}, "gen-1")

    assert captured["bytes"] == b"\x00\x01"


@pytest.mark.asyncio
async def test_empty_binary_response_fails_cleanly() -> None:
    node = _node()

    await node._handle_binary_video_response(b"")

    assert node.parameter_output_values["was_successful"] is False
    assert node.parameter_output_values["video_output"] is None


def test_payload_build_error_triggers_failure_exception() -> None:
    # _handle_failure_exception is what raises when the failure output has no
    # outgoing connection -- without it a ValueError guard sets was_successful=False
    # and quietly dead-ends the flow instead of crashing with immediate feedback.
    node = _node()
    err = ValueError("boom")
    with patch.object(node, "_handle_failure_exception") as mock_failure:
        node._handle_payload_build_error(err)
    mock_failure.assert_called_once_with(err)


def test_set_safe_defaults_clears_every_output() -> None:
    node = _node()
    node._set_safe_defaults()

    assert node.parameter_output_values["generation_id"] == ""
    assert node.parameter_output_values["provider_response"] is None
    assert node.parameter_output_values["video_output"] is None
