"""LTX Video Alpha Generation: payload, input validation, the ProRes toggle, outputs and direct download."""

from __future__ import annotations

import dataclasses
import json
import subprocess
import threading
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from griptape.artifacts import VideoUrlArtifact
from griptape_nodes.exe_types.core_types import NodeError, Parameter

from griptape_nodes_library.proxy.griptape_proxy_node import DIRECT_DOWNLOAD_PARAMETER
from griptape_nodes_library.proxy.hosted_artifacts import ArtifactKind, HostedArtifact, HostedArtifactError
from griptape_nodes_library.proxy.output_size import estimate_prores_4444_bytes
from griptape_nodes_library.video import ltx_video_alpha_gen as alpha_module
from griptape_nodes_library.video.ltx_video_alpha_gen import LTXVideoAlphaGen

GENERATION_ID = "gen-1"
MASK_URL = "https://provider.example/mask.mp4?sig=x"
PRORES_URL = "https://provider.example/rgba.mov?sig=x"
TOGGLE = "output_rgba_prores"
VIDEO_URL = "https://example.com/in.mp4"


def _node() -> LTXVideoAlphaGen:
    return LTXVideoAlphaGen(name="Alpha")


def _param(node: LTXVideoAlphaGen, name: str) -> Parameter:
    param = node.get_parameter_by_name(name)
    assert param is not None
    return param


def _hidden(node: LTXVideoAlphaGen, name: str) -> bool:
    return bool(_param(node, name).ui_options.get("hide"))


def _probed(width: int, height: int, frames: int = 100, duration: float = 4.0) -> Any:
    return alpha_module._ProbeResult(width, height, frames, duration)


def _stub_probe(node: LTXVideoAlphaGen, monkeypatch: pytest.MonkeyPatch, result: Any) -> MagicMock:
    probe = MagicMock(return_value=result)
    monkeypatch.setattr(node, "_probe_video", probe)
    return probe


# --- model and payload ------------------------------------------------------------------------


def test_model_id_and_single_choice() -> None:
    node = _node()

    assert node._get_api_model_id() == "ltx:video-alpha-gen"
    rows = {row["name"] for row in _param(node, "model").ui_options["data"]}
    assert rows == {"ltx"}


@pytest.mark.asyncio
async def test_payload_only_carries_video_uri(monkeypatch: pytest.MonkeyPatch) -> None:
    node = _node()
    node.set_parameter_value("video", VIDEO_URL)
    _stub_probe(node, monkeypatch, _probed(1920, 1080))
    monkeypatch.setattr(node, "_resolve_video_uri", AsyncMock(return_value="https://cdn.example/u.mp4"))

    assert await node._build_payload() == {"video_uri": "https://cdn.example/u.mp4"}


@pytest.mark.asyncio
async def test_payload_requires_a_video() -> None:
    node = _node()

    with pytest.raises(NodeError, match="input video"):
        await node._build_payload()


# --- validation -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("width", "height", "frames", "message"),
    [
        (1920, 1080, 3072, None),
        (1920, 1080, 3073, "3073 frames"),
        (3840, 2160, 100, None),
        (3842, 1080, 100, "long side is 3842"),
        (1080, 3842, 100, "long side is 3842"),
        (3840, 2162, 100, "short side is 2162"),
        (2162, 2162, 100, "short side is 2162"),
        (3840, 1080, 100, None),
        (3200, 900, 100, None),
        (3200, 899, 100, "aspect ratio"),
        (899, 3200, 100, "aspect ratio"),
        (900, 3200, 100, None),
    ],
)
def test_validation_boundaries(
    monkeypatch: pytest.MonkeyPatch, width: int, height: int, frames: int, message: str | None
) -> None:
    node = _node()
    _stub_probe(node, monkeypatch, _probed(width, height, frames))

    if message is None:
        node._validate_video_input(VIDEO_URL)
        return
    with pytest.raises(NodeError, match=message) as excinfo:
        node._validate_video_input(VIDEO_URL)
    assert "'input video'" in str(excinfo.value)


def _ffprobe_output(stream: dict[str, Any], duration: str | None = None) -> MagicMock:
    payload: dict[str, Any] = {"streams": [stream], "format": {"duration": duration} if duration else {}}
    return MagicMock(stdout=json.dumps(payload))


def _patch_ffprobe(monkeypatch: pytest.MonkeyPatch, completed: MagicMock) -> MagicMock:
    run = MagicMock(return_value=completed)
    monkeypatch.setattr(alpha_module.run, "get_or_fetch_platform_executables_else_raise", lambda: ("ffmpeg", "ffprobe"))
    monkeypatch.setattr(alpha_module, "resolve_media_location", lambda url: url)
    monkeypatch.setattr(alpha_module.subprocess, "run", run)
    return run


def test_probe_swaps_dimensions_of_a_quarter_turned_input(monkeypatch: pytest.MonkeyPatch) -> None:
    # Stored landscape 4000x1000 and rotated 90 degrees, so it is really 1000x4000. The long side is
    # over 3840 either way, but the message must name the displayed dimensions.
    stream = {
        "width": 4000,
        "height": 1000,
        "nb_frames": "10",
        "avg_frame_rate": "10/1",
        "side_data_list": [{"rotation": -90}],
    }
    _patch_ffprobe(monkeypatch, _ffprobe_output(stream))
    node = _node()

    probed = node._probe_video(VIDEO_URL)

    assert probed is not None
    assert (probed.width, probed.height) == (1000, 4000)
    with pytest.raises(NodeError, match="1000x4000"):
        node._validate_video_input(VIDEO_URL)


def test_probe_falls_back_to_duration_times_fps_for_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    stream = {"width": 1920, "height": 1080, "avg_frame_rate": "24000/1001"}
    _patch_ffprobe(monkeypatch, _ffprobe_output(stream, duration="10.0"))

    probed = _node()._probe_video(VIDEO_URL)

    assert probed is not None
    assert probed.frames == round(10.0 * 24000 / 1001)
    assert probed.duration_seconds == 10.0


def test_probe_failure_skips_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    run = MagicMock(side_effect=subprocess.CalledProcessError(1, "ffprobe"))
    monkeypatch.setattr(alpha_module.run, "get_or_fetch_platform_executables_else_raise", lambda: ("ffmpeg", "ffprobe"))
    monkeypatch.setattr(alpha_module, "resolve_media_location", lambda url: url)
    monkeypatch.setattr(alpha_module.subprocess, "run", run)
    node = _node()
    node.set_parameter_value("output_rgba_prores", True)
    node.set_parameter_value("video", VideoUrlArtifact(value=VIDEO_URL))
    node._probe_cache = {}

    node._validate_video_input(VIDEO_URL)

    # The ProRes is requested and a video is set, so only the failed probe makes this None.
    assert node._estimate_output_bytes() is None
    assert run.called


def test_validation_and_estimate_share_one_probe_per_run(monkeypatch: pytest.MonkeyPatch) -> None:
    node = _node()
    node.set_parameter_value("video", VIDEO_URL)
    node.set_parameter_value(TOGGLE, True)
    probe = _stub_probe(node, monkeypatch, _probed(1920, 1080))
    node._probe_cache = {}

    node._validate_video_input(VIDEO_URL)
    node._estimate_output_bytes()

    assert probe.call_count == 1


# --- toggle -----------------------------------------------------------------------------------


def test_toggle_defaults_to_off() -> None:
    node = _node()

    assert _param(node, TOGGLE).default_value is False
    assert node.get_parameter_value(TOGGLE) is False
    assert "42 MB" in _param(node, TOGGLE).tooltip


def test_estimate_is_none_when_off_and_the_prores_estimate_when_on(monkeypatch: pytest.MonkeyPatch) -> None:
    node = _node()
    node.set_parameter_value("video", VIDEO_URL)
    _stub_probe(node, monkeypatch, _probed(1920, 1080, frames=100, duration=4.0))

    assert node._estimate_output_bytes() is None

    node.set_parameter_value(TOGGLE, True)
    assert node._estimate_output_bytes() == estimate_prores_4444_bytes(1920, 1080, 4.0)


def test_rgba_outputs_are_hidden_for_a_new_node_and_follow_the_toggle() -> None:
    node = _node()
    assert _hidden(node, "rgba_video")
    assert _hidden(node, "rgba_video_file")

    node.set_parameter_value(TOGGLE, True)
    assert not _hidden(node, "rgba_video")
    assert not _hidden(node, "rgba_video_file")

    node.set_parameter_value(TOGGLE, False)
    assert _hidden(node, "rgba_video")
    assert _hidden(node, "rgba_video_file")


def test_rgba_visibility_is_correct_on_workflow_load() -> None:
    # Load restores values with initial_setup=True, which skips after_value_set.
    node = _node()
    node.set_parameter_value(TOGGLE, True, initial_setup=True)
    assert not _hidden(node, "rgba_video")
    assert not _hidden(node, "rgba_video_file")

    node.set_parameter_value(TOGGLE, False, initial_setup=True)
    assert _hidden(node, "rgba_video")
    assert _hidden(node, "rgba_video_file")


# --- direct download --------------------------------------------------------------------------


def test_direct_download_is_used_whenever_the_prores_is_not_requested() -> None:
    node = _node()
    assert node._direct_download_enabled()

    node.set_parameter_value(TOGGLE, True)
    assert not node._direct_download_enabled()

    node.set_parameter_value(DIRECT_DOWNLOAD_PARAMETER, True)
    assert node._direct_download_enabled()


def test_direct_download_urls_keep_mask_then_prores_order() -> None:
    node = _node()
    result = {"result": {"rgba_video_url": PRORES_URL, "video_url": MASK_URL}}

    assert node._direct_download_result_urls(result) == [
        (ArtifactKind.VIDEO, MASK_URL),
        (ArtifactKind.VIDEO, PRORES_URL),
    ]


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        ({"result": {"video_url": MASK_URL}}, [(ArtifactKind.VIDEO, MASK_URL)]),
        # No matte: report nothing rather than let the ProRes take the matte's position.
        ({"result": {"rgba_video_url": PRORES_URL}}, []),
        ({"result": {"video_url": "", "rgba_video_url": None}}, []),
        ({"result": "nope"}, []),
        ({}, []),
    ],
)
def test_direct_download_urls_are_defensive(result: dict[str, Any], expected: list[tuple[str, str]]) -> None:
    assert _node()._direct_download_result_urls(result) == expected


# --- _parse_result ----------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _SavedFile:
    location: str
    name: str


class _Harness:
    def __init__(self, node: LTXVideoAlphaGen, monkeypatch: pytest.MonkeyPatch) -> None:
        self.node = node
        self.resolved: list[tuple[str | None, int]] = []
        self.saved_to: list[Any] = []
        self.statuses: list[tuple[bool, str]] = []
        self.has_prores = True
        self.mask_ok = True
        self.prores_save_error: Exception | None = None

        monkeypatch.setattr(node, "_save_generated_media", self._save_media)
        monkeypatch.setattr(node, "_hosted_artifact", self._artifact)
        monkeypatch.setattr(node, "_save_artifact_to_project", self._save_artifact)
        monkeypatch.setattr(node, "_set_status_results", self._status)

    async def _save_media(self, _generation_id: str, output_param: str, factory: Any, **kwargs: Any) -> bool:
        assert kwargs["kind"] == ArtifactKind.VIDEO
        assert kwargs["position"] == 0
        assert kwargs.get("output_file") is None
        if not self.mask_ok:
            self.node.parameter_output_values[output_param] = None
            self._status(was_successful=False, result_details="The generation finished, but the matte video failed.")
            return False
        self.node.parameter_output_values[output_param] = factory("{outputs}/mask.mp4", "mask.mp4")
        return True

    async def _artifact(self, _generation_id: str, *, kind: str | None = None, position: int = 0) -> HostedArtifact:
        self.resolved.append((kind, position))
        if not self.has_prores:
            msg = "no second video"
            raise HostedArtifactError(msg)
        return HostedArtifact(index=position, kind=str(kind), url=PRORES_URL)

    async def _save_artifact(self, _artifact: HostedArtifact, _generation_id: str, **kwargs: Any) -> Any:
        self.saved_to.append(kwargs.get("output_file"))
        if self.prores_save_error:
            raise self.prores_save_error
        return _SavedFile(location="{outputs}/rgba.mov", name="rgba.mov")

    def _status(self, *, was_successful: bool, result_details: str) -> None:
        self.statuses.append((was_successful, result_details))


@pytest.mark.asyncio
async def test_toggle_off_never_resolves_or_saves_the_prores(monkeypatch: pytest.MonkeyPatch) -> None:
    node = _node()
    harness = _Harness(node, monkeypatch)

    await node._parse_result({}, GENERATION_ID)

    assert harness.resolved == []
    assert harness.saved_to == []
    assert node.parameter_output_values["rgba_video"] is None
    assert node.parameter_output_values["mask_video"] is not None
    success, details = harness.statuses[-1]
    assert success is True
    assert "matte video was saved" in details
    assert "not requested" in details


@pytest.mark.asyncio
async def test_toggle_on_saves_both_with_the_prores_at_video_position_one(monkeypatch: pytest.MonkeyPatch) -> None:
    node = _node()
    node.set_parameter_value(TOGGLE, True)
    harness = _Harness(node, monkeypatch)

    await node._parse_result({}, GENERATION_ID)

    assert harness.resolved == [(ArtifactKind.VIDEO, 1)]
    assert harness.saved_to == [node._rgba_file]
    assert node.parameter_output_values["mask_video"].value == "{outputs}/mask.mp4"
    assert node.parameter_output_values["rgba_video"].value == "{outputs}/rgba.mov"
    success, details = harness.statuses[-1]
    assert success is True
    assert "matte video was saved" in details
    assert "ProRes video was saved" in details


@pytest.mark.asyncio
async def test_missing_prores_is_tolerated(monkeypatch: pytest.MonkeyPatch) -> None:
    node = _node()
    node.set_parameter_value(TOGGLE, True)
    harness = _Harness(node, monkeypatch)
    harness.has_prores = False

    await node._parse_result({}, GENERATION_ID)

    assert harness.saved_to == []
    assert node.parameter_output_values["rgba_video"] is None
    success, details = harness.statuses[-1]
    assert success is True
    assert "not produced" in details


@pytest.mark.asyncio
async def test_prores_save_failure_keeps_the_run_successful(monkeypatch: pytest.MonkeyPatch) -> None:
    node = _node()
    node.set_parameter_value(TOGGLE, True)
    harness = _Harness(node, monkeypatch)
    harness.prores_save_error = OSError("disk full")

    await node._parse_result({}, GENERATION_ID)

    assert node.parameter_output_values["mask_video"] is not None
    assert node.parameter_output_values["rgba_video"] is None
    success, details = harness.statuses[-1]
    assert success is True
    assert "could not be saved" in details
    assert "disk full" in details


@pytest.mark.asyncio
async def test_mask_save_failure_fails_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    node = _node()
    node.set_parameter_value(TOGGLE, True)
    harness = _Harness(node, monkeypatch)
    harness.mask_ok = False

    await node._parse_result({}, GENERATION_ID)

    assert harness.resolved == []
    assert node.parameter_output_values["mask_video"] is None
    assert harness.statuses[-1][0] is False


# --- external URL warning and timeout ---------------------------------------------------------


@pytest.mark.parametrize(
    ("video", "video_uri", "warns"),
    [
        (VIDEO_URL, VIDEO_URL, True),
        (VIDEO_URL, "https://cdn.example/uploaded.mp4", False),
        ("/local/clip.mp4", "https://cdn.example/uploaded.mp4", False),
        ("http://example.com/in.mp4", "http://example.com/in.mp4", False),
        ("data:video/mp4;base64,AAAA", "data:video/mp4;base64,AAAA", False),
    ],
)
def test_warning_fires_only_for_a_passed_through_external_url(
    monkeypatch: pytest.MonkeyPatch, video: str, video_uri: str, warns: bool
) -> None:
    node = _node()
    clear = MagicMock()
    monkeypatch.setattr(node.status_component, "clear_execution_status", clear)

    node._warn_if_proxy_downloads_external_url(video, video_uri)

    assert clear.called is warns
    if warns:
        assert "another website" in clear.call_args.kwargs["initial_message"]


def test_submit_timeout_outlasts_the_proxy_download_window() -> None:
    assert LTXVideoAlphaGen.SUBMIT_TIMEOUT_SECONDS > 150


@pytest.mark.asyncio
async def test_build_payload_probes_off_the_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    node = _node()
    node.set_parameter_value("video", VideoUrlArtifact(value=VIDEO_URL))
    threads: list[int] = []
    main_thread = threading.get_ident()

    def recording_validate(_video: Any) -> None:
        threads.append(threading.get_ident())

    monkeypatch.setattr(node, "_validate_video_input", recording_validate)
    monkeypatch.setattr(node, "_resolve_video_uri", AsyncMock(return_value=VIDEO_URL))
    monkeypatch.setattr(node, "_warn_if_proxy_downloads_external_url", lambda *_: None)

    payload = await node._build_payload()

    assert payload == {"video_uri": VIDEO_URL}
    assert threads
    assert threads[0] != main_thread


@pytest.mark.asyncio
async def test_real_selection_saves_matte_and_prores_to_their_own_outputs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Runs the real `_hosted_artifact` position selection over both hosted videos."""
    node = _node()
    node.set_parameter_value("output_rgba_prores", True)
    hosted = [
        HostedArtifact(index=0, kind=ArtifactKind.VIDEO, url=MASK_URL, content_type="video/mp4"),
        HostedArtifact(index=1, kind=ArtifactKind.VIDEO, url=PRORES_URL, content_type="video/quicktime"),
    ]
    monkeypatch.setattr(node, "_hosted_artifacts", AsyncMock(return_value=hosted))
    saved: dict[str, str] = {}

    async def fake_save(artifact: HostedArtifact, _generation_id: str, *, output_file: Any = None) -> _SavedFile:
        target = "rgba" if output_file is node._rgba_file else "matte"
        saved[target] = artifact.url
        return _SavedFile(location=f"{{outputs}}/{target}", name=target)

    monkeypatch.setattr(node, "_save_artifact_to_project", fake_save)
    monkeypatch.setattr(node, "_download_artifact", AsyncMock(side_effect=AssertionError("hosted save path")))

    async def fake_save_generated_media(generation_id: str, output_param: str, factory: Any, **kwargs: Any) -> bool:
        artifact = await node._hosted_artifact(generation_id, kind=kwargs["kind"], position=kwargs["position"])
        result = await fake_save(artifact, generation_id)
        node.parameter_output_values[output_param] = factory(result.location, result.name)
        return True

    monkeypatch.setattr(node, "_save_generated_media", fake_save_generated_media)

    await node._parse_result({}, GENERATION_ID)

    assert saved == {"matte": MASK_URL, "rgba": PRORES_URL}


def test_direct_download_reads_the_live_result_keys() -> None:
    """The live API's keys differ from the preview docs (seen on a real job, 2026-11-05)."""
    result = {"result": {"matte_video_url": MASK_URL, "cutout_video_url": PRORES_URL}}

    assert _node()._direct_download_result_urls(result) == [
        (ArtifactKind.VIDEO, MASK_URL),
        (ArtifactKind.VIDEO, PRORES_URL),
    ]


def test_direct_download_prefers_live_keys_when_both_spellings_are_present() -> None:
    result = {"result": {"matte_video_url": MASK_URL, "video_url": "https://other/old.mp4"}}

    assert _node()._direct_download_result_urls(result) == [(ArtifactKind.VIDEO, MASK_URL)]
