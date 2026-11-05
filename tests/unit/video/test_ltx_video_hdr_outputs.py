"""LTX Video to Video HDR: models, colour space, outputs, direct download and the size estimate."""

from __future__ import annotations

import dataclasses
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from griptape_nodes.exe_types.core_types import NodeError, Parameter
from griptape_nodes.traits.options import Options

from griptape_nodes_library.media import exr_sequence
from griptape_nodes_library.proxy.hosted_artifacts import ArtifactKind, HostedArtifact, HostedArtifactError
from griptape_nodes_library.proxy.output_size import estimate_exr_zip_bytes
from griptape_nodes_library.video import ltx_video_to_video_hdr as hdr_module
from griptape_nodes_library.video.ltx_video_to_video_hdr import LTXVideoToVideoHDR

GENERATION_ID = "gen-1"
ZIP_URL = "https://provider.example/frames.zip?sig=x"
MP4_URL = "https://provider.example/preview.mp4?sig=x"
COLOURSPACE = "output_colorspace"


def _node() -> LTXVideoToVideoHDR:
    return LTXVideoToVideoHDR(name="HDR")


def _param(node: LTXVideoToVideoHDR, name: str) -> Parameter:
    param = node.get_parameter_by_name(name)
    assert param is not None
    return param


def _is_hidden(node: LTXVideoToVideoHDR) -> bool:
    return bool(_param(node, COLOURSPACE).ui_options.get("hide"))


# --- model dropdown ---------------------------------------------------------------------------


def test_dropdown_offers_both_models_and_defaults_to_the_new_one() -> None:
    node = _node()
    param = _param(node, "model")

    assert param.default_value == "ltx-hdr-2"
    assert node.get_parameter_value("model") == "ltx-hdr-2"
    rows = {row["name"]: row for row in param.ui_options["data"]}
    assert set(rows) == {"ltx-hdr-2", "ltx-2-3-pro"}
    assert "deprecated" in rows["ltx-2-3-pro"]["label"].lower()
    assert "deprecated" not in rows["ltx-hdr-2"].get("label", "").lower()


@pytest.mark.parametrize("saved", ["ltx-2-3-pro", "LTX 2.3 Pro", "gtc_ltx_2_3_pro"])
def test_legacy_saved_values_load_and_run_on_the_legacy_model(saved: str) -> None:
    node = _node()
    node.set_parameter_value("model", saved, initial_setup=True)

    assert node._selected_model() == "ltx-2-3-pro"
    assert node._get_api_model_id() == "ltx-2-3-pro:video-to-video-hdr"


# --- colour space visibility ------------------------------------------------------------------


def test_colourspace_dropdown_has_the_four_values_and_default() -> None:
    node = _node()
    param = _param(node, COLOURSPACE)

    assert param.default_value == "acescg"
    (options,) = param.find_elements_by_type(Options)
    assert options.choices == ["acescg", "acescct", "srgb_linear", "aces2065-1"]


def test_colourspace_visible_for_new_node_and_toggles_with_model() -> None:
    node = _node()
    assert not _is_hidden(node)

    node.set_parameter_value("model", "ltx-2-3-pro")
    assert _is_hidden(node)

    node.set_parameter_value("model", "ltx-hdr-2")
    assert not _is_hidden(node)


def test_colourspace_visibility_is_correct_on_workflow_load() -> None:
    # Load restores values with initial_setup=True, which skips after_value_set.
    node = _node()
    node.set_parameter_value("model", "ltx-2-3-pro", initial_setup=True)
    assert _is_hidden(node)

    node.set_parameter_value("model", "ltx-hdr-2", initial_setup=True)
    assert not _is_hidden(node)


# --- payload ----------------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "colourspace", "expected"),
    [
        ("ltx-hdr-2", "acescct", {"video_uri": "u", "output_colorspace": "acescct"}),
        ("ltx-hdr-2", "aces2065-1", {"video_uri": "u", "output_colorspace": "aces2065-1"}),
        ("ltx-2-3-pro", "acescct", {"video_uri": "u"}),
    ],
)
async def test_payload_sends_colourspace_only_for_the_new_model(
    monkeypatch: pytest.MonkeyPatch, model: str, colourspace: str, expected: dict[str, str]
) -> None:
    node = _node()
    node.set_parameter_value("model", model)
    node.set_parameter_value(COLOURSPACE, colourspace)
    node.set_parameter_value("video", "https://example.com/in.mp4")
    monkeypatch.setattr(node, "_validate_video_input", lambda _video: None)
    monkeypatch.setattr(node, "_resolve_video_uri", AsyncMock(return_value="u"))

    assert await node._build_payload() == expected


# --- direct download --------------------------------------------------------------------------


def test_supports_direct_download() -> None:
    node = _node()
    assert node.SUPPORTS_DIRECT_DOWNLOAD is True
    assert node.get_parameter_by_name("direct_download") is not None


def test_direct_download_urls_map_in_hosted_order() -> None:
    result = {"result": {"video_url": MP4_URL, "exr_frames_url": ZIP_URL}}

    assert _node()._direct_download_result_urls(result) == [
        (ArtifactKind.ARCHIVE, ZIP_URL),
        (ArtifactKind.VIDEO, MP4_URL),
    ]


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        ({"result": {"exr_frames_url": ZIP_URL}}, [(ArtifactKind.ARCHIVE, ZIP_URL)]),
        ({"result": {"video_url": MP4_URL}}, [(ArtifactKind.VIDEO, MP4_URL)]),
        ({"result": {"exr_frames_url": "", "video_url": None}}, []),
        ({"result": None}, []),
        ({"status": "completed"}, []),
        ({}, []),
    ],
)
def test_direct_download_urls_are_defensive(result: dict[str, Any], expected: list[tuple[str, str]]) -> None:
    assert _node()._direct_download_result_urls(result) == expected


# --- size estimate ----------------------------------------------------------------------------


def test_estimate_uses_the_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    node = _node()
    node.set_parameter_value("video", "https://example.com/in.mp4")
    monkeypatch.setattr(node, "_probe_video", lambda _url: (1920, 1080, 100))

    assert node._estimate_output_bytes() == estimate_exr_zip_bytes(100, 1920, 1080)


def test_estimate_is_none_when_the_probe_fails_or_there_is_no_video(monkeypatch: pytest.MonkeyPatch) -> None:
    node = _node()
    assert node._estimate_output_bytes() is None

    node.set_parameter_value("video", "https://example.com/in.mp4")
    monkeypatch.setattr(node, "_probe_video", lambda _url: None)
    assert node._estimate_output_bytes() is None


def test_estimate_and_validation_share_one_probe_per_run(monkeypatch: pytest.MonkeyPatch) -> None:
    node = _node()
    node.set_parameter_value("video", "https://example.com/in.mp4")
    run = MagicMock(return_value=subprocess.CompletedProcess([], 0, stdout="{}"))
    monkeypatch.setattr(hdr_module.subprocess, "run", run)
    monkeypatch.setattr(hdr_module.run, "get_or_fetch_platform_executables_else_raise", lambda: ("ffmpeg", "ffprobe"))
    probe = MagicMock(return_value=(1280, 720, 10))
    monkeypatch.setattr(node, "_probe_video", probe)
    node._probe_cache = {}

    node._estimate_output_bytes()
    assert node._validate_video_input(node.get_parameter_value("video")) is None

    probe.assert_called_once()


# --- _parse_result ----------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _SavedFile:
    location: str
    name: str


class _Harness:
    """Records the order of saves and sequence writes for `_parse_result`."""

    def __init__(self, node: LTXVideoToVideoHDR, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        self.events: list[str] = []
        self.statuses: list[tuple[bool, str]] = []
        self.node = node
        self.zip_path = tmp_path / "frames.zip"
        self.zip_path.write_bytes(b"zip")
        self.has_mp4 = True
        self.save_zip_ok = True
        self.unpack_error: Exception | None = None
        self.preview_save_error: Exception | None = None
        self.sequence = object()

        monkeypatch.setattr(node, "_save_generated_media", self._save)
        monkeypatch.setattr(node, "_save_artifact_to_project", self._save_artifact)
        monkeypatch.setattr(node, "_hosted_artifact", self._artifact)
        monkeypatch.setattr(node, "_set_status_results", self._status)
        monkeypatch.setattr(hdr_module.File, "resolve", lambda _self: str(self.zip_path))
        monkeypatch.setattr(exr_sequence, "write_exr_zip_to_sequence", self._write_sequence)
        monkeypatch.setattr(node._exr_outputs.file_sequence, "build_sequence", lambda **_: "DEST")

    async def _artifact(self, _generation_id: str, *, kind: str | None = None, position: int = 0) -> HostedArtifact:
        if kind == ArtifactKind.VIDEO and not self.has_mp4:
            msg = "no video"
            raise HostedArtifactError(msg)
        return HostedArtifact(index=1, kind=str(kind), url=MP4_URL)

    async def _save(self, _generation_id: str, output_param: str, factory: Any, **kwargs: Any) -> bool:
        self.events.append(f"save:{output_param}:{kwargs.get('output_file') is self.node._preview_file}")
        if output_param == "project_path":
            if not self.save_zip_ok:
                return False
            self.node.parameter_output_values[output_param] = "{outputs}/frames.zip"
        else:
            self.node.parameter_output_values[output_param] = factory("{outputs}/p.mp4", "p.mp4")
        return True

    async def _save_artifact(self, artifact: HostedArtifact, _generation_id: str, **kwargs: Any) -> Any:
        self.events.append(f"save:preview_video:{kwargs.get('output_file') is self.node._preview_file}")
        if self.preview_save_error:
            raise self.preview_save_error
        return _SavedFile(location="{outputs}/p.mp4", name="p.mp4")

    async def _write_sequence(self, source: Any, dest: Any) -> object:
        self.events.append(f"unpack:{source}:{dest}")
        if self.unpack_error:
            raise self.unpack_error
        return self.sequence

    def _status(self, *, was_successful: bool, result_details: str) -> None:
        self.statuses.append((was_successful, result_details))


@pytest.mark.asyncio
async def test_parse_result_saves_zip_then_mp4_then_unpacks_from_the_saved_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    node = _node()
    node.set_parameter_value("model", "ltx-hdr-2")
    node.set_parameter_value(COLOURSPACE, "srgb_linear")
    h = _Harness(node, monkeypatch, tmp_path)

    await node._parse_result({}, GENERATION_ID)

    assert h.events == [
        "save:project_path:False",
        "save:preview_video:True",
        f"unpack:{h.zip_path}:DEST",
    ]
    out = node.parameter_output_values
    assert out["exr_sequence"] is h.sequence
    assert out["colour_space"] == "Linear Rec.709 (sRGB)"
    assert out["preview_video"].value == "{outputs}/p.mp4"
    assert h.statuses[-1][0] is True


@pytest.mark.asyncio
async def test_parse_result_uses_acescg_for_the_legacy_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    node = _node()
    node.set_parameter_value("model", "ltx-2-3-pro")
    node.set_parameter_value(COLOURSPACE, "srgb_linear")
    _Harness(node, monkeypatch, tmp_path)

    await node._parse_result({}, GENERATION_ID)

    assert node.parameter_output_values["colour_space"] == "ACEScg"


@pytest.mark.asyncio
async def test_missing_mp4_is_tolerated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    node = _node()
    h = _Harness(node, monkeypatch, tmp_path)
    h.has_mp4 = False

    await node._parse_result({}, GENERATION_ID)

    assert not any(event.startswith("save:preview_video") for event in h.events)
    assert node.parameter_output_values["preview_video"] is None
    assert node.parameter_output_values["exr_sequence"] is h.sequence
    assert all(ok for ok, _ in h.statuses)


@pytest.mark.asyncio
async def test_unpack_failure_marks_the_run_failed_but_keeps_the_zip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    node = _node()
    h = _Harness(node, monkeypatch, tmp_path)
    h.unpack_error = NodeError("The downloaded file is not a valid ZIP archive.")

    await node._parse_result({}, GENERATION_ID)

    out = node.parameter_output_values
    assert out["project_path"] == "{outputs}/frames.zip"
    assert out["preview_video"] is not None
    assert out["exr_sequence"] is None
    ok, details = h.statuses[-1]
    assert ok is False
    assert "not a valid ZIP archive" in details
    assert "EXR ZIP was saved" in details


@pytest.mark.asyncio
async def test_zip_save_failure_skips_everything_else(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    node = _node()
    h = _Harness(node, monkeypatch, tmp_path)
    h.save_zip_ok = False

    await node._parse_result({}, GENERATION_ID)

    assert h.events == ["save:project_path:False"]


@pytest.mark.asyncio
async def test_success_message_does_not_claim_a_missing_preview(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    node = _node()
    h = _Harness(node, monkeypatch, tmp_path)
    h.has_mp4 = False

    await node._parse_result({}, GENERATION_ID)

    was_successful, details = h.statuses[-1]
    assert was_successful is True
    assert "has no preview video" in details
    assert "preview video was saved" not in details
    assert node.parameter_output_values["preview_video"] is None


@pytest.mark.asyncio
async def test_failed_preview_keeps_the_run_successful_and_says_why(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    node = _node()
    h = _Harness(node, monkeypatch, tmp_path)
    h.preview_save_error = OSError("disk full")

    await node._parse_result({}, GENERATION_ID)

    was_successful, details = h.statuses[-1]
    assert was_successful is True
    assert "preview video could not be saved: disk full" in details
    assert node.parameter_output_values["exr_sequence"] is h.sequence
    assert node.parameter_output_values["preview_video"] is None


@pytest.mark.asyncio
async def test_preview_is_resolved_only_once(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    node = _node()
    h = _Harness(node, monkeypatch, tmp_path)
    lookups: list[str | None] = []
    real_artifact = h._artifact

    async def counting_artifact(generation_id: str, *, kind: str | None = None, position: int = 0) -> HostedArtifact:
        lookups.append(kind)
        return await real_artifact(generation_id, kind=kind, position=position)

    monkeypatch.setattr(node, "_hosted_artifact", counting_artifact)

    await node._parse_result({}, GENERATION_ID)

    assert lookups.count(ArtifactKind.VIDEO) == 1
