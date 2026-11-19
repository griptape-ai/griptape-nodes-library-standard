"""Tests for the GLB to USD node: parameters, saving, Succeeded/Failed routing, menu metadata."""

from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from griptape_nodes.files.file import File
from PIL import Image
from pxr import Usd

from griptape_nodes_library.three_d import glb_to_usd as glb_to_usd_module
from griptape_nodes_library.three_d.glb_to_usd import GlbToUsd
from griptape_nodes_library.three_d.three_d_artifact import ThreeDUrlArtifact

from ._glb_builder import GlbSpec

if TYPE_CHECKING:
    from collections.abc import Callable

LIBRARY_JSON = Path(__file__).parents[3] / "griptape_nodes_library.json"


@pytest.fixture
def saved_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Route the node's project-file save into tmp_path instead of the workflow's outputs."""
    out = tmp_path / "saved"
    out.mkdir()

    def _destination(_workflow: object, filename: str, _situation: str, **_kwargs: object) -> SimpleNamespace:
        def _write(data: bytes) -> File:
            target = out / filename
            target.write_bytes(data)
            return File(str(target))

        return SimpleNamespace(write_bytes=_write)

    monkeypatch.setattr(glb_to_usd_module.ProjectFileDestination, "from_situation_in_workflow", _destination)
    return out


@pytest.fixture
def node() -> GlbToUsd:
    return GlbToUsd(name="glb_to_usd")


def _set(node: GlbToUsd, glb: Path, **values: str) -> None:
    node.set_parameter_value("model", ThreeDUrlArtifact(value=str(glb)))
    for name, value in values.items():
        node.set_parameter_value(name, value)


class TestParameters:
    def test_dropdowns_are_wireable(self, node: GlbToUsd) -> None:
        for name, choices in (
            ("output_format", ["usdz", "usdc", "usda"]),
            ("material_mode", ["UsdPreviewSurface", "MaterialX", "Both"]),
        ):
            param = node.get_parameter_by_name(name)
            assert param is not None
            assert param.ui_options["simple_dropdown"] == choices
            assert {m.name for m in param.allowed_modes} == {"INPUT", "PROPERTY"}

    def test_defaults(self, node: GlbToUsd) -> None:
        assert node.get_parameter_value("output_format") == "usdz"
        assert node.get_parameter_value("material_mode") == "UsdPreviewSurface"

    def test_control_outputs(self, node: GlbToUsd) -> None:
        names = {p.name for p in node.parameters}
        assert {"exec_in", "exec_out", "failure"} <= names


class TestProcess:
    def test_usdz(self, node: GlbToUsd, make_glb: Callable[..., Path], saved_dir: Path) -> None:
        _set(node, make_glb(), output_format="usdz", material_mode="Both", output_file="helmet.usdz")
        node.process()

        assert node.get_parameter_value("was_successful") is True
        artifact = node.parameter_output_values["usd_model"]
        assert isinstance(artifact, ThreeDUrlArtifact)
        assert Path(artifact.value) == saved_dir / "helmet.usdz"
        assert Usd.Stage.Open(artifact.value).GetDefaultPrim().GetPath() == "/Root"
        assert node.get_next_control_output() is node.control_parameter_out

    def test_output_extension_follows_format(
        self, node: GlbToUsd, make_glb: Callable[..., Path], saved_dir: Path
    ) -> None:
        _set(node, make_glb(), output_format="usda", output_file="model.usdz")
        node.process()
        assert (saved_dir / "model.usda").is_file()

    def test_usdc_writes_textures_beside_file(
        self, node: GlbToUsd, make_glb: Callable[..., Path], saved_dir: Path
    ) -> None:
        _set(node, make_glb(), output_format="usdc", output_file="model.usdc")
        node.process()

        texture_dirs = [p for p in saved_dir.iterdir() if p.is_dir()]
        assert len(texture_dirs) == 1
        assert texture_dirs[0].name.startswith("model_textures_")
        stage = Usd.Stage.Open(str(saved_dir / "model.usdc"))
        assets = [
            a.Get() for p in stage.Traverse() for a in p.GetAttributes() if str(a.GetTypeName()) == "asset" and a.Get()
        ]
        assert assets
        assert all(a.resolvedPath for a in assets)

    def test_summary_lists_warnings(self, node: GlbToUsd, make_glb: Callable[..., Path], saved_dir: Path) -> None:
        _set(node, make_glb(spec=GlbSpec(with_camera=True)))
        node.process()
        details = node.get_parameter_value("result_details")
        assert "Cameras were ignored." in details
        assert "2 triangles" in details


class TestFailures:
    def test_missing_model_fails_validation(self, node: GlbToUsd) -> None:
        errors = node.validate_before_node_run() or []
        assert any("no GLB/glTF model" in str(e) for e in errors)

    def test_bad_wired_value_fails_validation(self, node: GlbToUsd, make_glb: Callable[..., Path]) -> None:
        _set(node, make_glb())
        node.parameter_values["output_format"] = "fbx"  # a wired value the dropdown trait never saw
        errors = node.validate_before_node_run() or []
        assert any("Output Format 'fbx'" in str(e) for e in errors)

    def test_bad_input_without_failure_connection_raises(self, node: GlbToUsd, make_glb: Callable[..., Path]) -> None:
        _set(node, make_glb())
        node.parameter_values["material_mode"] = "Arnold"
        with pytest.raises(ValueError, match="Material Type 'Arnold'"):
            node.process()
        assert node.get_parameter_value("was_successful") is False
        assert node.get_next_control_output() is node.failure_output

    def test_bad_input_with_failure_connection_routes_to_failed(
        self, node: GlbToUsd, make_glb: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(node, "_has_outgoing_connections", lambda parameter: parameter is node.failure_output)
        _set(node, make_glb())
        node.parameter_values["output_format"] = "fbx"

        assert not (node.validate_before_node_run() or [])  # doesn't block the run
        node.process()  # doesn't raise
        assert node.get_parameter_value("was_successful") is False
        assert node.get_next_control_output() is node.failure_output

    def test_unreadable_glb_names_the_file(
        self, node: GlbToUsd, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(node, "_has_outgoing_connections", lambda parameter: parameter is node.failure_output)
        bad = tmp_path / "broken_model.glb"
        bad.write_bytes(b"not a glTF")
        _set(node, bad)
        node.process()
        assert "broken_model" in node.get_parameter_value("result_details")
        assert node.get_next_control_output() is node.failure_output


class TestReviewFixes:
    def test_truncated_glb_routes_to_failed(
        self, node: GlbToUsd, make_glb: Callable[..., Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(node, "_has_outgoing_connections", lambda parameter: parameter is node.failure_output)
        truncated = tmp_path / "truncated.glb"
        truncated.write_bytes(make_glb().read_bytes()[:12])  # struct.error inside pygltflib
        _set(node, truncated)
        node.process()  # must not raise
        assert node.get_parameter_value("was_successful") is False
        assert node.get_next_control_output() is node.failure_output

    def test_external_uri_routes_to_failed(
        self, node: GlbToUsd, make_glb: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(node, "_has_outgoing_connections", lambda parameter: parameter is node.failure_output)
        _set(node, make_glb(spec=GlbSpec(base_image_uri="/etc/passwd")))
        node.process()
        assert "Only self-contained models are supported" in node.get_parameter_value("result_details")
        assert node.get_next_control_output() is node.failure_output

    def test_textures_written_without_extension_coercion(
        self, node: GlbToUsd, make_glb: Callable[..., Path], saved_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[dict[str, object]] = []
        original = File.write_bytes

        def _spy(self: File, content: bytes, **kwargs: object) -> object:
            calls.append(kwargs)
            return original(self, content, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(File, "write_bytes", _spy)
        jpeg = io.BytesIO()
        Image.new("RGB", (2, 2), (10, 20, 30)).save(jpeg, format="JPEG")
        spec = GlbSpec(alpha_mode="BLEND", base_image=jpeg.getvalue(), base_image_mime=None)
        _set(node, make_glb(spec=spec), output_format="usdc", output_file="model.usdc")
        node.process()

        assert node.get_parameter_value("was_successful") is True
        assert calls
        assert all(c.get("coerce_extension_to_match_bytes") is False for c in calls)
        stage = Usd.Stage.Open(str(saved_dir / "model.usdc"))
        assets = [
            a.Get() for p in stage.Traverse() for a in p.GetAttributes() if str(a.GetTypeName()) == "asset" and a.Get()
        ]
        assert any(a.path.endswith(".jpg") for a in assets)
        assert all(a.resolvedPath for a in assets)


def test_library_entry_is_beta() -> None:
    nodes = json.loads(LIBRARY_JSON.read_text())["nodes"]
    entry = next(n for n in nodes if n["class_name"] == "GlbToUsd")
    assert entry["file_path"] == "griptape_nodes_library/three_d/glb_to_usd.py"
    assert {"type": "lifecycle_stage", "stage": "BETA"} in entry["metadata"]["declarations"]
