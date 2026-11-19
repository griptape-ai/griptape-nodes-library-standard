"""Tests for the GLB/glTF -> USD converter behind the GLB to USD node."""

from __future__ import annotations

import io
import zipfile
from typing import TYPE_CHECKING

import numpy as np
import pytest
from PIL import Image
from pxr import Gf, Usd, UsdGeom, UsdShade

from griptape_nodes_library.three_d._gltf_to_usd import (
    GltfConversionError,
    MaterialMode,
    _triangulate,
    convert_gltf_to_usd,
)

from ._glb_builder import CHILD_SCALE, PARENT_TRANSLATION, QUAD_POSITIONS, QUAD_UVS, GlbSpec

if TYPE_CHECKING:
    from pathlib import Path

MATERIALX_NODE_IDS = {
    "ND_standard_surface_surfaceshader",
    "ND_image_color3",
    "ND_image_color4",
    "ND_image_vector3",
    "ND_convert_color4_color3",
    "ND_convert_float_color3",
    "ND_extract_color4",
    "ND_extract_vector3",
    "ND_multiply_color3",
    "ND_multiply_float",
    "ND_normalmap_float",
    "ND_geompropvalue_vector2",
    "ND_ifgreatereq_float",
}


def _convert(make_glb, tmp_path: Path, suffix: str = "usda", *, mode=MaterialMode.PREVIEW_SURFACE, spec=None):
    glb = make_glb(spec=spec)
    result = convert_gltf_to_usd(glb, tmp_path / "out" / f"model.{suffix}", material_mode=mode)
    return result, Usd.Stage.Open(str(result.output_path))


def _material(stage: Usd.Stage) -> UsdShade.Material:
    return UsdShade.Material(stage.GetPrimAtPath("/Root/Materials/QuadMaterial"))


def _mesh(stage: Usd.Stage) -> UsdGeom.Mesh:
    return UsdGeom.Mesh(stage.GetPrimAtPath("/Root/Geom/Parent/Child/Quad"))


def _source_of(shader_input: UsdShade.Input | UsdShade.Output) -> tuple[str, str]:
    sources, _ = shader_input.GetConnectedSources()
    assert sources, f"{shader_input.GetFullName()} is not connected"
    return sources[0].source.GetPath().name, sources[0].sourceName


class TestStage:
    def test_stage_metadata(self, make_glb, tmp_path: Path) -> None:
        _, stage = _convert(make_glb, tmp_path)
        assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.y
        assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0
        assert stage.GetDefaultPrim().GetPath() == "/Root"
        assert Usd.ModelAPI(stage.GetDefaultPrim()).GetKind() == "component"


class TestGeometry:
    def test_mesh_topology_and_primvars(self, make_glb, tmp_path: Path) -> None:
        result, stage = _convert(make_glb, tmp_path)
        mesh = _mesh(stage)
        assert (result.mesh_count, result.triangle_count) == (1, 2)
        assert list(mesh.GetFaceVertexCountsAttr().Get()) == [3, 3]
        assert list(mesh.GetFaceVertexIndicesAttr().Get()) == [0, 1, 2, 0, 2, 3]
        np.testing.assert_allclose(np.array(mesh.GetPointsAttr().Get()), QUAD_POSITIONS)
        assert mesh.GetSubdivisionSchemeAttr().Get() == UsdGeom.Tokens.none
        assert mesh.GetNormalsInterpolation() == UsdGeom.Tokens.vertex

    def test_uvs_are_flipped_to_usd_origin(self, make_glb, tmp_path: Path) -> None:
        _, stage = _convert(make_glb, tmp_path)
        st = UsdGeom.PrimvarsAPI(_mesh(stage)).GetPrimvar("st")
        assert st.GetInterpolation() == UsdGeom.Tokens.vertex
        expected = QUAD_UVS.copy()
        expected[:, 1] = 1.0 - expected[:, 1]
        np.testing.assert_allclose(np.array(st.Get()), expected)

    def test_node_transforms_compose(self, make_glb, tmp_path: Path) -> None:
        _, stage = _convert(make_glb, tmp_path)
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        box = cache.ComputeWorldBound(stage.GetDefaultPrim()).ComputeAlignedRange()
        lo = np.array(PARENT_TRANSLATION) + QUAD_POSITIONS.min(axis=0) * CHILD_SCALE
        hi = np.array(PARENT_TRANSLATION) + QUAD_POSITIONS.max(axis=0) * CHILD_SCALE
        np.testing.assert_allclose(np.array(box.GetMin()), lo)
        np.testing.assert_allclose(np.array(box.GetMax()), hi)

    @pytest.mark.parametrize(
        ("mode", "indices", "expected"),
        [
            (4, [0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5]),  # trailing partial triangle dropped
            (5, [0, 1, 2, 3], [0, 1, 2, 2, 1, 3]),  # strip: odd triangle winding flipped
            (6, [0, 1, 2, 3], [0, 1, 2, 0, 2, 3]),  # fan
            (5, [0, 1], []),
        ],
    )
    def test_triangulate(self, mode: int, indices: list[int], expected: list[int]) -> None:
        assert _triangulate(np.array(indices, dtype=np.int64), mode).tolist() == expected

    def test_triangulate_rejects_non_triangle_modes(self) -> None:
        with pytest.raises(ValueError, match="Unsupported glTF primitive mode"):
            _triangulate(np.array([0, 1, 2]), 1)

    def test_triangle_strip_primitive(self, make_glb, tmp_path: Path) -> None:
        spec = GlbSpec(mode=5, indices=np.array([0, 1, 3, 2], dtype=np.uint16))
        result, _ = _convert(make_glb, tmp_path, spec=spec)
        assert result.triangle_count == 2


class TestPreviewSurface:
    def test_network(self, make_glb, tmp_path: Path) -> None:
        _, stage = _convert(make_glb, tmp_path)
        material = _material(stage)
        surface = UsdShade.Shader(material.GetPrim().GetChild("PreviewSurface"))
        assert surface.GetIdAttr().Get() == "UsdPreviewSurface"
        assert _source_of(material.GetSurfaceOutput()) == ("PreviewSurface", "surface")
        assert _source_of(surface.GetInput("diffuseColor")) == ("BaseColorTex", "rgb")
        # glTF packs roughness in G and metallic in B.
        assert _source_of(surface.GetInput("roughness")) == ("MetalRoughTex", "g")
        assert _source_of(surface.GetInput("metallic")) == ("MetalRoughTex", "b")
        assert _source_of(surface.GetInput("normal")) == ("NormalTex", "rgb")
        assert _source_of(surface.GetInput("emissiveColor")) == ("EmissiveTex", "rgb")

    def test_texture_settings(self, make_glb, tmp_path: Path) -> None:
        _, stage = _convert(make_glb, tmp_path)
        prim = _material(stage).GetPrim()

        def tex(name: str) -> UsdShade.Shader:
            return UsdShade.Shader(prim.GetChild(name))

        assert tex("BaseColorTex").GetInput("sourceColorSpace").Get() == "sRGB"
        assert tex("MetalRoughTex").GetInput("sourceColorSpace").Get() == "raw"
        assert tex("MetalRoughTex").GetInput("scale").Get() == Gf.Vec4f(1.0, 0.25, 0.5, 1.0)
        # Normal: n = (2c - 1) * (scale, scale, 1) with normalTexture.scale = 0.5
        assert tex("NormalTex").GetInput("scale").Get() == Gf.Vec4f(1.0, 1.0, 2.0, 1.0)
        assert tex("NormalTex").GetInput("bias").Get() == Gf.Vec4f(-0.5, -0.5, -1.0, 0.0)
        assert tex("EmissiveTex").GetInput("scale").Get() == Gf.Vec4f(1.0, 0.5, 0.25, 1.0)
        assert _source_of(tex("BaseColorTex").GetInput("st")) == ("Reader_st", "result")

    def test_opaque_ignores_alpha(self, make_glb, tmp_path: Path) -> None:
        result, stage = _convert(make_glb, tmp_path, spec=GlbSpec(alpha_mode="OPAQUE"))
        surface = UsdShade.Shader(_material(stage).GetPrim().GetChild("PreviewSurface"))
        assert not surface.GetInput("opacity")
        base = next(p for p in result.texture_paths if p.name.startswith("base_"))
        assert base.name.endswith("_rgb.png")
        assert Image.open(base).mode == "RGB"

    def test_mask_sets_threshold(self, make_glb, tmp_path: Path) -> None:
        _, stage = _convert(make_glb, tmp_path, spec=GlbSpec(alpha_mode="MASK"))
        surface = UsdShade.Shader(_material(stage).GetPrim().GetChild("PreviewSurface"))
        assert _source_of(surface.GetInput("opacity")) == ("BaseColorTex", "a")
        assert surface.GetInput("opacityThreshold").Get() == pytest.approx(0.4)

    def test_blend_keeps_alpha_texture(self, make_glb, tmp_path: Path) -> None:
        result, _ = _convert(make_glb, tmp_path, spec=GlbSpec(alpha_mode="BLEND"))
        base = next(p for p in result.texture_paths if p.name.startswith("base_"))
        assert Image.open(base).mode == "RGBA"

    def test_untextured_uses_factors(self, make_glb, tmp_path: Path) -> None:
        _, stage = _convert(make_glb, tmp_path, spec=GlbSpec(textured=False))
        surface = UsdShade.Shader(_material(stage).GetPrim().GetChild("PreviewSurface"))
        assert surface.GetInput("metallic").Get() == pytest.approx(0.5)
        assert surface.GetInput("roughness").Get() == pytest.approx(0.25)
        assert surface.GetInput("diffuseColor").Get() == Gf.Vec3f(1.0, 1.0, 1.0)


class TestMaterialX:
    def test_only_mtlx_context(self, make_glb, tmp_path: Path) -> None:
        _, stage = _convert(make_glb, tmp_path, mode=MaterialMode.MATERIALX)
        prim = _material(stage).GetPrim()
        assert prim.GetAttribute("outputs:mtlx:surface").HasAuthoredConnections()
        # outputs:surface is a built-in Material schema attribute; it just mustn't be wired.
        assert not prim.GetAttribute("outputs:surface").HasAuthoredConnections()

    def test_uses_known_standard_library_nodes(self, make_glb, tmp_path: Path) -> None:
        _, stage = _convert(make_glb, tmp_path, mode=MaterialMode.MATERIALX, spec=GlbSpec(alpha_mode="MASK"))
        ids = {UsdShade.Shader(p).GetIdAttr().Get() for p in _material(stage).GetPrim().GetChildren()}
        assert ids <= MATERIALX_NODE_IDS
        assert "ND_standard_surface_surfaceshader" in ids

    def test_surface_wiring(self, make_glb, tmp_path: Path) -> None:
        _, stage = _convert(make_glb, tmp_path, mode=MaterialMode.MATERIALX)
        prim = _material(stage).GetPrim()
        surface = UsdShade.Shader(prim.GetChild("mtlx_Surface"))
        assert surface.GetInput("base").Get() == 1.0  # standard_surface defaults to 0.8
        assert _source_of(surface.GetInput("base_color")) == ("mtlx_BaseColorRGB", "out")
        assert _source_of(surface.GetInput("metalness")) == ("mtlx_MetallicFactor", "out")
        assert _source_of(surface.GetInput("specular_roughness")) == ("mtlx_RoughnessFactor", "out")
        assert _source_of(surface.GetInput("normal")) == ("mtlx_NormalMap", "out")
        base_image = UsdShade.Shader(prim.GetChild("mtlx_BaseColorImage"))
        assert base_image.GetInput("file").GetAttr().GetColorSpace() == "srgb_texture"
        assert _source_of(base_image.GetInput("texcoord")) == ("mtlx_TexCoord_st", "out")

    def test_both_contexts(self, make_glb, tmp_path: Path) -> None:
        _, stage = _convert(make_glb, tmp_path, mode=MaterialMode.BOTH)
        prim = _material(stage).GetPrim()
        assert prim.GetAttribute("outputs:surface").HasAuthoredConnections()
        assert prim.GetAttribute("outputs:mtlx:surface").HasAuthoredConnections()

    def test_skips_unused_occlusion_texture(self, make_glb, tmp_path: Path) -> None:
        result, _ = _convert(make_glb, tmp_path, mode=MaterialMode.MATERIALX)
        assert all(p.exists() for p in result.texture_paths)


class TestOutputFormats:
    @pytest.mark.parametrize("suffix", ["usda", "usdc"])
    def test_textures_beside_layer_resolve(self, make_glb, tmp_path: Path, suffix: str) -> None:
        result, stage = _convert(make_glb, tmp_path, suffix)
        assert result.texture_paths
        assert all(p.parent == tmp_path / "out" / "model_textures" for p in result.texture_paths)
        for prim in stage.Traverse():
            shader = UsdShade.Shader(prim)
            if shader and shader.GetInput("file"):
                assert shader.GetInput("file").Get().resolvedPath

    def test_usdz_is_self_contained(self, make_glb, tmp_path: Path) -> None:
        result, stage = _convert(make_glb, tmp_path, "usdz", mode=MaterialMode.BOTH)
        assert result.texture_paths == []
        assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["model.usdz"]
        names = zipfile.ZipFile(result.output_path).namelist()
        assert names[0].endswith(".usdc")
        assert any(n.endswith(".png") for n in names)
        file_input = UsdShade.Shader(_material(stage).GetPrim().GetChild("BaseColorTex")).GetInput("file")
        assert file_input.Get().resolvedPath

    def test_webp_textures_become_png(self, make_glb, tmp_path: Path) -> None:
        result, _ = _convert(make_glb, tmp_path, spec=GlbSpec(image_format="WEBP", alpha_mode="BLEND"))
        assert {p.suffix for p in result.texture_paths} == {".png"}
        assert Image.open(io.BytesIO(result.texture_paths[0].read_bytes())).format == "PNG"


class TestErrorsAndWarnings:
    def test_draco_is_rejected(self, make_glb, tmp_path: Path) -> None:
        glb = make_glb(spec=GlbSpec(extensions_required=["KHR_draco_mesh_compression"]))
        with pytest.raises(GltfConversionError, match="Draco"):
            convert_gltf_to_usd(glb, tmp_path / "out.usdz")

    def test_not_a_gltf(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.glb"
        bad.write_bytes(b"definitely not a glTF")
        with pytest.raises(GltfConversionError, match=r"bad\.glb"):
            convert_gltf_to_usd(bad, tmp_path / "out.usdz")

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(GltfConversionError, match="does not exist"):
            convert_gltf_to_usd(tmp_path / "missing.glb", tmp_path / "out.usdz")

    def test_unsupported_features_are_reported(self, make_glb, tmp_path: Path) -> None:
        result, _ = _convert(make_glb, tmp_path, spec=GlbSpec(with_camera=True))
        assert "Cameras were ignored." in result.warnings
