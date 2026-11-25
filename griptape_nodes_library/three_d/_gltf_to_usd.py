"""GLB / glTF 2.0 -> USD converter used by the GLB to USD node.

Kept free of Griptape imports so it can be unit tested on its own.

Materials are authored as UsdPreviewSurface, MaterialX (standard_surface), or both on the
same UsdShade.Material (separate `surface` and `mtlx:surface` render contexts).

Scope (prototype):
  * Static triangle geometry, node hierarchy and transforms, normals, UV sets, vertex colour.
  * glTF metallic-roughness PBR: base colour (+alpha), metallic/roughness, normal,
    occlusion (UsdPreviewSurface only), emissive (+KHR_materials_emissive_strength), ior.
  * Self-contained input only: a .glb, or a .gltf whose buffers and images are data: URIs.
    External file URIs are rejected rather than resolved, so a crafted file can't read
    arbitrary paths into the output.
  * Image format is sniffed from the bytes (the declared mimeType may be missing or wrong).
    PNG/JPEG are kept; any other format Pillow can read is re-encoded to PNG.
  * Not converted (reported as warnings): skinning, animation, morph targets, cameras,
    lights, KHR_texture_transform, KHR_materials_unlit, Draco/meshopt compression, KTX2.
"""

from __future__ import annotations

import base64
import io
import struct
import tempfile
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from pathlib import Path

import numpy as np
import pygltflib
from PIL import Image as PILImage
from PIL import UnidentifiedImageError
from pxr import Gf, Kind, Sdf, Tf, Usd, UsdGeom, UsdShade, UsdUtils, Vt


class MaterialMode(StrEnum):
    PREVIEW_SURFACE = "UsdPreviewSurface"
    MATERIALX = "MaterialX"
    BOTH = "Both"


class OutputFormat(StrEnum):
    USDZ = "usdz"
    USDC = "usdc"
    USDA = "usda"


@dataclass
class ConversionResult:
    output_path: Path
    texture_paths: list[Path] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    mesh_count: int = 0
    material_count: int = 0
    triangle_count: int = 0


class GltfConversionError(Exception):
    pass


# --- glTF constants ---------------------------------------------------------------------------

_COMPONENT_DTYPES = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}
_NORMALIZE_DIVISORS = {np.int8: 127.0, np.uint8: 255.0, np.int16: 32767.0, np.uint16: 65535.0}
_TYPE_SIZES = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}


class _PrimitiveMode(IntEnum):
    TRIANGLES = 4
    TRIANGLE_STRIP = 5
    TRIANGLE_FAN = 6


_WRAP_PREVIEW = {10497: "repeat", 33071: "clamp", 33648: "mirror"}
_WRAP_MTLX = {10497: "periodic", 33071: "clamp", 33648: "mirror"}

_UNSUPPORTED_REQUIRED_EXTENSIONS = {
    "KHR_draco_mesh_compression": "Draco-compressed meshes",
    "EXT_meshopt_compression": "meshopt-compressed meshes",
    "KHR_texture_basisu": "KTX2/Basis textures",
}

_KTX2_MAGIC = b"\xabKTX 20\xbb\r\n\x1a\n"
_KEPT_IMAGE_FORMATS = {"PNG": "png", "JPEG": "jpg"}  # Pillow format -> file extension USD reads as-is

# Failures that malformed or truncated input can raise while parsing or decoding. Converted to
# GltfConversionError so callers (the node's Failed path) get one exception type.
_MALFORMED_INPUT_ERRORS = (
    struct.error,
    IndexError,
    KeyError,
    TypeError,
    ValueError,
    AttributeError,
    OSError,
    Tf.ErrorException,
)


# --- Public API -------------------------------------------------------------------------------


def convert_gltf_to_usd(
    input_path: Path,
    output_path: Path,
    *,
    material_mode: MaterialMode = MaterialMode.PREVIEW_SURFACE,
    texture_dir_name: str | None = None,
) -> ConversionResult:
    """Convert a .glb/.gltf file to .usdz, .usdc or .usda.

    For .usdc/.usda, textures are written to `<output dir>/<texture_dir_name>/` and referenced
    relatively. For .usdz they are packaged inside the archive.
    """
    output_format = OutputFormat(output_path.suffix.lower().lstrip("."))
    try:
        return _convert(input_path, output_path, output_format, material_mode, texture_dir_name)
    except GltfConversionError:
        raise
    except _MALFORMED_INPUT_ERRORS as e:
        msg = f"Attempted to convert {input_path.name} to USD. Failed with {type(e).__name__}: {e}"
        raise GltfConversionError(msg) from e


def _convert(
    input_path: Path,
    output_path: Path,
    output_format: OutputFormat,
    material_mode: MaterialMode,
    texture_dir_name: str | None,
) -> ConversionResult:
    gltf = _load_gltf(input_path)

    if output_format is OutputFormat.USDZ:
        with tempfile.TemporaryDirectory(prefix="gltf2usd_") as tmp:
            staging_layer = Path(tmp) / f"{output_path.stem}.usdc"
            result = _ConversionJob(gltf, staging_layer, material_mode, "textures").run()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if not UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(str(staging_layer)), str(output_path)):
                raise GltfConversionError(f"Attempted to package {output_path.name}. USD failed to create the usdz.")
            result.output_path = output_path
            result.texture_paths = []
            return result

    tex_dir = texture_dir_name or f"{output_path.stem}_textures"
    return _ConversionJob(gltf, output_path, material_mode, tex_dir).run()


# --- Loading ----------------------------------------------------------------------------------


def _load_gltf(input_path: Path) -> pygltflib.GLTF2:
    if not input_path.is_file():
        raise GltfConversionError(f"Attempted to read {input_path}. The file does not exist.")
    try:
        gltf = pygltflib.GLTF2().load(str(input_path))
    except _MALFORMED_INPUT_ERRORS as e:
        raise GltfConversionError(f"Attempted to read {input_path.name}. It is not a valid glTF/GLB file ({e}).") from e
    if gltf is None:
        raise GltfConversionError(f"Attempted to read {input_path.name}. It is not a valid glTF/GLB file.")

    blocked = [
        _UNSUPPORTED_REQUIRED_EXTENSIONS[e]
        for e in (gltf.extensionsRequired or [])
        if e in _UNSUPPORTED_REQUIRED_EXTENSIONS
    ]
    if blocked:
        raise GltfConversionError(
            f"Attempted to convert {input_path.name}. It uses {', '.join(blocked)}, which this node can't decode yet."
        )
    return gltf


# --- Conversion job ---------------------------------------------------------------------------


class _ConversionJob:
    def __init__(self, gltf: pygltflib.GLTF2, layer_path: Path, material_mode: MaterialMode, tex_dir: str) -> None:
        self.gltf = gltf
        self.layer_path = layer_path
        self.material_mode = material_mode
        self.tex_dir_name = tex_dir
        self.tex_dir = layer_path.parent / tex_dir
        self.result = ConversionResult(output_path=layer_path)
        self._buffers: dict[int, bytes] = {}
        self._image_files: dict[tuple[int, bool], str | None] = {}
        self._materials: dict[int, UsdShade.Material] = {}
        self._warned: set[str] = set()

    def run(self) -> ConversionResult:
        self.layer_path.parent.mkdir(parents=True, exist_ok=True)
        stage = Usd.Stage.CreateNew(str(self.layer_path))
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)  # glTF is Y-up, metres
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)

        root = UsdGeom.Xform.Define(stage, "/Root")
        stage.SetDefaultPrim(root.GetPrim())
        Usd.ModelAPI(root.GetPrim()).SetKind(Kind.Tokens.component)
        self.stage = stage
        self.geom_root = UsdGeom.Scope.Define(stage, "/Root/Geom").GetPath()
        self.mat_root = UsdGeom.Scope.Define(stage, "/Root/Materials").GetPath()

        self._warn_unsupported_features()
        for node_index in self._scene_root_nodes():
            self._convert_node(node_index, self.geom_root, set())

        stage.GetRootLayer().Save()
        self.result.material_count = len(self._materials)
        return self.result

    # -- warnings -----------------------------------------------------------------------------

    def _warn(self, message: str) -> None:
        if message not in self._warned:
            self._warned.add(message)
            self.result.warnings.append(message)

    def _warn_unsupported_features(self) -> None:
        g = self.gltf
        if g.skins:
            self._warn("Skinning was ignored; meshes are exported in their bind pose.")
        if g.animations:
            self._warn("Animations were ignored; only the static pose is exported.")
        if any(p.targets for m in (g.meshes or []) for p in m.primitives):
            self._warn("Morph targets (blend shapes) were ignored.")
        if g.cameras:
            self._warn("Cameras were ignored.")
        if "KHR_lights_punctual" in (g.extensionsUsed or []):
            self._warn("Lights were ignored.")

    # -- hierarchy ----------------------------------------------------------------------------

    def _scene_root_nodes(self) -> list[int]:
        g = self.gltf
        if g.scenes:
            scene_index = g.scene if g.scene is not None else 0
            return list(g.scenes[scene_index].nodes or [])
        children = {c for n in (g.nodes or []) for c in (n.children or [])}
        return [i for i in range(len(g.nodes or [])) if i not in children]

    def _convert_node(self, node_index: int, parent_path: Sdf.Path, visiting: set[int]) -> None:
        if node_index in visiting:
            self._warn("The glTF node graph contains a cycle; it was broken during conversion.")
            return
        node = self.gltf.nodes[node_index]
        name = self._unique_child_name(parent_path, node.name or f"node_{node_index}")
        xform = UsdGeom.Xform.Define(self.stage, parent_path.AppendChild(name))
        matrix = _node_matrix(node)
        if matrix != Gf.Matrix4d(1.0):
            xform.AddTransformOp().Set(matrix)

        if node.mesh is not None:
            self._convert_mesh(node.mesh, xform.GetPath())

        for child in node.children or []:
            self._convert_node(child, xform.GetPath(), visiting | {node_index})

    def _unique_child_name(self, parent_path: Sdf.Path, raw_name: str) -> str:
        base = Tf.MakeValidIdentifier(raw_name) or "prim"
        name, i = base, 1
        while self.stage.GetPrimAtPath(parent_path.AppendChild(name)):
            name = f"{base}_{i}"
            i += 1
        return name

    # -- geometry -----------------------------------------------------------------------------

    def _convert_mesh(self, mesh_index: int, parent_path: Sdf.Path) -> None:
        gltf_mesh = self.gltf.meshes[mesh_index]
        prims = gltf_mesh.primitives
        for prim_index, primitive in enumerate(prims):
            mode = _PrimitiveMode.TRIANGLES if primitive.mode is None else primitive.mode
            if mode not in tuple(_PrimitiveMode):
                self._warn("Point and line primitives were skipped; only triangle geometry is converted.")
                continue
            if primitive.attributes.POSITION is None:
                continue
            raw_name = gltf_mesh.name or f"mesh_{mesh_index}"
            if len(prims) > 1:
                raw_name = f"{raw_name}_{prim_index}"
            mesh_path = parent_path.AppendChild(self._unique_child_name(parent_path, raw_name))
            self._author_primitive(primitive, mode, mesh_path)

    def _author_primitive(self, primitive: pygltflib.Primitive, mode: int, mesh_path: Sdf.Path) -> None:
        attrs = primitive.attributes
        if attrs.POSITION is None:
            return
        points = self._read_accessor(attrs.POSITION).astype(np.float32)
        if primitive.indices is not None:
            indices = self._read_accessor(primitive.indices).reshape(-1).astype(np.int64)
        else:
            indices = np.arange(len(points), dtype=np.int64)
        indices = _triangulate(indices, mode)
        if indices.size == 0:
            return

        mesh = UsdGeom.Mesh.Define(self.stage, mesh_path)
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(indices.size // 3, 3, dtype=np.int32)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(indices.astype(np.int32)))
        mesh.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(np.stack([points.min(axis=0), points.max(axis=0)])))

        if attrs.NORMAL is not None:
            normals = self._read_accessor(attrs.NORMAL).astype(np.float32)
            mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normals))
            mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)

        pv_api = UsdGeom.PrimvarsAPI(mesh)
        for set_index in range(4):
            accessor = getattr(attrs, f"TEXCOORD_{set_index}", None)
            if accessor is None:
                continue
            uv = self._read_accessor(accessor).astype(np.float32).copy()
            uv[:, 1] = 1.0 - uv[:, 1]  # glTF origin is top-left, USD st origin is bottom-left
            pv = pv_api.CreatePrimvar(_st_name(set_index), Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
            pv.Set(Vt.Vec2fArray.FromNumpy(uv))

        if attrs.COLOR_0 is not None:
            colors = self._read_accessor(attrs.COLOR_0).astype(np.float32)
            mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex).Set(Vt.Vec3fArray.FromNumpy(colors[:, :3].copy()))
            if colors.shape[1] == 4:
                mesh.CreateDisplayOpacityPrimvar(UsdGeom.Tokens.vertex).Set(
                    Vt.FloatArray.FromNumpy(colors[:, 3].copy())
                )

        if primitive.material is not None:
            gltf_mat = self.gltf.materials[primitive.material]
            if gltf_mat.doubleSided:
                mesh.CreateDoubleSidedAttr(True)
            material = self._get_material(primitive.material)
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)

        self.result.mesh_count += 1
        self.result.triangle_count += indices.size // 3

    # -- accessors ----------------------------------------------------------------------------

    def _read_accessor(self, accessor_index: int) -> np.ndarray:
        acc = self.gltf.accessors[accessor_index]
        dtype = np.dtype(_COMPONENT_DTYPES[acc.componentType]).newbyteorder("<")
        width = _TYPE_SIZES[acc.type]

        if acc.bufferView is None:
            data = np.zeros((acc.count, width), dtype=dtype)
        else:
            data = self._read_buffer_view_array(acc.bufferView, acc.byteOffset or 0, acc.count, width, dtype)

        if acc.sparse is not None:
            data = data.copy()
            sp = acc.sparse
            idx_dtype = np.dtype(_COMPONENT_DTYPES[sp.indices.componentType]).newbyteorder("<")
            idx = self._read_buffer_view_array(
                sp.indices.bufferView, sp.indices.byteOffset or 0, sp.count, 1, idx_dtype
            )
            vals = self._read_buffer_view_array(sp.values.bufferView, sp.values.byteOffset or 0, sp.count, width, dtype)
            data[idx.reshape(-1)] = vals

        if acc.normalized and dtype.type in _NORMALIZE_DIVISORS:
            return np.maximum(data.astype(np.float32) / _NORMALIZE_DIVISORS[dtype.type], -1.0)
        return data

    def _read_buffer_view_array(
        self, view_index: int, offset: int, count: int, width: int, dtype: np.dtype
    ) -> np.ndarray:
        view = self.gltf.bufferViews[view_index]
        buf = self._buffer_bytes(view.buffer)
        start = (view.byteOffset or 0) + offset
        packed_stride = dtype.itemsize * width
        stride = view.byteStride or packed_stride
        if stride == packed_stride:
            return np.frombuffer(buf, dtype=dtype, count=count * width, offset=start).reshape(count, width)
        return np.ndarray(
            shape=(count, width), dtype=dtype, buffer=buf, offset=start, strides=(stride, dtype.itemsize)
        ).copy()

    def _buffer_bytes(self, buffer_index: int) -> bytes:
        if buffer_index in self._buffers:
            return self._buffers[buffer_index]
        uri = self.gltf.buffers[buffer_index].uri
        if uri is None:
            data = self.gltf.binary_blob()
        elif uri.startswith("data:"):
            data = _decode_data_uri(uri)
        else:
            raise _external_resource_error("buffer", uri)
        if data is None:
            raise GltfConversionError("Attempted to read mesh data. The GLB has no binary chunk.")
        declared = self.gltf.buffers[buffer_index].byteLength or 0
        if len(data) < declared:
            msg = (
                f"Attempted to read buffer {buffer_index}. It holds {len(data)} bytes but declares {declared};"
                " the file looks truncated."
            )
            raise GltfConversionError(msg)
        self._buffers[buffer_index] = data
        return data

    # -- textures -----------------------------------------------------------------------------

    def _texture_file(self, texture_index: int, *, drop_alpha: bool = False) -> str | None:
        """Write the texture's image next to the layer and return its relative asset path.

        drop_alpha: write an RGB copy. glTF OPAQUE materials must ignore base-colour alpha, but
        USD renderers may premultiply a texture's alpha on load and darken it, so strip it.
        """
        texture = self.gltf.textures[texture_index]
        source = texture.source
        for ext_name in ("EXT_texture_webp", "KHR_texture_basisu"):
            ext = (texture.extensions or {}).get(ext_name)
            if ext and "source" in ext:
                source = ext["source"]
        if source is None:
            return None
        key = (source, drop_alpha)
        if key in self._image_files:
            return self._image_files[key]

        image = self.gltf.images[source]
        data = self._image_bytes(image)
        # Trust the bytes, not mimeType: it may be missing or wrong, and a mislabelled file gets
        # renamed on save (extension coercion), leaving the USD pointing at a path that isn't there.
        if data.startswith(_KTX2_MAGIC):
            self._warn("KTX2 textures can't be read by USD and were dropped.")
            self._image_files[key] = None
            return None
        try:
            pil_format = PILImage.open(io.BytesIO(data)).format
        except (UnidentifiedImageError, OSError):
            self._warn("Textures in an unrecognised image format were dropped.")
            self._image_files[key] = None
            return None
        ext = _KEPT_IMAGE_FORMATS.get(pil_format or "")
        if ext is None:  # WebP and anything else Pillow reads: re-encode to PNG
            out = io.BytesIO()
            PILImage.open(io.BytesIO(data)).save(out, format="PNG")
            data, ext = out.getvalue(), "png"

        suffix = ""
        if drop_alpha and ext == "png":
            pil = PILImage.open(io.BytesIO(data))
            if "A" in pil.getbands() or "transparency" in pil.info:
                out = io.BytesIO()
                pil.convert("RGB").save(out, format="PNG")
                data, suffix = out.getvalue(), "_rgb"

        stem = Tf.MakeValidIdentifier(Path(image.name).stem) if image.name else ""
        filename = f"{stem or 'texture'}_{source}{suffix}.{ext}"
        self.tex_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.tex_dir / filename
        out_path.write_bytes(data)
        self.result.texture_paths.append(out_path)
        rel = f"./{self.tex_dir_name}/{filename}"
        self._image_files[key] = rel
        return rel

    def _image_bytes(self, image: pygltflib.Image) -> bytes:
        if image.bufferView is not None:
            view = self.gltf.bufferViews[image.bufferView]
            buf = self._buffer_bytes(view.buffer)
            start = view.byteOffset or 0
            return buf[start : start + view.byteLength]
        uri = image.uri or ""
        if uri.startswith("data:"):
            return _decode_data_uri(uri)
        raise _external_resource_error("image", uri)

    def _texture_ref(self, info: _TextureInfo | None, *, drop_alpha: bool = False) -> _TexRef | None:
        if info is None or info.index is None:
            return None
        asset = self._texture_file(info.index, drop_alpha=drop_alpha)
        if asset is None:
            return None
        if (info.extensions or {}).get("KHR_texture_transform"):
            self._warn("KHR_texture_transform (UV offset/scale/rotation) isn't applied yet.")
        sampler_index = self.gltf.textures[info.index].sampler
        sampler = self.gltf.samplers[sampler_index] if sampler_index is not None else None
        return _TexRef(
            asset=asset,
            st=_st_name(info.texCoord or 0),
            wrap_s=(sampler.wrapS if sampler and sampler.wrapS else 10497),
            wrap_t=(sampler.wrapT if sampler and sampler.wrapT else 10497),
        )

    # -- materials ----------------------------------------------------------------------------

    def _get_material(self, material_index: int) -> UsdShade.Material:
        if material_index in self._materials:
            return self._materials[material_index]
        gm = self.gltf.materials[material_index]
        name = self._unique_child_name(self.mat_root, gm.name or f"material_{material_index}")
        material = UsdShade.Material.Define(self.stage, self.mat_root.AppendChild(name))
        params = self._material_params(gm)
        if self.material_mode in (MaterialMode.PREVIEW_SURFACE, MaterialMode.BOTH):
            _PreviewSurfaceBuilder(self.stage, material, params).build()
        if self.material_mode in (MaterialMode.MATERIALX, MaterialMode.BOTH):
            if gm.occlusionTexture is not None:
                self._warn(
                    "MaterialX standard_surface has no ambient occlusion input; occlusion maps were skipped for MaterialX."
                )
            _MaterialXBuilder(self.stage, material, params).build()
        self._materials[material_index] = material
        return material

    def _material_params(self, gm: pygltflib.Material) -> _MaterialParams:
        pbr = gm.pbrMetallicRoughness or pygltflib.PbrMetallicRoughness()
        exts = gm.extensions or {}
        emissive_strength = float(exts.get("KHR_materials_emissive_strength", {}).get("emissiveStrength", 1.0))
        if "KHR_materials_unlit" in exts:
            self._warn(
                "KHR_materials_unlit isn't supported; unlit materials were converted as lit PBR"
                " using their glTF fallback factors."
            )
        unsupported = sorted(
            set(exts) - {"KHR_materials_emissive_strength", "KHR_materials_ior", "KHR_materials_unlit"}
        )
        if unsupported:
            self._warn(f"Material extensions not converted: {', '.join(unsupported)}.")
        normal_info = gm.normalTexture
        occlusion_info = gm.occlusionTexture
        return _MaterialParams(
            base_color=_vec4(pbr.baseColorFactor, (1.0, 1.0, 1.0, 1.0)),
            metallic=1.0 if pbr.metallicFactor is None else float(pbr.metallicFactor),
            roughness=1.0 if pbr.roughnessFactor is None else float(pbr.roughnessFactor),
            emissive=_scaled_vec3(gm.emissiveFactor, emissive_strength),
            alpha_mode=gm.alphaMode or "OPAQUE",
            alpha_cutoff=0.5 if gm.alphaCutoff is None else float(gm.alphaCutoff),
            ior=float(exts.get("KHR_materials_ior", {}).get("ior", 1.5)),
            normal_scale=float(normal_info.scale) if normal_info and normal_info.scale is not None else 1.0,
            occlusion_strength=float(occlusion_info.strength)
            if occlusion_info and occlusion_info.strength is not None
            else 1.0,
            base_color_tex=self._texture_ref(pbr.baseColorTexture, drop_alpha=(gm.alphaMode or "OPAQUE") == "OPAQUE"),
            metal_rough_tex=self._texture_ref(pbr.metallicRoughnessTexture),
            normal_tex=self._texture_ref(normal_info),
            # Only UsdPreviewSurface has an occlusion input; don't write an unused texture.
            occlusion_tex=None if self.material_mode is MaterialMode.MATERIALX else self._texture_ref(occlusion_info),
            emissive_tex=self._texture_ref(gm.emissiveTexture),
        )


# --- Material parameter model ------------------------------------------------------------------

type _TextureInfo = pygltflib.TextureInfo | pygltflib.NormalMaterialTexture | pygltflib.OcclusionTextureInfo


@dataclass
class _TexRef:
    asset: str
    st: str
    wrap_s: int
    wrap_t: int


@dataclass
class _MaterialParams:
    base_color: tuple[float, float, float, float]
    metallic: float
    roughness: float
    emissive: tuple[float, float, float]
    alpha_mode: str
    alpha_cutoff: float
    ior: float
    normal_scale: float
    occlusion_strength: float
    base_color_tex: _TexRef | None
    metal_rough_tex: _TexRef | None
    normal_tex: _TexRef | None
    occlusion_tex: _TexRef | None
    emissive_tex: _TexRef | None

    @property
    def has_emission(self) -> bool:
        return self.emissive_tex is not None or any(c > 0.0 for c in self.emissive)


# --- UsdPreviewSurface -------------------------------------------------------------------------


class _PreviewSurfaceBuilder:
    def __init__(self, stage: Usd.Stage, material: UsdShade.Material, p: _MaterialParams) -> None:
        self.stage = stage
        self.material = material
        self.p = p
        # Shaders live directly under the Material: a Scope there is invalid UsdShade hierarchy.
        self.base = material.GetPath()
        self._readers: dict[str, UsdShade.Shader] = {}

    def build(self) -> None:
        p = self.p
        surf = self._shader("PreviewSurface", "UsdPreviewSurface")
        surf.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        self.material.CreateSurfaceOutput().ConnectToSource(surf.ConnectableAPI(), "surface")
        surf.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(0)
        surf.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(p.ior)

        diffuse = surf.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
        if p.base_color_tex:
            tex = self._texture("BaseColorTex", p.base_color_tex, "sRGB", scale=p.base_color)
            diffuse.ConnectToSource(tex.ConnectableAPI(), "rgb")
        else:
            diffuse.Set(Gf.Vec3f(*p.base_color[:3]))

        if p.alpha_mode in ("BLEND", "MASK"):
            opacity = surf.CreateInput("opacity", Sdf.ValueTypeNames.Float)
            if p.base_color_tex:
                opacity.ConnectToSource(
                    self._texture("BaseColorTex", p.base_color_tex, "sRGB", scale=p.base_color).ConnectableAPI(), "a"
                )
            else:
                opacity.Set(p.base_color[3])
            if p.alpha_mode == "MASK":
                surf.CreateInput("opacityThreshold", Sdf.ValueTypeNames.Float).Set(p.alpha_cutoff)

        metallic = surf.CreateInput("metallic", Sdf.ValueTypeNames.Float)
        roughness = surf.CreateInput("roughness", Sdf.ValueTypeNames.Float)
        if p.metal_rough_tex:
            # glTF packs roughness in G and metallic in B.
            tex = self._texture("MetalRoughTex", p.metal_rough_tex, "raw", scale=(1.0, p.roughness, p.metallic, 1.0))
            roughness.ConnectToSource(tex.ConnectableAPI(), "g")
            metallic.ConnectToSource(tex.ConnectableAPI(), "b")
        else:
            metallic.Set(p.metallic)
            roughness.Set(p.roughness)

        if p.normal_tex:
            s = p.normal_scale
            tex = self._texture(
                "NormalTex", p.normal_tex, "raw", scale=(2.0 * s, 2.0 * s, 2.0, 1.0), bias=(-s, -s, -1.0, 0.0)
            )
            surf.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(tex.ConnectableAPI(), "rgb")

        if p.occlusion_tex:
            k = p.occlusion_strength
            tex = self._texture(
                "OcclusionTex", p.occlusion_tex, "raw", scale=(k, k, k, 1.0), bias=(1.0 - k, 1.0 - k, 1.0 - k, 0.0)
            )
            surf.CreateInput("occlusion", Sdf.ValueTypeNames.Float).ConnectToSource(tex.ConnectableAPI(), "r")

        if p.has_emission:
            emissive = surf.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f)
            if p.emissive_tex:
                tex = self._texture("EmissiveTex", p.emissive_tex, "sRGB", scale=(*p.emissive, 1.0))
                emissive.ConnectToSource(tex.ConnectableAPI(), "rgb")
            else:
                emissive.Set(Gf.Vec3f(*p.emissive))

    def _shader(self, name: str, shader_id: str) -> UsdShade.Shader:
        shader = UsdShade.Shader.Define(self.stage, self.base.AppendChild(name))
        shader.CreateIdAttr(shader_id)
        return shader

    def _reader(self, st: str) -> UsdShade.Shader:
        if st not in self._readers:
            reader = self._shader(f"Reader_{st}", "UsdPrimvarReader_float2")
            reader.CreateInput("varname", Sdf.ValueTypeNames.String).Set(st)
            reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
            self._readers[st] = reader
        return self._readers[st]

    def _texture(
        self, name: str, ref: _TexRef, color_space: str, *, scale: tuple, bias: tuple | None = None
    ) -> UsdShade.Shader:
        path = self.base.AppendChild(name)
        existing = UsdShade.Shader.Get(self.stage, path)
        if existing:
            return existing
        tex = self._shader(name, "UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(ref.asset))
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            self._reader(ref.st).ConnectableAPI(), "result"
        )
        tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set(_WRAP_PREVIEW.get(ref.wrap_s, "repeat"))
        tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set(_WRAP_PREVIEW.get(ref.wrap_t, "repeat"))
        tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(color_space)
        if tuple(scale) != (1.0, 1.0, 1.0, 1.0):
            tex.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*scale))
        if bias is not None:
            tex.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*bias))
        for out_name, out_type in (
            ("rgb", Sdf.ValueTypeNames.Float3),
            ("r", Sdf.ValueTypeNames.Float),
            ("g", Sdf.ValueTypeNames.Float),
            ("b", Sdf.ValueTypeNames.Float),
            ("a", Sdf.ValueTypeNames.Float),
        ):
            tex.CreateOutput(out_name, out_type)
        return tex


# --- MaterialX (standard_surface) ---------------------------------------------------------------


class _MaterialXBuilder:
    """Authors a MaterialX standard_surface network as UsdShade prims (mtlx render context)."""

    def __init__(self, stage: Usd.Stage, material: UsdShade.Material, p: _MaterialParams) -> None:
        self.stage = stage
        self.material = material
        self.p = p
        self.base = material.GetPath()
        self._texcoords: dict[str, UsdShade.Shader] = {}
        self._images: dict[str, UsdShade.Shader] = {}

    def build(self) -> None:
        p = self.p
        surf = self._node("Surface", "ND_standard_surface_surfaceshader", "out", Sdf.ValueTypeNames.Token)
        self.material.CreateSurfaceOutput("mtlx").ConnectToSource(surf.ConnectableAPI(), "out")
        surf.CreateInput("base", Sdf.ValueTypeNames.Float).Set(1.0)
        surf.CreateInput("specular", Sdf.ValueTypeNames.Float).Set(1.0)
        surf.CreateInput("specular_IOR", Sdf.ValueTypeNames.Float).Set(p.ior)

        base_color = surf.CreateInput("base_color", Sdf.ValueTypeNames.Color3f)
        if p.base_color_tex:
            img = self._image(
                "BaseColorImage", "ND_image_color4", Sdf.ValueTypeNames.Color4f, p.base_color_tex, srgb=True
            )
            rgb = self._node("BaseColorRGB", "ND_convert_color4_color3", "out", Sdf.ValueTypeNames.Color3f)
            self._connect(rgb, "in", Sdf.ValueTypeNames.Color4f, img)
            if p.base_color[:3] != (1.0, 1.0, 1.0):
                mul = self._node("BaseColorFactor", "ND_multiply_color3", "out", Sdf.ValueTypeNames.Color3f)
                self._connect(mul, "in1", Sdf.ValueTypeNames.Color3f, rgb)
                mul.CreateInput("in2", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*p.base_color[:3]))
                rgb = mul
            base_color.ConnectToSource(rgb.ConnectableAPI(), "out")
        else:
            base_color.Set(Gf.Vec3f(*p.base_color[:3]))

        if p.alpha_mode in ("BLEND", "MASK"):
            alpha_src = self._alpha_source()
            if p.alpha_mode == "MASK":
                cut = self._node("AlphaCutoff", "ND_ifgreatereq_float", "out", Sdf.ValueTypeNames.Float)
                self._value_or_connect(cut, "value1", alpha_src)
                cut.CreateInput("value2", Sdf.ValueTypeNames.Float).Set(p.alpha_cutoff)
                cut.CreateInput("in1", Sdf.ValueTypeNames.Float).Set(1.0)
                cut.CreateInput("in2", Sdf.ValueTypeNames.Float).Set(0.0)
                alpha_src = cut
            opacity = self._node("Opacity", "ND_convert_float_color3", "out", Sdf.ValueTypeNames.Color3f)
            self._value_or_connect(opacity, "in", alpha_src)
            surf.CreateInput("opacity", Sdf.ValueTypeNames.Color3f).ConnectToSource(opacity.ConnectableAPI(), "out")

        metalness = surf.CreateInput("metalness", Sdf.ValueTypeNames.Float)
        roughness = surf.CreateInput("specular_roughness", Sdf.ValueTypeNames.Float)
        if p.metal_rough_tex:
            img = self._image(
                "MetalRoughImage", "ND_image_vector3", Sdf.ValueTypeNames.Float3, p.metal_rough_tex, srgb=False
            )
            metalness.ConnectToSource(self._channel("Metallic", img, 2, p.metallic).ConnectableAPI(), "out")
            roughness.ConnectToSource(self._channel("Roughness", img, 1, p.roughness).ConnectableAPI(), "out")
        else:
            metalness.Set(p.metallic)
            roughness.Set(p.roughness)

        if p.normal_tex:
            img = self._image("NormalImage", "ND_image_vector3", Sdf.ValueTypeNames.Float3, p.normal_tex, srgb=False)
            nmap = self._node("NormalMap", "ND_normalmap_float", "out", Sdf.ValueTypeNames.Float3)
            self._connect(nmap, "in", Sdf.ValueTypeNames.Float3, img)
            nmap.CreateInput("scale", Sdf.ValueTypeNames.Float).Set(p.normal_scale)
            surf.CreateInput("normal", Sdf.ValueTypeNames.Float3).ConnectToSource(nmap.ConnectableAPI(), "out")

        if p.has_emission:
            surf.CreateInput("emission", Sdf.ValueTypeNames.Float).Set(1.0)
            emission = surf.CreateInput("emission_color", Sdf.ValueTypeNames.Color3f)
            if p.emissive_tex:
                src = self._image(
                    "EmissiveImage", "ND_image_color3", Sdf.ValueTypeNames.Color3f, p.emissive_tex, srgb=True
                )
                if tuple(p.emissive) != (1.0, 1.0, 1.0):
                    mul = self._node("EmissiveFactor", "ND_multiply_color3", "out", Sdf.ValueTypeNames.Color3f)
                    self._connect(mul, "in1", Sdf.ValueTypeNames.Color3f, src)
                    mul.CreateInput("in2", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*p.emissive))
                    src = mul
                emission.ConnectToSource(src.ConnectableAPI(), "out")
            else:
                emission.Set(Gf.Vec3f(*p.emissive))

    def _alpha_source(self) -> UsdShade.Shader | float:
        p = self.p
        if not p.base_color_tex:
            return p.base_color[3]
        img = self._image("BaseColorImage", "ND_image_color4", Sdf.ValueTypeNames.Color4f, p.base_color_tex, srgb=True)
        alpha = self._node("BaseAlpha", "ND_extract_color4", "out", Sdf.ValueTypeNames.Float)
        self._connect(alpha, "in", Sdf.ValueTypeNames.Color4f, img)
        alpha.CreateInput("index", Sdf.ValueTypeNames.Int).Set(3)
        if p.base_color[3] == 1.0:
            return alpha
        mul = self._node("BaseAlphaFactor", "ND_multiply_float", "out", Sdf.ValueTypeNames.Float)
        self._connect(mul, "in1", Sdf.ValueTypeNames.Float, alpha)
        mul.CreateInput("in2", Sdf.ValueTypeNames.Float).Set(p.base_color[3])
        return mul

    def _channel(self, name: str, img: UsdShade.Shader, index: int, factor: float) -> UsdShade.Shader:
        ext = self._node(f"{name}Channel", "ND_extract_vector3", "out", Sdf.ValueTypeNames.Float)
        self._connect(ext, "in", Sdf.ValueTypeNames.Float3, img)
        ext.CreateInput("index", Sdf.ValueTypeNames.Int).Set(index)
        if factor == 1.0:
            return ext
        mul = self._node(f"{name}Factor", "ND_multiply_float", "out", Sdf.ValueTypeNames.Float)
        self._connect(mul, "in1", Sdf.ValueTypeNames.Float, ext)
        mul.CreateInput("in2", Sdf.ValueTypeNames.Float).Set(factor)
        return mul

    def _image(
        self, name: str, node_id: str, out_type: Sdf.ValueTypeName, ref: _TexRef, *, srgb: bool
    ) -> UsdShade.Shader:
        if name in self._images:
            return self._images[name]
        img = self._node(name, node_id, "out", out_type)
        file_input = img.CreateInput("file", Sdf.ValueTypeNames.Asset)
        file_input.Set(Sdf.AssetPath(ref.asset))
        if srgb:
            file_input.GetAttr().SetColorSpace("srgb_texture")
        img.CreateInput("uaddressmode", Sdf.ValueTypeNames.String).Set(_WRAP_MTLX.get(ref.wrap_s, "periodic"))
        img.CreateInput("vaddressmode", Sdf.ValueTypeNames.String).Set(_WRAP_MTLX.get(ref.wrap_t, "periodic"))
        img.CreateInput("texcoord", Sdf.ValueTypeNames.Float2).ConnectToSource(
            self._texcoord(ref.st).ConnectableAPI(), "out"
        )
        self._images[name] = img
        return img

    def _texcoord(self, st: str) -> UsdShade.Shader:
        if st not in self._texcoords:
            node = self._node(f"TexCoord_{st}", "ND_geompropvalue_vector2", "out", Sdf.ValueTypeNames.Float2)
            node.CreateInput("geomprop", Sdf.ValueTypeNames.String).Set(st)
            self._texcoords[st] = node
        return self._texcoords[st]

    def _node(self, name: str, node_id: str, out_name: str, out_type: Sdf.ValueTypeName) -> UsdShade.Shader:
        shader = UsdShade.Shader.Define(self.stage, self.base.AppendChild(f"mtlx_{name}"))
        shader.CreateIdAttr(node_id)
        shader.CreateOutput(out_name, out_type)
        return shader

    @staticmethod
    def _connect(dst: UsdShade.Shader, input_name: str, input_type: Sdf.ValueTypeName, src: UsdShade.Shader) -> None:
        dst.CreateInput(input_name, input_type).ConnectToSource(src.ConnectableAPI(), "out")

    @staticmethod
    def _value_or_connect(dst: UsdShade.Shader, input_name: str, src: UsdShade.Shader | float) -> None:
        inp = dst.CreateInput(input_name, Sdf.ValueTypeNames.Float)
        if isinstance(src, UsdShade.Shader):
            inp.ConnectToSource(src.ConnectableAPI(), "out")
        else:
            inp.Set(float(src))


# --- Helpers -----------------------------------------------------------------------------------


def _decode_data_uri(uri: str) -> bytes:
    header, sep, payload = uri.partition(",")
    if not sep or ";base64" not in header:
        msg = "Attempted to decode a data: URI. Only base64 data: URIs are supported."
        raise GltfConversionError(msg)
    return base64.b64decode(payload, validate=True)


def _external_resource_error(kind: str, uri: str) -> GltfConversionError:
    shown = uri if len(uri) <= 80 else f"{uri[:77]}..."
    return GltfConversionError(
        f"Attempted to read an external {kind} ('{shown}'). Only self-contained models are supported:"
        " use a .glb, or a .gltf with its buffers and images embedded as data: URIs."
    )


def _vec4(values: list[float] | None, default: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    if not values or len(values) != 4:
        return default
    return (float(values[0]), float(values[1]), float(values[2]), float(values[3]))


def _scaled_vec3(values: list[float] | None, scale: float) -> tuple[float, float, float]:
    if not values or len(values) != 3:
        return (0.0, 0.0, 0.0)
    return (float(values[0]) * scale, float(values[1]) * scale, float(values[2]) * scale)


def _st_name(set_index: int) -> str:
    return "st" if set_index == 0 else f"st{set_index}"


def _node_matrix(node: pygltflib.Node) -> Gf.Matrix4d:
    if node.matrix and len(node.matrix) == 16:
        # glTF stores column-major for column vectors; read row-wise it is exactly USD's
        # row-vector matrix, so no transpose is needed.
        return Gf.Matrix4d(*[float(v) for v in node.matrix])
    m = Gf.Matrix4d(1.0)
    if node.scale:
        m = m * Gf.Matrix4d().SetScale(Gf.Vec3d(*node.scale))
    if node.rotation:
        x, y, z, w = node.rotation
        m = m * Gf.Matrix4d().SetRotate(Gf.Quatd(w, x, y, z))
    if node.translation:
        m = m * Gf.Matrix4d().SetTranslate(Gf.Vec3d(*node.translation))
    return m


def _triangulate(indices: np.ndarray, mode: int) -> np.ndarray:
    n = indices.size - 2
    match mode:
        case _PrimitiveMode.TRIANGLES:
            return indices[: indices.size - indices.size % 3]
        case _PrimitiveMode.TRIANGLE_FAN:
            if n <= 0:
                return np.empty(0, dtype=np.int64)
            return np.stack([np.full(n, indices[0]), indices[1:-1], indices[2:]], axis=1).reshape(-1)
        case _PrimitiveMode.TRIANGLE_STRIP:
            if n <= 0:
                return np.empty(0, dtype=np.int64)
            # Flip winding on odd triangles.
            tris = np.stack([indices[:-2], indices[1:-1], indices[2:]], axis=1)
            odd = np.arange(n) % 2 == 1
            tris[odd] = tris[odd][:, [1, 0, 2]]
            return tris.reshape(-1)
        case _:
            msg = f"Unsupported glTF primitive mode: {mode!r}"
            raise ValueError(msg)
