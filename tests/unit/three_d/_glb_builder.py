"""Builds small synthetic GLBs for the GLB to USD tests, so no sample assets are needed."""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pygltflib
from PIL import Image

if TYPE_CHECKING:
    from pathlib import Path

# Quad in the XY plane, two triangles, facing +Z.
QUAD_POSITIONS = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)
QUAD_NORMALS = np.array([[0, 0, 1]] * 4, dtype=np.float32)
QUAD_UVS = np.array([[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
QUAD_INDICES = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint16)

PARENT_TRANSLATION = (1.0, 2.0, 3.0)
CHILD_SCALE = (2.0, 2.0, 2.0)


@dataclass
class GlbSpec:
    alpha_mode: str = "OPAQUE"
    textured: bool = True
    image_format: str = "PNG"  # PNG or WEBP
    mode: int | None = None  # None -> triangles
    indices: np.ndarray = field(default_factory=lambda: QUAD_INDICES)
    extensions_required: list[str] = field(default_factory=list)
    with_camera: bool = False


def _png(color: tuple[int, ...], image_format: str = "PNG") -> bytes:
    mode = "RGBA" if len(color) == 4 else "RGB"
    out = io.BytesIO()
    Image.new(mode, (2, 2), color).save(out, format=image_format)
    return out.getvalue()


class _BlobBuilder:
    def __init__(self) -> None:
        self.blob = bytearray()
        self.views: list[pygltflib.BufferView] = []

    def add(self, data: bytes, target: int | None = None) -> int:
        while len(self.blob) % 4:
            self.blob.append(0)
        self.views.append(
            pygltflib.BufferView(buffer=0, byteOffset=len(self.blob), byteLength=len(data), target=target)
        )
        self.blob.extend(data)
        return len(self.views) - 1


def build_glb(path: Path, spec: GlbSpec | None = None) -> Path:
    spec = spec or GlbSpec()
    blob = _BlobBuilder()
    accessors: list[pygltflib.Accessor] = []

    def accessor(array: np.ndarray, component: int, type_: str, *, target: int | None, minmax: bool = False) -> int:
        view = blob.add(array.tobytes(), target)
        acc = pygltflib.Accessor(bufferView=view, componentType=component, count=len(array), type=type_)
        if minmax:
            acc.min = array.min(axis=0).tolist()
            acc.max = array.max(axis=0).tolist()
        accessors.append(acc)
        return len(accessors) - 1

    pos = accessor(QUAD_POSITIONS, pygltflib.FLOAT, "VEC3", target=pygltflib.ARRAY_BUFFER, minmax=True)
    nrm = accessor(QUAD_NORMALS, pygltflib.FLOAT, "VEC3", target=pygltflib.ARRAY_BUFFER)
    uv = accessor(QUAD_UVS, pygltflib.FLOAT, "VEC2", target=pygltflib.ARRAY_BUFFER)
    idx = accessor(spec.indices, pygltflib.UNSIGNED_SHORT, "SCALAR", target=pygltflib.ELEMENT_ARRAY_BUFFER)

    images: list[pygltflib.Image] = []
    textures: list[pygltflib.Texture] = []
    material = pygltflib.Material(name="QuadMaterial", alphaMode=spec.alpha_mode)
    pbr = pygltflib.PbrMetallicRoughness(baseColorFactor=[1.0, 1.0, 1.0, 1.0], metallicFactor=0.5, roughnessFactor=0.25)
    if spec.alpha_mode == "MASK":
        material.alphaCutoff = 0.4

    if spec.textured:
        mime = "image/webp" if spec.image_format == "WEBP" else "image/png"
        for name, color in (
            ("base", (200, 100, 50, 128)),
            ("metalrough", (0, 128, 255)),
            ("normal", (128, 128, 255)),
            ("emissive", (255, 255, 255)),
        ):
            view = blob.add(_png(color, spec.image_format))
            images.append(pygltflib.Image(name=name, bufferView=view, mimeType=mime))
            textures.append(pygltflib.Texture(source=len(images) - 1))
        pbr.baseColorTexture = pygltflib.TextureInfo(index=0)
        pbr.metallicRoughnessTexture = pygltflib.TextureInfo(index=1)
        material.normalTexture = pygltflib.NormalMaterialTexture(index=2, scale=0.5)
        material.emissiveTexture = pygltflib.TextureInfo(index=3)
        material.emissiveFactor = [1.0, 0.5, 0.25]
    material.pbrMetallicRoughness = pbr

    primitive = pygltflib.Primitive(
        attributes=pygltflib.Attributes(POSITION=pos, NORMAL=nrm, TEXCOORD_0=uv),
        indices=idx,
        material=0,
        mode=spec.mode,
    )
    nodes = [
        pygltflib.Node(name="Parent", translation=list(PARENT_TRANSLATION), children=[1]),
        pygltflib.Node(name="Child", mesh=0, scale=list(CHILD_SCALE)),
    ]
    extra: dict[str, Any] = {}
    if spec.with_camera:
        extra["cameras"] = [
            pygltflib.Camera(type="perspective", perspective=pygltflib.Perspective(yfov=0.8, znear=0.1))
        ]

    gltf = pygltflib.GLTF2(
        asset=pygltflib.Asset(version="2.0"),
        scene=0,
        scenes=[pygltflib.Scene(nodes=[0])],
        nodes=nodes,
        meshes=[pygltflib.Mesh(name="Quad", primitives=[primitive])],
        materials=[material],
        accessors=accessors,
        bufferViews=blob.views,
        buffers=[pygltflib.Buffer(byteLength=len(blob.blob))],
        images=images,
        textures=textures,
        extensionsRequired=spec.extensions_required,
        extensionsUsed=list(spec.extensions_required),
        **extra,
    )
    gltf.set_binary_blob(bytes(blob.blob))
    gltf.save_binary(str(path))
    return path
