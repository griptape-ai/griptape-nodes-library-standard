"""ConvertSpzToPly — download an SPZ splat and re-save it as a PLY file."""

from __future__ import annotations

import asyncio
import gzip
import io
import logging
import struct
from typing import Any

import numpy as np

from griptape_nodes.exe_types.core_types import ParameterMode
from griptape_nodes.exe_types.node_types import DataNode
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.files.file import File, FileDestination, FileLoadError

from griptape_nodes_library.splat.parameter_splat import ParameterSplat
from griptape_nodes_library.splat.splat_artifact import SplatUrlArtifact

logger = logging.getLogger("griptape_nodes")

__all__ = ["ConvertSpzToPly"]

_NGSP_MAGIC = 0x5053474E  # "NGSP" little-endian — used for both legacy and v4
_NGSP_HEADER_SIZE = 32
_LEGACY_HEADER_SIZE = 16
_TOC_ENTRY_SIZE = 16
_GZIP_MAGIC = b"\x1f\x8b"
_MIN_SMALLEST_THREE_VERSION = 3  # v3+ rotations use smallest-three (4 bytes); v1/v2 use 3 bytes
_SH_DIM_FOR_DEGREE = {0: 0, 1: 3, 2: 8, 3: 15, 4: 24}


def _decode_positions_24bit(buf: bytes, num_points: int, frac_bits: int) -> np.ndarray:
    """Decode N * 9-byte 24-bit signed fixed-point positions."""
    raw = np.frombuffer(buf, dtype=np.uint8).reshape(num_points, 9).astype(np.int32)
    positions = np.zeros((num_points, 3), dtype=np.float32)
    for axis in range(3):
        b = raw[:, axis * 3 : axis * 3 + 3]
        fixed = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        # Sign-extend 24-bit → 32-bit: if bit 23 is set, subtract 2^24.
        fixed = np.where(fixed & 0x800000, fixed - 0x1000000, fixed)
        positions[:, axis] = fixed.astype(np.float32) / (1 << frac_bits)
    return positions


def _decode_positions_float16(buf: bytes, num_points: int) -> np.ndarray:
    """v1 positions: N * 6 bytes, three float16 per point."""
    return np.frombuffer(buf, dtype=np.float16).reshape(num_points, 3).astype(np.float32)


def _decode_rotations_legacy_3byte(buf: bytes, num_points: int) -> np.ndarray:
    """v1/v2 rotations: 3 bytes per point, (x,y,z) components, reconstruct w."""
    raw = np.frombuffer(buf, dtype=np.uint8).reshape(num_points, 3).astype(np.float32)
    xyz = raw / 127.5 - 1.0  # (N, 3)
    w = np.sqrt(np.clip(1.0 - np.sum(xyz * xyz, axis=1), 0.0, 1.0)).astype(np.float32)
    # PLY 3DGS convention: rot_0=w, rot_1=x, rot_2=y, rot_3=z
    return np.stack([w, xyz[:, 0], xyz[:, 1], xyz[:, 2]], axis=1).astype(np.float32)


def _decode_rotations_smallest_three(buf: bytes, num_points: int) -> np.ndarray:
    """v3+ rotations: 4 bytes per point, smallest-three encoding.

    Packing (LSB first): [field0:10][field1:10][field2:10][i_largest:2]
    The loop fills rotation[i] for i = 3, 2, 1, 0 skipping i_largest — so the
    LSB-most 10-bit field gets assigned to the highest non-largest index.
    Quaternion component order is (x, y, z, w).
    """
    raw = np.frombuffer(buf, dtype=np.uint8).reshape(num_points, 4).astype(np.uint32)
    packed = raw[:, 0] | (raw[:, 1] << 8) | (raw[:, 2] << 16) | (raw[:, 3] << 24)
    i_largest = ((packed >> 30) & 0x3).astype(np.int64)

    def _extract(shift: int) -> np.ndarray:
        field = (packed >> shift) & 0x3FF
        mag = (field & 0x1FF).astype(np.float32) / 511.0 * float(np.sqrt(0.5))
        sign = np.where(field & 0x200, -1.0, 1.0).astype(np.float32)
        return (mag * sign).astype(np.float32)

    f0 = _extract(0)
    f1 = _extract(10)
    f2 = _extract(20)

    # Output in quaternion order (x, y, z, w)
    quat_xyzw = np.zeros((num_points, 4), dtype=np.float32)
    # Fill slots in descending order (3, 2, 1, 0), skipping i_largest, with fields [f0, f1, f2]
    fields = [f0, f1, f2]
    for row in range(num_points):
        li = int(i_largest[row])
        field_idx = 0
        for slot in range(3, -1, -1):
            if slot == li:
                continue
            quat_xyzw[row, slot] = fields[field_idx][row]
            field_idx += 1
        sum_sq = quat_xyzw[row, 0] ** 2 + quat_xyzw[row, 1] ** 2 + quat_xyzw[row, 2] ** 2 + quat_xyzw[row, 3] ** 2
        quat_xyzw[row, li] = float(np.sqrt(max(0.0, 1.0 - sum_sq)))

    # Convert (x, y, z, w) → (w, x, y, z) for PLY 3DGS convention
    return np.stack([quat_xyzw[:, 3], quat_xyzw[:, 0], quat_xyzw[:, 1], quat_xyzw[:, 2]], axis=1)


def _decode_alphas(buf: bytes) -> np.ndarray:
    """Dequantize alphas: byte → pre-sigmoid logit for PLY."""
    raw = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    p = np.clip(raw / 255.0, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p)).astype(np.float32)


def _decode_colors(buf: bytes, num_points: int) -> np.ndarray:
    """Dequantize DC color component: packed = clamp((color * 0.15 + 0.5) * 255)."""
    raw = np.frombuffer(buf, dtype=np.uint8).reshape(num_points, 3).astype(np.float32)
    return ((raw / 255.0) - 0.5) / 0.15


def _decode_scales(buf: bytes, num_points: int) -> np.ndarray:
    """Dequantize log-scales: byte / 16 - 10."""
    raw = np.frombuffer(buf, dtype=np.uint8).reshape(num_points, 3).astype(np.float32)
    return (raw / 16.0) - 10.0


def _decode_sh(buf: bytes, num_points: int, sh_dim: int) -> np.ndarray | None:
    """Dequantize spherical harmonics: (byte - 128) / 128."""
    if sh_dim == 0 or len(buf) == 0:
        return None
    raw = np.frombuffer(buf, dtype=np.uint8).reshape(num_points, sh_dim, 3).astype(np.float32)
    return ((raw - 128.0) / 128.0).astype(np.float32)


def _parse_legacy_spz(raw_bytes: bytes) -> dict[str, Any]:
    """Parse legacy (v1/v2/v3) SPZ: gzip-wrapped, 16-byte header, uncompressed streams."""
    if raw_bytes[:2] != _GZIP_MAGIC:
        raise ValueError("Legacy SPZ must start with gzip magic bytes")
    blob = gzip.decompress(raw_bytes)

    if len(blob) < _LEGACY_HEADER_SIZE:
        raise ValueError("Gunzipped SPZ too small for legacy header")

    magic, version, num_points, sh_degree, frac_bits, flags, _reserved = struct.unpack_from(
        "<IIIBBBB", blob, 0
    )
    if magic != _NGSP_MAGIC:
        raise ValueError(f"Not an SPZ file (magic={magic:#010x})")
    if version < 1 or version > 3:
        raise ValueError(f"Legacy SPZ version must be 1-3, got {version}")

    # Stream order: positions, alphas, colors, scales, rotations, sh
    offset = _LEGACY_HEADER_SIZE
    pos_size = 6 if version == 1 else 9
    pos_buf = blob[offset : offset + num_points * pos_size]
    offset += num_points * pos_size

    alpha_buf = blob[offset : offset + num_points]
    offset += num_points

    color_buf = blob[offset : offset + num_points * 3]
    offset += num_points * 3

    scale_buf = blob[offset : offset + num_points * 3]
    offset += num_points * 3

    rot_size = 4 if version >= _MIN_SMALLEST_THREE_VERSION else 3
    rot_buf = blob[offset : offset + num_points * rot_size]
    offset += num_points * rot_size

    sh_dim = _SH_DIM_FOR_DEGREE.get(sh_degree, 0)
    sh_buf = blob[offset : offset + num_points * sh_dim * 3]

    if version == 1:
        positions = _decode_positions_float16(pos_buf, num_points)
    else:
        positions = _decode_positions_24bit(pos_buf, num_points, frac_bits)

    if rot_size == 3:
        rotations = _decode_rotations_legacy_3byte(rot_buf, num_points)
    else:
        rotations = _decode_rotations_smallest_three(rot_buf, num_points)

    return {
        "num_points": num_points,
        "sh_degree": sh_degree,
        "sh_dim": sh_dim,
        "positions": positions,
        "opacities": _decode_alphas(alpha_buf),
        "f_dc": _decode_colors(color_buf, num_points),
        "scales": _decode_scales(scale_buf, num_points),
        "rotations": rotations,
        "sh_coeffs": _decode_sh(sh_buf, num_points, sh_dim),
    }


def _parse_ngsp_v4(raw_bytes: bytes) -> dict[str, Any]:
    """Parse NGSP v4: plaintext header, TOC, ZSTD-compressed streams."""
    import zstandard as zstd

    if len(raw_bytes) < _NGSP_HEADER_SIZE:
        raise ValueError("Buffer too small for NGSP v4 header")

    magic, version, num_points, sh_degree, frac_bits, flags, num_streams, toc_offset = struct.unpack_from(
        "<IIIBBBBI", raw_bytes, 0
    )
    if magic != _NGSP_MAGIC:
        raise ValueError(f"Not an SPZ file (magic={magic:#010x})")
    if version != 4:
        raise ValueError(f"Expected NGSP v4, got version {version}")

    offset = toc_offset
    stream_sizes: list[int] = []
    for _ in range(num_streams):
        comp_size, _uncomp_size = struct.unpack_from("<QQ", raw_bytes, offset)
        stream_sizes.append(comp_size)
        offset += _TOC_ENTRY_SIZE

    dctx = zstd.ZstdDecompressor()
    streams: list[bytes] = []
    data_start = offset
    for comp_size in stream_sizes:
        streams.append(dctx.decompress(raw_bytes[data_start : data_start + comp_size]))
        data_start += comp_size

    if len(streams) < 6:
        raise ValueError(f"Expected 6 attribute streams in NGSP v4, got {len(streams)}")

    sh_dim = _SH_DIM_FOR_DEGREE.get(sh_degree, 0)

    return {
        "num_points": num_points,
        "sh_degree": sh_degree,
        "sh_dim": sh_dim,
        "positions": _decode_positions_24bit(streams[0], num_points, frac_bits),
        "opacities": _decode_alphas(streams[1]),
        "f_dc": _decode_colors(streams[2], num_points),
        "scales": _decode_scales(streams[3], num_points),
        "rotations": _decode_rotations_smallest_three(streams[4], num_points),
        "sh_coeffs": _decode_sh(streams[5], num_points, sh_dim),
    }


def _cloud_to_ply_bytes(cloud: dict[str, Any]) -> bytes:
    """Assemble a 3DGS-compatible binary_little_endian PLY from decoded arrays."""
    N = cloud["num_points"]
    sh_dim = cloud["sh_dim"]
    sh_coeffs = cloud["sh_coeffs"]

    dtype_fields: list[tuple[str, str]] = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
    ]
    if sh_coeffs is not None:
        for i in range(sh_dim * 3):
            dtype_fields.append((f"f_rest_{i}", "f4"))
    dtype_fields += [
        ("opacity", "f4"),
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
    ]

    vertex = np.zeros(N, dtype=dtype_fields)
    vertex["x"] = cloud["positions"][:, 0]
    vertex["y"] = cloud["positions"][:, 1]
    vertex["z"] = cloud["positions"][:, 2]
    vertex["f_dc_0"] = cloud["f_dc"][:, 0]
    vertex["f_dc_1"] = cloud["f_dc"][:, 1]
    vertex["f_dc_2"] = cloud["f_dc"][:, 2]
    if sh_coeffs is not None:
        for i in range(sh_dim):
            for ch in range(3):
                vertex[f"f_rest_{i * 3 + ch}"] = sh_coeffs[:, i, ch]
    vertex["opacity"] = cloud["opacities"]
    vertex["scale_0"] = cloud["scales"][:, 0]
    vertex["scale_1"] = cloud["scales"][:, 1]
    vertex["scale_2"] = cloud["scales"][:, 2]
    vertex["rot_0"] = cloud["rotations"][:, 0]
    vertex["rot_1"] = cloud["rotations"][:, 1]
    vertex["rot_2"] = cloud["rotations"][:, 2]
    vertex["rot_3"] = cloud["rotations"][:, 3]

    prop_names = [name for name, _ in dtype_fields]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {N}\n"
        + "\n".join(f"property float {name}" for name in prop_names)
        + "\nend_header\n"
    )
    buf = io.BytesIO()
    buf.write(header.encode("ascii"))
    buf.write(vertex.tobytes())
    return buf.getvalue()


def _spz_bytes_to_ply_bytes(spz_bytes: bytes) -> bytes:
    """Decode SPZ bytes (legacy gzip v1/v2/v3 or NGSP v4) and return PLY bytes.

    Produces the standard inria/gaussian-splatting PLY layout so the result opens
    in Blender, Unreal, and any 3DGS viewer.
    """
    if len(spz_bytes) < 2:
        raise ValueError("SPZ buffer is empty")
    cloud = _parse_legacy_spz(spz_bytes) if spz_bytes[:2] == _GZIP_MAGIC else _parse_ngsp_v4(spz_bytes)
    return _cloud_to_ply_bytes(cloud)


class ConvertSpzToPly(DataNode):
    """Convert an SPZ Gaussian splat to PLY for broader software compatibility.

    Reads any SPZ artifact or URL (e.g. from WorldLabsWorldGeneration), converts
    the compressed splat data to the standard inria/gaussian-splatting PLY format,
    and saves it to disk. The output SplatUrlArtifact can be wired downstream to
    LoadSplat, or opened in Blender, Unreal, or any PLY-compatible viewer.
    """

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)
        self.category = "Splat"
        self.description = "Convert an SPZ Gaussian splat to PLY for broader software compatibility."

        self.add_parameter(
            ParameterSplat(
                name="splat_in",
                tooltip="Wire an SPZ splat (e.g. from WorldLabsWorldGeneration) or pick a local .spz file.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                accept_any=True,
                clickable_file_browser=True,
                ui_options={"display_name": "Splat In (SPZ)"},
            )
        )

        self._output_file = ProjectFileParameter(
            node=self,
            name="output_file",
            default_filename="splat_converted.ply",
        )
        self._output_file.add_parameter()

        self.add_parameter(
            ParameterSplat(
                name="splat_out",
                tooltip="The PLY splat for downstream wiring or viewing.",
                allowed_modes={ParameterMode.OUTPUT},
                clickable_file_browser=False,
                settable=False,
                ui_options={"display_name": "Splat Out (PLY)"},
            )
        )

    async def aprocess(self) -> None:
        raw = self.get_parameter_value("splat_in")
        url = self._extract_splat_url(raw)
        if not url:
            self.parameter_output_values["splat_out"] = None
            return

        upstream_meta = dict(getattr(raw, "meta", {}) or {})
        if upstream_meta.get("format") == "ply":
            self.parameter_output_values["splat_out"] = (
                raw if isinstance(raw, SplatUrlArtifact) else SplatUrlArtifact(value=url, meta=upstream_meta)
            )
            return

        try:
            spz_bytes = await File(url).aread_bytes()
        except FileLoadError as exc:
            raise RuntimeError(f"Failed to load SPZ from {url!r}: {exc}") from exc

        ply_bytes = await asyncio.to_thread(_spz_bytes_to_ply_bytes, spz_bytes)

        dest: FileDestination = self._output_file.build_file()
        try:
            saved = await dest.awrite_bytes(ply_bytes)
        except Exception as exc:
            raise RuntimeError(f"Failed to save PLY: {exc}") from exc

        upstream_meta["format"] = "ply"
        self.parameter_output_values["splat_out"] = SplatUrlArtifact(
            value=saved.location,
            meta=upstream_meta,
        )

    @staticmethod
    def _extract_splat_url(raw: Any) -> str | None:
        if not raw:
            return None
        if isinstance(raw, str):
            return raw.strip() or None
        url = getattr(raw, "value", None)
        if isinstance(url, str) and url:
            return url
        if isinstance(raw, dict):
            url = raw.get("value") or raw.get("url")
            return url if isinstance(url, str) and url else None
        return None
