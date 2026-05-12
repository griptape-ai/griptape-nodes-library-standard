"""Tests for SPZ→PLY decoder functions in convert_spz_to_ply.

Verifies each decode step against hand-computed expected values derived from
the official SPZ spec (nianticlabs/spz), and validates a full round-trip
against the Niantic reference output when the real test files are present.
"""

from __future__ import annotations

import gzip
import io
import math
import struct
from pathlib import Path

import numpy as np
import pytest

from griptape_nodes_library.splat.convert_spz_to_ply import (
    _LEGACY_HEADER_SIZE,
    _NGSP_MAGIC,
    _cloud_to_ply_bytes,
    _decode_alphas,
    _decode_colors,
    _decode_positions_24bit,
    _decode_positions_float16,
    _decode_rotations_legacy_3byte,
    _decode_rotations_smallest_three,
    _decode_scales,
    _parse_legacy_spz,
    _spz_bytes_to_ply_bytes,
)

# ---------------------------------------------------------------------------
# Reference files (skip integration tests if absent)
# ---------------------------------------------------------------------------
_REF_SPZ = Path("/Users/kateforsberg/GriptapeNodes/outputs/splat_500k_3.spz")
_REF_PLY = Path("/Users/kateforsberg/Desktop/splat_500k_3_niantic.ply")
_have_ref = pytest.mark.skipif(
    not (_REF_SPZ.exists() and _REF_PLY.exists()),
    reason="Reference files not present",
)


def _read_ply_arrays(data: bytes) -> dict[str, np.ndarray]:
    buf = io.BytesIO(data)
    props: list[str] = []
    while True:
        line = buf.readline().decode("ascii").strip()
        if line.startswith("property float"):
            props.append(line.split()[-1])
        if line == "end_header":
            break
    dtype = [(p, "f4") for p in props]
    arr = np.frombuffer(buf.read(), dtype=dtype)
    return {p: arr[p] for p in props}


# ---------------------------------------------------------------------------
# Position decoding
# ---------------------------------------------------------------------------
class TestDecodePositions24bit:
    def test_zero(self) -> None:
        buf = bytes(9)  # 3 coords × 3 bytes = 0.0 each
        result = _decode_positions_24bit(buf, 1, frac_bits=12)
        np.testing.assert_array_equal(result, [[0.0, 0.0, 0.0]])

    def test_positive_value(self) -> None:
        # fixed24 = 4096 → float = 4096 / 2^12 = 1.0
        fixed = 4096
        b = [fixed & 0xFF, (fixed >> 8) & 0xFF, (fixed >> 16) & 0xFF]
        buf = bytes(b * 3)  # same for x, y, z
        result = _decode_positions_24bit(buf, 1, frac_bits=12)
        np.testing.assert_allclose(result, [[1.0, 1.0, 1.0]], atol=1e-6)

    def test_negative_value_sign_extension(self) -> None:
        # fixed24 = -1 → all bytes 0xFF
        buf = bytes([0xFF, 0xFF, 0xFF] * 3)
        result = _decode_positions_24bit(buf, 1, frac_bits=12)
        # -1 / 4096 = -0.000244...
        np.testing.assert_allclose(result, [[-1 / 4096, -1 / 4096, -1 / 4096]], atol=1e-6)

    def test_frac_bits_effect(self) -> None:
        fixed = 256
        b = [fixed & 0xFF, (fixed >> 8) & 0xFF, 0]
        buf = bytes(b * 3)
        r8 = _decode_positions_24bit(buf, 1, frac_bits=8)
        r12 = _decode_positions_24bit(buf, 1, frac_bits=12)
        # frac_bits=8 → 256/256=1.0; frac_bits=12 → 256/4096=0.0625
        np.testing.assert_allclose(r8[0, 0], 1.0, atol=1e-6)
        np.testing.assert_allclose(r12[0, 0], 0.0625, atol=1e-6)

    def test_multi_point(self) -> None:
        def encode24(v: int) -> bytes:
            v = v & 0xFFFFFF
            return bytes([v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF])

        p0 = encode24(4096) * 3  # (1.0, 1.0, 1.0)
        p1 = encode24(-4096 & 0xFFFFFF) * 3  # (-1.0, -1.0, -1.0)
        result = _decode_positions_24bit(p0 + p1, 2, frac_bits=12)
        np.testing.assert_allclose(result[0], [1.0, 1.0, 1.0], atol=1e-5)
        np.testing.assert_allclose(result[1], [-1.0, -1.0, -1.0], atol=1e-5)


class TestDecodePositionsFloat16:
    def test_basic(self) -> None:
        arr = np.array([[1.0, -2.0, 0.5]], dtype=np.float16)
        result = _decode_positions_float16(arr.tobytes(), 1)
        np.testing.assert_allclose(result, [[1.0, -2.0, 0.5]], atol=1e-3)


# ---------------------------------------------------------------------------
# Rotation decoding
# ---------------------------------------------------------------------------
class TestDecodeRotationsLegacy3byte:
    def test_identity_quaternion(self) -> None:
        # w=1, x=y=z=0 → encode x=y=z=0 → byte 127 (127.5 * 0 + 127.5 ≈ 127)
        # w is reconstructed as sqrt(1 - 0) = 1.0
        buf = bytes([127, 127, 127])
        result = _decode_rotations_legacy_3byte(buf, 1)
        # PLY order: [w, x, y, z]
        assert result.shape == (1, 4)
        np.testing.assert_allclose(result[0, 0], 1.0, atol=0.01)  # w
        np.testing.assert_allclose(result[0, 1:], 0.0, atol=0.01)  # x,y,z

    def test_output_is_unit_quaternion(self) -> None:
        buf = bytes([100, 150, 200])
        result = _decode_rotations_legacy_3byte(buf, 1)
        norm = np.sqrt(np.sum(result**2))
        np.testing.assert_allclose(norm, 1.0, atol=1e-5)

    def test_ply_order_is_wxyz(self) -> None:
        # x component only (y=z=0, w reconstructed)
        # byte[0]=255 → x = (255/127.5 - 1) ≈ 1.0
        buf = bytes([255, 127, 127])
        result = _decode_rotations_legacy_3byte(buf, 1)
        # w = sqrt(1 - x²) ≈ 0, x ≈ 1; PLY [w,x,y,z] → [~0, ~1, ~0, ~0]
        assert result[0, 1] > 0.9  # x is slot 1


class TestDecodeRotationsSmallestThree:
    def _encode_smallest_three(self, w: float, x: float, y: float, z: float) -> bytes:
        """Encode a quaternion (w,x,y,z) using smallest-three, matching spz C++ packer."""
        components = [x, y, z, w]  # xyzw order internally
        abs_vals = [abs(c) for c in components]
        i_largest = max(range(4), key=lambda i: abs_vals[i])
        # ensure largest is positive
        sign = 1 if components[i_largest] >= 0 else -1
        other_indices = [i for i in range(3, -1, -1) if i != i_largest]
        sqrt1_2 = math.sqrt(0.5)

        packed = i_largest << 30
        shift = 0
        for idx in other_indices:
            val = components[idx] * sign
            mag = int(round(abs(val) / sqrt1_2 * 511))
            mag = min(mag, 511)
            neg_bit = 1 if val < 0 else 0
            field = (neg_bit << 9) | mag
            packed |= field << shift
            shift += 10

        return struct.pack("<I", packed)

    def test_identity_quaternion(self) -> None:
        buf = self._encode_smallest_three(1.0, 0.0, 0.0, 0.0)
        result = _decode_rotations_smallest_three(buf, 1)
        # PLY order [w, x, y, z]
        np.testing.assert_allclose(result[0], [1.0, 0.0, 0.0, 0.0], atol=0.005)

    def test_output_is_unit_quaternion(self) -> None:
        buf = self._encode_smallest_three(0.5, 0.5, 0.5, 0.5)
        result = _decode_rotations_smallest_three(buf, 1)
        norm = np.sqrt(np.sum(result**2))
        np.testing.assert_allclose(norm, 1.0, atol=1e-5)

    def test_round_trip_various_quaternions(self) -> None:
        test_quats = [
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
            (0.7071, 0.7071, 0.0, 0.0),
            (0.5, 0.5, 0.5, 0.5),
        ]
        for w, x, y, z in test_quats:
            norm = math.sqrt(w * w + x * x + y * y + z * z)
            w, x, y, z = w / norm, x / norm, y / norm, z / norm
            buf = self._encode_smallest_three(w, x, y, z)
            result = _decode_rotations_smallest_three(buf, 1)
            # PLY [w, x, y, z] — allow sign flip (q == -q for rotation)
            decoded = result[0]
            expected = np.array([w, x, y, z])
            ok = np.allclose(decoded, expected, atol=0.005) or np.allclose(decoded, -expected, atol=0.005)
            assert ok, f"Round-trip failed for q=({w},{x},{y},{z}): got {decoded}"


# ---------------------------------------------------------------------------
# Attribute decoding
# ---------------------------------------------------------------------------
class TestDecodeAlphas:
    def test_mid_value(self) -> None:
        # byte=127 → p ≈ 0.498 → logit ≈ -0.00784
        result = _decode_alphas(bytes([127]))
        expected = math.log(127 / 255 / (1 - 127 / 255))
        assert abs(result[0] - expected) < 1e-4

    def test_fully_opaque_is_inf(self) -> None:
        result = _decode_alphas(bytes([255]))
        assert math.isinf(result[0]) and result[0] > 0

    def test_fully_transparent_is_neg_inf(self) -> None:
        result = _decode_alphas(bytes([0]))
        assert math.isinf(result[0]) and result[0] < 0

    def test_monotone_increasing(self) -> None:
        buf = bytes(range(1, 255))
        result = _decode_alphas(buf)
        assert (np.diff(result) > 0).all()


class TestDecodeColors:
    def test_mid_byte_near_zero(self) -> None:
        # byte=127 → (127/255 - 0.5) / 0.15 ≈ -0.0131
        buf = bytes([127, 127, 127])
        result = _decode_colors(buf, 1)
        expected = (127 / 255 - 0.5) / 0.15
        np.testing.assert_allclose(result[0], [expected] * 3, atol=1e-4)

    def test_byte_255_encodes_positive_sh(self) -> None:
        buf = bytes([255, 255, 255])
        result = _decode_colors(buf, 1)
        assert (result > 0).all()

    def test_byte_0_encodes_negative_sh(self) -> None:
        buf = bytes([0, 0, 0])
        result = _decode_colors(buf, 1)
        assert (result < 0).all()


class TestDecodeScales:
    def test_byte_160_gives_zero(self) -> None:
        # 160 / 16.0 - 10 = 0.0
        buf = bytes([160, 160, 160])
        result = _decode_scales(buf, 1)
        np.testing.assert_allclose(result[0], [0.0, 0.0, 0.0], atol=1e-5)

    def test_byte_0_gives_minus_ten(self) -> None:
        buf = bytes([0, 0, 0])
        result = _decode_scales(buf, 1)
        np.testing.assert_allclose(result[0], [-10.0, -10.0, -10.0], atol=1e-5)

    def test_byte_255_gives_near_six(self) -> None:
        buf = bytes([255, 255, 255])
        result = _decode_scales(buf, 1)
        np.testing.assert_allclose(result[0], [255 / 16 - 10] * 3, atol=1e-4)


# ---------------------------------------------------------------------------
# Coordinate system conversion (RUB → RDF)
# ---------------------------------------------------------------------------
class TestCoordinateConversion:
    """SPZ uses RUB (Y-up); 3DGS PLY uses RDF (Y-down).
    The fix: negate Y and Z positions, and negate rot_2 (y) and rot_3 (z).
    """

    def _make_minimal_cloud(self, pos, rot) -> dict:
        n = 1
        return {
            "num_points": n,
            "sh_degree": 0,
            "sh_dim": 0,
            "positions": np.array([pos], dtype=np.float32),
            "opacities": np.array([0.0], dtype=np.float32),
            "f_dc": np.zeros((n, 3), dtype=np.float32),
            "scales": np.zeros((n, 3), dtype=np.float32),
            "rotations": np.array([rot], dtype=np.float32),
            "sh_coeffs": None,
        }

    def test_y_is_negated(self) -> None:
        cloud = self._make_minimal_cloud([1.0, 2.0, 3.0], [1.0, 0.0, 0.0, 0.0])
        ply = _read_ply_arrays(_cloud_to_ply_bytes(cloud))
        assert ply["y"][0] == pytest.approx(-2.0)

    def test_z_is_negated(self) -> None:
        cloud = self._make_minimal_cloud([1.0, 2.0, 3.0], [1.0, 0.0, 0.0, 0.0])
        ply = _read_ply_arrays(_cloud_to_ply_bytes(cloud))
        assert ply["z"][0] == pytest.approx(-3.0)

    def test_x_is_unchanged(self) -> None:
        cloud = self._make_minimal_cloud([5.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
        ply = _read_ply_arrays(_cloud_to_ply_bytes(cloud))
        assert ply["x"][0] == pytest.approx(5.0)

    def test_rot0_w_unchanged(self) -> None:
        cloud = self._make_minimal_cloud([0.0, 0.0, 0.0], [0.7071, 0.7071, 0.0, 0.0])
        ply = _read_ply_arrays(_cloud_to_ply_bytes(cloud))
        assert ply["rot_0"][0] == pytest.approx(0.7071, abs=1e-4)

    def test_rot1_x_unchanged(self) -> None:
        cloud = self._make_minimal_cloud([0.0, 0.0, 0.0], [0.7071, 0.7071, 0.0, 0.0])
        ply = _read_ply_arrays(_cloud_to_ply_bytes(cloud))
        assert ply["rot_1"][0] == pytest.approx(0.7071, abs=1e-4)

    def test_rot2_y_negated(self) -> None:
        cloud = self._make_minimal_cloud([0.0, 0.0, 0.0], [0.7071, 0.0, 0.7071, 0.0])
        ply = _read_ply_arrays(_cloud_to_ply_bytes(cloud))
        assert ply["rot_2"][0] == pytest.approx(-0.7071, abs=1e-4)

    def test_rot3_z_negated(self) -> None:
        cloud = self._make_minimal_cloud([0.0, 0.0, 0.0], [0.7071, 0.0, 0.0, 0.7071])
        ply = _read_ply_arrays(_cloud_to_ply_bytes(cloud))
        assert ply["rot_3"][0] == pytest.approx(-0.7071, abs=1e-4)


# ---------------------------------------------------------------------------
# Legacy header parsing
# ---------------------------------------------------------------------------
class TestParseLegacySpz:
    def _make_spz_v2(self, num_points: int = 2, sh_degree: int = 0, frac_bits: int = 12) -> bytes:
        header = struct.pack(
            "<IIIBBBB",
            _NGSP_MAGIC,
            2,  # version
            num_points,
            sh_degree,
            frac_bits,
            0,  # flags
            0,  # reserved
        )
        assert len(header) == _LEGACY_HEADER_SIZE
        # streams: positions(9B each), alphas(1B), colors(3B), scales(3B), rotations(3B)
        # Use zeroed buffers — byte=0 decodes to valid (but extreme) values
        pos = bytes(9 * num_points)
        alpha = bytes([128] * num_points)  # ≈ logit(0) = 0
        color = bytes([128] * 3 * num_points)
        scale = bytes([160] * 3 * num_points)  # log_scale=0
        rot = bytes([127] * 3 * num_points)  # near-identity
        blob = header + pos + alpha + color + scale + rot
        return gzip.compress(blob)

    def test_parses_num_points(self) -> None:
        spz = self._make_spz_v2(num_points=3)
        cloud = _parse_legacy_spz(spz)
        assert cloud["num_points"] == 3

    def test_parses_frac_bits_from_header(self) -> None:
        spz8 = self._make_spz_v2(frac_bits=8)
        spz12 = self._make_spz_v2(frac_bits=12)
        # Position buf is all zeros → float = 0.0 regardless of frac_bits in this case,
        # but we can verify it doesn't raise and returns float32 positions
        c8 = _parse_legacy_spz(spz8)
        c12 = _parse_legacy_spz(spz12)
        assert c8["positions"].dtype == np.float32
        assert c12["positions"].dtype == np.float32

    def test_wrong_magic_raises(self) -> None:
        blob = struct.pack("<IIIBBBB", 0xDEADBEEF, 2, 1, 0, 12, 0, 0) + bytes(100)
        spz = gzip.compress(blob)
        with pytest.raises(ValueError, match="Not an SPZ"):
            _parse_legacy_spz(spz)

    def test_invalid_version_raises(self) -> None:
        blob = struct.pack("<IIIBBBB", _NGSP_MAGIC, 5, 1, 0, 12, 0, 0) + bytes(100)
        spz = gzip.compress(blob)
        with pytest.raises(ValueError, match="version"):
            _parse_legacy_spz(spz)

    def test_non_gzip_raises(self) -> None:
        with pytest.raises(ValueError, match="gzip"):
            _parse_legacy_spz(b"notgzip")


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------
class TestSpzBytesToPlyBytes:
    def test_empty_buffer_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            _spz_bytes_to_ply_bytes(b"")

    def test_one_byte_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            _spz_bytes_to_ply_bytes(b"\x1f")


# ---------------------------------------------------------------------------
# Integration: full round-trip against Niantic reference
# ---------------------------------------------------------------------------
class TestRoundTripVsReference:
    @_have_ref
    def test_point_count_matches(self) -> None:
        ply_bytes = _spz_bytes_to_ply_bytes(_REF_SPZ.read_bytes())
        ours = _read_ply_arrays(ply_bytes)
        ref = _read_ply_arrays(_REF_PLY.read_bytes())
        assert len(ours["x"]) == len(ref["x"])

    @_have_ref
    @pytest.mark.parametrize("field", ["x", "y", "z", "rot_0", "rot_1", "rot_2", "rot_3", "scale_0", "f_dc_0"])
    def test_field_matches_reference(self, field: str) -> None:
        ply_bytes = _spz_bytes_to_ply_bytes(_REF_SPZ.read_bytes())
        ours = _read_ply_arrays(ply_bytes)
        ref = _read_ply_arrays(_REF_PLY.read_bytes())
        np.testing.assert_allclose(
            ours[field], ref[field], atol=1e-4, err_msg=f"Field '{field}' does not match reference"
        )

    @_have_ref
    def test_opacity_matches_reference_including_inf(self) -> None:
        ply_bytes = _spz_bytes_to_ply_bytes(_REF_SPZ.read_bytes())
        ours = _read_ply_arrays(ply_bytes)
        ref = _read_ply_arrays(_REF_PLY.read_bytes())
        # inf positions must align
        assert np.array_equal(np.isfinite(ours["opacity"]), np.isfinite(ref["opacity"]))
        # finite values must match
        mask = np.isfinite(ref["opacity"])
        np.testing.assert_allclose(ours["opacity"][mask], ref["opacity"][mask], atol=1e-4)
