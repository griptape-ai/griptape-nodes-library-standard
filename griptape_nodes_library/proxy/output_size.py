"""Pure output-size estimators used by the direct-download guard in ``GriptapeProxyNode``.

All sizes are in bytes. Messages shown to users use decimal gigabytes (1 GB = 1e9 bytes).
"""

from __future__ import annotations

BYTES_PER_GB = 1_000_000_000
BYTES_PER_MB = 1_000_000

# LTX's published ProRes 4444 data rates, in decimal megabytes per second.
PRORES_4444_MB_PER_SECOND_1080P = 42.0
PRORES_4444_MB_PER_SECOND_2160P = 172.0
PIXELS_1080P = 1920 * 1080
# 3840x2160 is exactly 4x the pixels of 1080p; 172 / 42 is about 4.1, so scaling by pixel count
# from the 1080p figure lands within a few percent of LTX's 4K figure.

# Half-float RGB: three channels of two bytes each.
EXR_HALF_FLOAT_RGB_BYTES_PER_PIXEL = 6
# Fraction of the uncompressed size an EXR ZIP frame occupies. 1.0 is conservative (assumes no
# compression), so the guard errs towards asking for direct download. To be calibrated from real jobs.
EXR_ZIP_COMPRESSION_FACTOR = 1.0


def format_gb(size_bytes: float) -> str:
    """Format a byte count as decimal gigabytes, e.g. ``"2.4 GB"``."""
    return f"{size_bytes / BYTES_PER_GB:.1f} GB"


def estimate_prores_4444_bytes(width: int, height: int, duration_seconds: float) -> int:
    """Estimate the size of a ProRes 4444 video, scaling LTX's 1080p rate by pixel count."""
    if width <= 0 or height <= 0 or duration_seconds <= 0:
        return 0
    mb_per_second = PRORES_4444_MB_PER_SECOND_1080P * (width * height) / PIXELS_1080P
    return int(mb_per_second * BYTES_PER_MB * duration_seconds)


def estimate_exr_zip_bytes(frames: int, width: int, height: int) -> int:
    """Estimate the size of an EXR half-float RGB ZIP sequence."""
    if frames <= 0 or width <= 0 or height <= 0:
        return 0
    raw = frames * width * height * EXR_HALF_FLOAT_RGB_BYTES_PER_PIXEL
    return int(raw * EXR_ZIP_COMPRESSION_FACTOR)
