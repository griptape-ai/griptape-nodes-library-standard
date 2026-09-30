"""Tests for image_utils helpers."""

from __future__ import annotations

import numpy as np
import pytest
from griptape.artifacts import ImageUrlArtifact
from PIL import Image

from griptape_nodes_library.utils.image_utils import extract_image_url, premultiply_rgba, unpremultiply_rgba


class TestExtractImageUrl:
    """The falsy-on-missing contract callers rely on.

    Callers guard with ``if not extract_image_url(...)``. Returning the string
    ``"None"`` for a missing image would slip past that guard and only fail later
    with an opaque AttributeError, so absence must come back falsy.
    """

    def test_none_returns_empty_string(self) -> None:
        assert extract_image_url(None) == ""

    def test_none_is_falsy_not_the_literal_none_string(self) -> None:
        result = extract_image_url(None)
        assert not result
        assert result != "None"

    def test_dict_with_none_value_returns_empty_string(self) -> None:
        assert extract_image_url({"value": None}) == ""

    def test_artifact_returns_its_url(self) -> None:
        assert extract_image_url(ImageUrlArtifact("https://example.com/a.png")) == "https://example.com/a.png"

    def test_dict_returns_its_value(self) -> None:
        assert extract_image_url({"value": "https://example.com/b.png"}) == "https://example.com/b.png"

    def test_string_passes_through(self) -> None:
        assert extract_image_url("https://example.com/c.png") == "https://example.com/c.png"

    def test_empty_string_stays_empty(self) -> None:
        assert extract_image_url("") == ""


def alpha_ramp(rgb: tuple[int, int, int]) -> Image.Image:
    """One row of a flat colour with alpha running 0..255 across it."""
    arr = np.zeros((1, 256, 4), dtype=np.uint8)
    arr[..., :3] = rgb
    arr[0, :, 3] = np.arange(256)
    return Image.fromarray(arr, mode="RGBA")


class TestPremultiplyRgba:
    def test_scales_colour_by_alpha(self) -> None:
        out = np.asarray(premultiply_rgba(alpha_ramp((200, 100, 50)))).astype(int)
        assert out[0, 128].tolist() == [100, 50, 25, 128]

    def test_leaves_alpha_unchanged(self) -> None:
        out = np.asarray(premultiply_rgba(alpha_ramp((200, 100, 50))))
        assert out[0, :, 3].tolist() == list(range(256))

    def test_opaque_and_transparent_endpoints(self) -> None:
        out = np.asarray(premultiply_rgba(alpha_ramp((200, 100, 50)))).astype(int)
        assert out[0, 0, :3].tolist() == [0, 0, 0]
        assert out[0, 255, :3].tolist() == [200, 100, 50]

    def test_invert_scales_by_complement(self) -> None:
        out = np.asarray(premultiply_rgba(alpha_ramp((200, 100, 50)), invert=True)).astype(int)
        assert out[0, 0, :3].tolist() == [200, 100, 50]
        assert out[0, 255, :3].tolist() == [0, 0, 0]

    def test_rgb_input_is_treated_as_opaque(self) -> None:
        out = premultiply_rgba(Image.new("RGB", (2, 2), (10, 20, 30)))
        assert out.getpixel((0, 0)) == (10, 20, 30, 255)


class TestUnpremultiplyRgba:
    def test_divides_colour_by_alpha(self) -> None:
        img = Image.new("RGBA", (1, 1), (100, 50, 25, 128))
        r, g, b, a = unpremultiply_rgba(img).getpixel((0, 0))  # type: ignore[misc]
        assert (r, g, b, a) == (199, 100, 50, 128)

    def test_zero_alpha_becomes_black(self) -> None:
        img = Image.new("RGBA", (1, 1), (100, 50, 25, 0))
        assert unpremultiply_rgba(img).getpixel((0, 0)) == (0, 0, 0, 0)

    def test_over_range_colour_is_clipped(self) -> None:
        # Colour above alpha is invalid premultiplied data; it must clip, not wrap.
        img = Image.new("RGBA", (1, 1), (200, 0, 0, 100))
        assert unpremultiply_rgba(img).getpixel((0, 0)) == (255, 0, 0, 100)

    @pytest.mark.parametrize("invert", [False, True])
    def test_round_trip_recovers_colour(self, invert: bool) -> None:  # noqa: FBT001
        src = alpha_ramp((200, 100, 50))
        out = np.asarray(unpremultiply_rgba(premultiply_rgba(src, invert=invert), invert=invert)).astype(int)
        # 8-bit premultiplied colour quantises coarsely near zero, so only check where
        # the multiplier leaves enough levels for a tight bound.
        factor = np.arange(256) / 255.0
        if invert:
            factor = 1.0 - factor
        usable = factor >= 0.25  # noqa: PLR2004
        err = np.abs(out[0, usable, :3] - np.array([200, 100, 50]))
        assert err.max() <= 2  # noqa: PLR2004
