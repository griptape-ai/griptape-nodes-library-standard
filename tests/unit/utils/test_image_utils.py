"""Tests for image_utils helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from griptape.artifacts import ImageArtifact, ImageUrlArtifact
from griptape_nodes.image import Exif, GriptapeImage, ImageFormat, ImageMetadata

from griptape_nodes_library.utils import image_utils
from griptape_nodes_library.utils.image_utils import (
    ImageMode,
    apply_border_radius,
    apply_mask_transformations,
    converted,
    edited_pixels,
    extract_channel_from_image,
    extract_image_url,
    get_image_color_info,
    image_artifact_from_bytes,
    load_image_from_artifact,
    load_image_from_url,
    luma,
    open_image_bytes,
    pixels_of,
    premultiply_rgba,
    resize_image_for_resolution,
    shrink_image_to_size,
    unpremultiply_rgba,
    upright,
    validate_image_format,
)

if TYPE_CHECKING:
    from pathlib import Path


def pixel(image: GriptapeImage, x: int = 0, y: int = 0) -> tuple[int, ...]:
    """One pixel's channel values, as ``getpixel`` returned them."""
    return tuple(int(value) for value in image.to_array()[y, x])


def gray_values(image: GriptapeImage) -> list[int]:
    """The distinct values of a one-channel image."""
    assert image.channels == 1
    return np.unique(image.to_array()).tolist()


def png_bytes(pixels: np.ndarray) -> bytes:
    return GriptapeImage.from_array(pixels).encode(ImageFormat.PNG)


_ORIENTATION_TAG = 0x0112
_ROTATE_90_CLOCKWISE = 6


def rotated_jpeg(width: int, height: int) -> bytes:
    """A JPEG whose left half is red, tagged to display turned 90 degrees clockwise."""
    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    pixels[:, : width // 2] = (255, 0, 0)
    exif = Exif()
    exif[_ORIENTATION_TAG] = _ROTATE_90_CLOCKWISE
    return GriptapeImage.from_array(pixels, ImageMetadata(exif=exif)).encode(ImageFormat.JPEG, quality=95)


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


def alpha_ramp(rgb: tuple[int, int, int]) -> GriptapeImage:
    """One row of a flat colour with alpha running 0..255 across it."""
    arr = np.zeros((1, 256, 4), dtype=np.uint8)
    arr[..., :3] = rgb
    arr[0, :, 3] = np.arange(256)
    return GriptapeImage.from_array(arr)


class TestPremultiplyRgba:
    def test_scales_colour_by_alpha(self) -> None:
        out = premultiply_rgba(alpha_ramp((200, 100, 50))).to_array().astype(int)
        assert out[0, 128].tolist() == [100, 50, 25, 128]

    def test_leaves_alpha_unchanged(self) -> None:
        out = premultiply_rgba(alpha_ramp((200, 100, 50))).to_array()
        assert out[0, :, 3].tolist() == list(range(256))

    def test_opaque_and_transparent_endpoints(self) -> None:
        out = premultiply_rgba(alpha_ramp((200, 100, 50))).to_array().astype(int)
        assert out[0, 0, :3].tolist() == [0, 0, 0]
        assert out[0, 255, :3].tolist() == [200, 100, 50]

    def test_invert_scales_by_complement(self) -> None:
        out = premultiply_rgba(alpha_ramp((200, 100, 50)), invert=True).to_array().astype(int)
        assert out[0, 0, :3].tolist() == [200, 100, 50]
        assert out[0, 255, :3].tolist() == [0, 0, 0]

    def test_rgb_input_is_treated_as_opaque(self) -> None:
        out = premultiply_rgba(GriptapeImage.new(2, 2, (10, 20, 30)))
        assert pixel(out) == (10, 20, 30, 255)


class TestUnpremultiplyRgba:
    def test_divides_colour_by_alpha(self) -> None:
        img = GriptapeImage.new(1, 1, (100, 50, 25, 128))
        assert pixel(unpremultiply_rgba(img)) == (199, 100, 50, 128)

    def test_zero_alpha_becomes_black(self) -> None:
        img = GriptapeImage.new(1, 1, (100, 50, 25, 0))
        assert pixel(unpremultiply_rgba(img)) == (0, 0, 0, 0)

    def test_over_range_colour_is_clipped(self) -> None:
        # Colour above alpha is invalid premultiplied data; it must clip, not wrap.
        img = GriptapeImage.new(1, 1, (200, 0, 0, 100))
        assert pixel(unpremultiply_rgba(img)) == (255, 0, 0, 100)

    @pytest.mark.parametrize("invert", [False, True])
    def test_round_trip_recovers_colour(self, invert: bool) -> None:  # noqa: FBT001
        src = alpha_ramp((200, 100, 50))
        out = unpremultiply_rgba(premultiply_rgba(src, invert=invert), invert=invert).to_array().astype(int)
        # 8-bit premultiplied colour quantises coarsely near zero, so only check where
        # the multiplier leaves enough levels for a tight bound.
        factor = np.arange(256) / 255.0
        if invert:
            factor = 1.0 - factor
        usable = factor >= 0.25  # noqa: PLR2004
        err = np.abs(out[0, usable, :3] - np.array([200, 100, 50]))
        assert err.max() <= 2  # noqa: PLR2004


class TestExtractChannelFromImage:
    def test_alpha_of_rgb_image_is_fully_opaque(self) -> None:
        image = GriptapeImage.new(3, 2, (200, 100, 50))
        mask = extract_channel_from_image(image, "alpha")
        assert get_image_color_info(mask).color_space == "L"
        assert (mask.width, mask.height) == (3, 2)
        assert gray_values(mask) == [255]

    @pytest.mark.parametrize(("channel", "expected"), [("red", 200), ("green", 100), ("blue", 50)])
    def test_colour_channels_of_rgb_image(self, channel: str, expected: int) -> None:
        mask = extract_channel_from_image(GriptapeImage.new(2, 2, (200, 100, 50)), channel)
        assert gray_values(mask) == [expected]

    @pytest.mark.parametrize(("channel", "expected"), [("red", 200), ("green", 100), ("blue", 50), ("alpha", 7)])
    def test_channels_of_rgba_image(self, channel: str, expected: int) -> None:
        mask = extract_channel_from_image(GriptapeImage.new(2, 2, (200, 100, 50, 7)), channel)
        assert gray_values(mask) == [expected]

    @pytest.mark.parametrize(("channel", "expected"), [("red", 9), ("green", 9), ("blue", 9), ("alpha", 77)])
    def test_channels_of_gray_alpha_image(self, channel: str, expected: int) -> None:
        mask = extract_channel_from_image(GriptapeImage.new(2, 2, (9, 77)), channel)
        assert gray_values(mask) == [expected]

    @pytest.mark.parametrize(("channel", "expected"), [("red", 9), ("green", 9), ("blue", 9), ("alpha", 9)])
    def test_channels_of_gray_image(self, channel: str, expected: int) -> None:
        mask = extract_channel_from_image(GriptapeImage.new(2, 2, (9,)), channel)
        assert gray_values(mask) == [expected]

    def test_sixteen_bit_channel_is_scaled_to_eight_bit(self) -> None:
        pixels = np.full((2, 2, 3), (65535, 32896, 0), dtype=np.uint16)
        mask = extract_channel_from_image(GriptapeImage.from_array(pixels), "green")
        assert mask.to_array().dtype == np.uint8
        assert gray_values(mask) == [128]

    def test_unknown_channel_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown mask channel: 'cyan'"):
            extract_channel_from_image(GriptapeImage.new(2, 2, (1, 2, 3)), "cyan", "mask")


class TestUpright:
    def test_opening_turns_the_pixels_upright_and_resets_the_tag(self) -> None:
        image = open_image_bytes(rotated_jpeg(40, 20), "test")
        assert (image.width, image.height) == (20, 40)
        # Turned clockwise, the red left half is on top
        red, _, blue = pixel(image, 10, 5)
        assert red > 200 > blue  # noqa: PLR2004
        assert image.metadata.exif is not None
        assert image.metadata.exif.orientation == 1

    def test_header_reads_keep_the_stored_pixels(self) -> None:
        image = open_image_bytes(rotated_jpeg(40, 20), "test", turn_upright=False)
        assert (image.width, image.height) == (40, 20)
        assert image.metadata.exif is not None
        assert image.metadata.exif.orientation == _ROTATE_90_CLOCKWISE

    def test_an_upright_image_is_returned_as_is(self) -> None:
        image = GriptapeImage.open(png_bytes(np.zeros((2, 2, 3), dtype=np.uint8)))
        assert upright(image) is image

    def test_reencoded_edit_displays_upright(self) -> None:
        image = open_image_bytes(rotated_jpeg(40, 20), "test")
        reread = GriptapeImage.open(image.encode(ImageFormat.PNG))
        # A viewer honoring the tag shows what the node edited
        assert reread.metadata.exif is None or reread.metadata.exif.orientation == 1
        assert (reread.width, reread.height) == (20, 40)


class TestLuma:
    @pytest.mark.parametrize(
        ("rgb", "expected"),
        [((255, 0, 0), 76), ((0, 255, 0), 150), ((0, 0, 255), 29), ((255, 255, 255), 255), ((0, 0, 0), 0)],
    )
    def test_matches_pillow_l_conversion(self, rgb: tuple[int, int, int], expected: int) -> None:
        assert luma(np.array([[rgb]], dtype=np.uint8)).tolist() == [[expected]]

    def test_rounds_like_pillow_fixed_point(self) -> None:
        # 0.299 * 230 + 0.587 * 156 + 0.114 * 247 is 188.5; Pillow's 16-bit weights give 188.
        assert luma(np.array([[[230, 156, 247]]], dtype=np.uint8)).tolist() == [[188]]


def primaries() -> GriptapeImage:
    """Red, green, blue, white, and an arbitrary colour, in one row."""
    return GriptapeImage.from_array(
        np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255], [10, 200, 77]]], dtype=np.uint8)
    )


class TestConverted:
    def test_rgb_to_gray_is_pillow_luma(self) -> None:
        gray = converted(primaries(), ImageMode.GRAY)
        assert gray.channels == 1
        assert gray.to_array()[0, :, 0].tolist() == [76, 150, 29, 255, 129]

    def test_rgb_to_gray_alpha_is_luma_with_opaque_alpha(self) -> None:
        gray_alpha = converted(primaries(), ImageMode.GRAY_ALPHA)
        assert gray_alpha.channels == 2  # noqa: PLR2004
        assert gray_alpha.to_array()[0].tolist() == [[76, 255], [150, 255], [29, 255], [255, 255], [129, 255]]

    def test_rgba_to_gray_drops_alpha(self) -> None:
        gray = converted(GriptapeImage.new(2, 2, (255, 0, 0, 10)), ImageMode.GRAY)
        assert gray.channels == 1
        assert gray_values(gray) == [76]

    def test_rgba_to_gray_alpha_keeps_alpha(self) -> None:
        gray_alpha = converted(GriptapeImage.new(2, 2, (0, 255, 0, 10)), ImageMode.GRAY_ALPHA)
        assert pixel(gray_alpha) == (150, 10)

    def test_rgba_to_rgb_drops_alpha_without_flattening(self) -> None:
        rgb = converted(GriptapeImage.new(2, 2, (132, 103, 47, 71)), ImageMode.RGB)
        assert pixel(rgb) == (132, 103, 47)

    def test_rgb_to_rgba_adds_opaque_alpha(self) -> None:
        assert pixel(converted(GriptapeImage.new(2, 2, (1, 2, 3)), ImageMode.RGBA)) == (1, 2, 3, 255)

    def test_gray_to_rgb_repeats_gray(self) -> None:
        assert pixel(converted(GriptapeImage.new(2, 2, (109,)), ImageMode.RGB)) == (109, 109, 109)

    def test_gray_alpha_to_rgba_keeps_alpha(self) -> None:
        assert pixel(converted(GriptapeImage.new(2, 2, (109, 78)), ImageMode.RGBA)) == (109, 109, 109, 78)

    @pytest.mark.parametrize("mode", list(ImageMode))
    def test_every_mode_is_eight_bit_with_its_channel_count(self, mode: ImageMode) -> None:
        out = converted(GriptapeImage.new(3, 2, (1, 2, 3, 4)), mode)
        assert get_image_color_info(out).color_space == str(mode)
        assert (out.width, out.height) == (3, 2)

    def test_sixteen_bit_gray_is_scaled_not_clipped(self) -> None:
        # What grayscale_to_eight_bit did: 16-bit gray scaled to 8 bits instead of clipped to 255
        sixteen = GriptapeImage.from_array(np.array([[0, 257, 32896, 65535]], dtype=np.uint16))
        out = converted(sixteen, ImageMode.GRAY)
        assert out.to_array().dtype == np.uint8
        assert out.to_array()[0, :, 0].tolist() == [0, 1, 128, 255]

    @pytest.mark.parametrize("mode", list(ImageMode))
    def test_sixteen_bit_colour_is_scaled_in_every_mode(self, mode: ImageMode) -> None:
        sixteen = GriptapeImage.from_array(np.full((2, 2, 4), (65535, 0, 32896, 65535), dtype=np.uint16))
        eight = GriptapeImage.new(2, 2, (255, 0, 128, 255))
        assert converted(sixteen, mode).to_array().tolist() == converted(eight, mode).to_array().tolist()

    @pytest.mark.parametrize("mode", list(ImageMode))
    def test_float_endpoints_map_to_eight_bit_endpoints(self, mode: ImageMode) -> None:
        floats = GriptapeImage.from_array(np.array([[0.0, 1.0]], dtype=np.float32))
        out = converted(floats, mode).to_array()
        assert out.dtype == np.uint8
        assert out[0, 0, 0] == 0
        assert out[0, 1, 0] == 255  # noqa: PLR2004

    def test_float_gray_is_scaled_linearly(self) -> None:
        floats = GriptapeImage.from_array(np.array([[0.25, 0.5]], dtype=np.float32))
        out = converted(floats, ImageMode.GRAY).to_array()[0, :, 0].astype(int)
        assert abs(out[0] - 63.75) <= 1
        assert abs(out[1] - 127.5) <= 1

    def test_gray_matches_pillow_luma_for_every_pixel(self) -> None:
        pixels = np.random.default_rng(1).integers(0, 256, (256, 256, 3), dtype=np.uint8)
        gray = converted(GriptapeImage.from_array(pixels), ImageMode.GRAY).to_array()[..., 0]
        assert gray.tolist() == luma(pixels).tolist()

    def test_gray_alpha_matches_pillow_luma_for_every_pixel(self) -> None:
        pixels = np.random.default_rng(1).integers(0, 256, (256, 256, 3), dtype=np.uint8)
        gray = converted(GriptapeImage.from_array(pixels), ImageMode.GRAY_ALPHA).to_array()[..., 0]
        assert gray.tolist() == luma(pixels).tolist()

    def test_keeps_metadata(self) -> None:
        image = GriptapeImage.from_array(
            np.zeros((2, 2, 3), dtype=np.uint8), metadata=ImageMetadata(text={"workflow": "abc"})
        )
        assert converted(image, ImageMode.GRAY).metadata.text == {"workflow": "abc"}


class TestPixelsOf:
    def test_is_three_dimensional_uint8_in_mode(self) -> None:
        pixels = pixels_of(GriptapeImage.new(3, 2, (255, 0, 0)), ImageMode.GRAY)
        assert pixels.shape == (2, 3, 1)
        assert pixels.dtype == np.uint8
        assert np.unique(pixels).tolist() == [76]


class TestEditedPixels:
    def test_applies_function_to_converted_pixels(self) -> None:
        out = edited_pixels(GriptapeImage.new(2, 2, (10, 20, 30)), ImageMode.RGBA, lambda p: 255 - p)
        assert pixel(out) == (245, 235, 225, 0)

    def test_function_may_change_channel_count(self) -> None:
        out = edited_pixels(GriptapeImage.new(2, 2, (10, 20, 30)), ImageMode.RGB, lambda p: p[:, :, :1].copy())
        assert out.channels == 1
        assert gray_values(out) == [10]

    def test_keeps_metadata_text(self) -> None:
        image = GriptapeImage.from_array(
            np.zeros((2, 2, 3), dtype=np.uint8), metadata=ImageMetadata(text={"workflow": "abc"})
        )
        out = edited_pixels(image, ImageMode.RGB, lambda p: 255 - p)
        assert out.metadata.text == {"workflow": "abc"}
        assert GriptapeImage.open(out.encode(ImageFormat.PNG)).metadata.text == {"workflow": "abc"}

    def test_keeps_metadata_text_of_opened_image(self) -> None:
        source = GriptapeImage.from_array(
            np.zeros((2, 2, 3), dtype=np.uint8), metadata=ImageMetadata(text={"workflow": "abc"})
        ).encode(ImageFormat.PNG)
        out = edited_pixels(GriptapeImage.open(source), ImageMode.GRAY, lambda p: p)
        assert out.metadata.text == {"workflow": "abc"}


class TestApplyBorderRadius:
    def test_zero_radius_returns_image_unchanged(self) -> None:
        image = GriptapeImage.new(4, 4, (1, 2, 3))
        assert apply_border_radius(image, 0) is image

    def test_rgb_gets_transparent_corners_and_opaque_centre(self) -> None:
        out = apply_border_radius(GriptapeImage.new(10, 10, (255, 0, 0)), 4)
        assert out.channels == 4  # noqa: PLR2004
        assert pixel(out, 0, 0)[3] == 0
        assert pixel(out, 9, 9)[3] == 0
        assert pixel(out, 5, 5) == (255, 0, 0, 255)

    def test_keeps_existing_alpha_inside_the_corners(self) -> None:
        out = apply_border_radius(GriptapeImage.new(10, 10, (255, 0, 0, 128)), 4)
        assert pixel(out, 0, 0) == (255, 0, 0, 0)
        assert pixel(out, 5, 5) == (255, 0, 0, 128)

    @pytest.mark.parametrize("alpha", [255, 128])
    def test_straight_edges_keep_full_alpha(self, alpha: int) -> None:
        out = apply_border_radius(GriptapeImage.new(10, 10, (255, 0, 0, alpha)), 4).to_array()[:, :, 3]
        # Between the corners, the outermost rows and columns are inside the rounded rectangle
        for edge in (out[0, 4:6], out[9, 4:6], out[4:6, 0], out[4:6, 9]):
            assert edge.tolist() == [alpha, alpha]

    def test_transparent_pixels_stay_transparent(self) -> None:
        out = apply_border_radius(GriptapeImage.new(10, 10, (255, 0, 0, 0)), 4)
        assert np.unique(out.to_array()[:, :, 3]).tolist() == [0]


class TestGetImageColorInfo:
    @pytest.mark.parametrize(
        ("color", "expected"), [((1,), "L"), ((1, 2), "LA"), ((1, 2, 3), "RGB"), ((1, 2, 3, 4), "RGBA")]
    )
    def test_eight_bit_names_are_pillow_modes(self, color: tuple[int, ...], expected: str) -> None:
        info = get_image_color_info(GriptapeImage.new(2, 2, color))
        assert info.color_space == expected
        assert info.channels == len(color)

    @pytest.mark.parametrize(
        ("dtype", "channels", "expected"),
        [
            (np.uint16, 1, "I;16"),
            (np.uint16, 2, "LA;uint16"),
            (np.uint16, 3, "RGB;uint16"),
            (np.uint16, 4, "RGBA;uint16"),
            (np.float16, 3, "RGB;half"),
            (np.float16, 4, "RGBA;half"),
            (np.float32, 1, "L;float"),
            (np.float32, 3, "RGB;float"),
        ],
    )
    def test_other_pixel_types_are_named_after_the_mode(self, dtype: type, channels: int, expected: str) -> None:
        info = get_image_color_info(GriptapeImage.from_array(np.zeros((2, 2, channels), dtype=dtype)))
        assert info.color_space == expected
        assert info.channels == channels


class TestImageArtifactFromBytes:
    @pytest.mark.parametrize(("image_format", "expected"), [(ImageFormat.PNG, "png"), (ImageFormat.JPEG, "jpeg")])
    def test_reads_format_and_size_from_header(self, image_format: ImageFormat, expected: str) -> None:
        data = GriptapeImage.new(5, 3, (1, 2, 3)).encode(image_format)
        artifact = image_artifact_from_bytes(data)
        assert isinstance(artifact, ImageArtifact)
        assert artifact.value == data
        assert artifact.format == expected
        assert (artifact.width, artifact.height) == (5, 3)

    def test_unreadable_bytes_raise_value_error(self) -> None:
        with pytest.raises(ValueError, match="Failed to load image from: upload"):
            image_artifact_from_bytes(b"not an image at all", "upload")

    def test_svg_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="SVG files are not supported"):
            image_artifact_from_bytes(b'<svg xmlns="http://www.w3.org/2000/svg"></svg>')


class TestShrinkImageToSize:
    def test_under_the_limit_returns_the_same_bytes(self) -> None:
        data = png_bytes(np.zeros((4, 4, 3), dtype=np.uint8))
        assert shrink_image_to_size(data, len(data)) is data

    def test_tries_lossless_webp_first(self) -> None:
        # A flat BMP is large; lossless WEBP of it is tiny, so the first attempt fits
        data = GriptapeImage.new(64, 64, (10, 20, 30)).encode(ImageFormat.BMP)
        limit = len(data) // 10
        out = shrink_image_to_size(data, limit)
        assert len(out) <= limit
        shrunk = GriptapeImage.open(out)
        assert shrunk.format == ImageFormat.WEBP
        assert (shrunk.width, shrunk.height) == (64, 64)
        # Lossless: every pixel is exactly the source colour
        assert np.unique(shrunk.to_array().reshape(-1, shrunk.channels), axis=0).tolist() == [[10, 20, 30]]

    def test_falls_back_to_lossy_when_lossless_is_too_big(self) -> None:
        noise = np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8)
        data = png_bytes(noise)
        lossless_size = len(GriptapeImage.from_array(noise).encode(ImageFormat.WEBP, lossless=True))
        limit = lossless_size // 2
        out = shrink_image_to_size(data, limit)
        assert len(out) <= limit
        assert GriptapeImage.open(out).format == ImageFormat.WEBP

    def test_unreadable_bytes_are_returned_unchanged(self) -> None:
        data = b"not an image" * 100
        assert shrink_image_to_size(data, 10) is data


class _OpensAsMpo(GriptapeImage):
    """Opens images reporting MPO, which sniffing never reports for real JPEG bytes."""

    @classmethod
    def open(cls, source: bytes | Path) -> GriptapeImage:
        image = GriptapeImage.open(source)
        image._format = ImageFormat.MPO  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        return image


class TestResizeImageForResolution:
    def test_within_bounds_returns_the_same_bytes(self) -> None:
        data = png_bytes(np.zeros((4, 8, 3), dtype=np.uint8))
        assert resize_image_for_resolution(data, 8) is data

    @pytest.mark.parametrize("image_format", [ImageFormat.PNG, ImageFormat.JPEG, ImageFormat.WEBP])
    def test_keeps_writable_source_format(self, image_format: ImageFormat) -> None:
        data = GriptapeImage.new(40, 20, (10, 20, 30)).encode(image_format)
        resized = GriptapeImage.open(resize_image_for_resolution(data, 10))
        assert resized.format == image_format
        assert (resized.width, resized.height) == (10, 5)

    def test_mpo_is_written_as_jpeg(self, monkeypatch: pytest.MonkeyPatch) -> None:
        data = GriptapeImage.new(40, 20, (10, 20, 30)).encode(ImageFormat.JPEG)
        monkeypatch.setattr(image_utils, "GriptapeImage", _OpensAsMpo)
        out = resize_image_for_resolution(data, 10)
        monkeypatch.undo()
        assert out is not data
        resized = GriptapeImage.open(out)
        assert resized.format == ImageFormat.JPEG
        assert (resized.width, resized.height) == (10, 5)

    def test_unwritable_source_format_is_written_as_webp(self) -> None:
        # A PPM opens (as a format ImageFormat doesn't name) but can't be written back
        ppm = b"P6\n4 2\n255\n" + bytes(range(24))
        assert GriptapeImage.open(ppm).format is None
        resized = GriptapeImage.open(resize_image_for_resolution(ppm, 2))
        assert resized.format == ImageFormat.WEBP
        assert (resized.width, resized.height) == (2, 1)

    def test_keeps_metadata_text(self) -> None:
        data = GriptapeImage.from_array(
            np.zeros((20, 40, 3), dtype=np.uint8), metadata=ImageMetadata(text={"workflow": "abc"})
        ).encode(ImageFormat.PNG)
        resized = GriptapeImage.open(resize_image_for_resolution(data, 10))
        assert resized.metadata.text == {"workflow": "abc"}

    def test_unreadable_bytes_are_returned_unchanged(self) -> None:
        data = b"not an image"
        assert resize_image_for_resolution(data, 10) is data


class TestValidateImageFormat:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [("png", ImageFormat.PNG), ("JPEG", ImageFormat.JPEG), ("jpg", ImageFormat.JPEG), ("tif", ImageFormat.TIFF)],
    )
    def test_returns_the_named_format(self, name: str, expected: ImageFormat) -> None:
        assert validate_image_format(name) == expected

    @pytest.mark.parametrize("name", ["MPO", "ICO", "nonsense"])
    def test_unwritable_or_unknown_format_raises(self, name: str) -> None:
        with pytest.raises(ValueError, match=f"Unsupported output_format: '{name}'. Supported formats: BMP, "):
            validate_image_format(name, "output_format")


class TestLoadImageFromUrl:
    def test_loads_local_file(self, tmp_path: Path) -> None:
        path = tmp_path / "image.png"
        path.write_bytes(GriptapeImage.new(5, 3, (1, 2, 3)).encode(ImageFormat.PNG))
        image = load_image_from_url(str(path))
        assert (image.width, image.height) == (5, 3)
        assert pixel(image) == (1, 2, 3)

    def test_svg_extension_raises(self) -> None:
        with pytest.raises(ValueError, match="SVG files are not supported"):
            load_image_from_url("https://example.com/image.SVG")

    def test_svg_content_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "image.png"
        path.write_bytes(b'<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg"></svg>')
        with pytest.raises(ValueError, match="SVG files are not supported"):
            load_image_from_url(str(path))

    def test_missing_file_raises_value_error(self, tmp_path: Path) -> None:
        path = tmp_path / "missing.png"
        with pytest.raises(ValueError, match="Failed to load image from"):
            load_image_from_url(str(path))

    def test_unreadable_file_raises_value_error(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.png"
        path.write_bytes(b"not an image")
        with pytest.raises(ValueError, match="Failed to load image from"):
            load_image_from_url(str(path))


class TestLoadImageFromArtifact:
    def test_image_artifact(self) -> None:
        data = GriptapeImage.new(5, 3, (1, 2, 3)).encode(ImageFormat.PNG)
        image = load_image_from_artifact(ImageArtifact(data, format="png", width=5, height=3), "node")
        assert (image.width, image.height) == (5, 3)

    def test_image_url_artifact_and_string(self, tmp_path: Path) -> None:
        path = tmp_path / "image.png"
        path.write_bytes(GriptapeImage.new(5, 3, (1, 2, 3)).encode(ImageFormat.PNG))
        assert load_image_from_artifact(ImageUrlArtifact(str(path)), "node").width == 5  # noqa: PLR2004
        assert load_image_from_artifact(str(path), "node").width == 5  # noqa: PLR2004

    def test_unreadable_image_artifact_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Failed to load image from: node"):
            load_image_from_artifact(ImageArtifact(b"not an image", format="png", width=1, height=1), "node")

    def test_unsupported_type_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="Unsupported image type: int"):
            load_image_from_artifact(42, "node")  # type: ignore[arg-type]


def square_mask() -> GriptapeImage:
    """A 9x9 black mask with a white 3x3 square in the middle."""
    pixels = np.zeros((9, 9, 1), dtype=np.uint8)
    pixels[3:6, 3:6] = 255
    return GriptapeImage.from_array(pixels)


def white_extent(mask: GriptapeImage) -> tuple[int, int, int, int]:
    """(top, left, bottom, right) of the white pixels, inclusive."""
    ys, xs = np.nonzero(mask.to_array()[:, :, 0] == 255)  # noqa: PLR2004
    return (int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max()))


class TestApplyMaskTransformations:
    def test_defaults_leave_the_mask_unchanged(self) -> None:
        assert apply_mask_transformations(square_mask()).to_array().tolist() == square_mask().to_array().tolist()

    def test_negative_grows(self) -> None:
        assert white_extent(apply_mask_transformations(square_mask(), grow_shrink=-1)) == (2, 2, 6, 6)
        assert white_extent(apply_mask_transformations(square_mask(), grow_shrink=-2)) == (1, 1, 7, 7)

    def test_positive_shrinks(self) -> None:
        assert white_extent(apply_mask_transformations(square_mask(), grow_shrink=1)) == (4, 4, 4, 4)

    def test_fractional_grow_shrink_below_one_pixel_does_nothing(self) -> None:
        out = apply_mask_transformations(square_mask(), grow_shrink=0.5)
        assert out.to_array().tolist() == square_mask().to_array().tolist()

    def test_invert(self) -> None:
        out = apply_mask_transformations(square_mask(), invert=True).to_array()[:, :, 0]
        assert out.tolist() == (255 - square_mask().to_array()[:, :, 0]).tolist()

    def test_grow_shrink_happens_before_invert(self) -> None:
        # Shrinking then inverting leaves one black pixel; inverting first would grow the black square instead
        out = apply_mask_transformations(square_mask(), grow_shrink=1, invert=True).to_array()[:, :, 0]
        assert np.argwhere(out == 0).tolist() == [[4, 4]]

    def test_blur_softens_edges_and_keeps_symmetry(self) -> None:
        out = apply_mask_transformations(square_mask(), blur_radius=1).to_array()[:, :, 0].astype(int)
        assert 0 < out[4, 4] < 255  # noqa: PLR2004
        assert 0 < out[4, 2] < out[4, 3] < out[4, 4]
        assert out.tolist() == out.T.tolist()

    def test_blur_happens_after_invert(self) -> None:
        blurred = apply_mask_transformations(square_mask(), blur_radius=1).to_array().astype(int)
        inverted_then_blurred = apply_mask_transformations(square_mask(), invert=True, blur_radius=1).to_array()
        assert np.abs((255 - blurred) - inverted_then_blurred.astype(int)).max() <= 1
