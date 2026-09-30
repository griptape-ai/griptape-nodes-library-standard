"""Tests for MergeImages' alpha handling and opacity control."""

import numpy as np
import pytest
from PIL import Image

from griptape_nodes_library.image.merge_images import (
    AlphaConvention,
    Background,
    Layout,
    MergeImages,
)


@pytest.fixture
def node() -> MergeImages:
    return MergeImages(name="test_merge_images")


def solid(color: tuple[int, int, int, int], size: tuple[int, int] = (4, 4)) -> Image.Image:
    return Image.new("RGBA", size, color)


def alpha_at(image: Image.Image, xy: tuple[int, int] = (0, 0)) -> int:
    """Read one pixel's alpha, narrowing `getpixel`'s union return type."""
    pixel = image.getpixel(xy)
    assert isinstance(pixel, tuple)
    return pixel[3]


def run(node: MergeImages, images: list[Image.Image], **params: object) -> Image.Image:
    """Drive the node's merge path directly, skipping artifact loading and file writing."""
    alpha_convention = AlphaConvention(params.get("input_alpha", AlphaConvention.STRAIGHT))
    layout = Layout(params.get("layout", Layout.COMPOSITE))
    opacity = float(params.get("opacity", 1.0))  # type: ignore[arg-type]
    background = Background(params.get("background", Background.TRANSPARENT))

    prepared = [node._prepare_image(img, alpha_convention) for img in images]
    merged = node._merge(node._apply_opacity(prepared, opacity, layout), layout)
    return node._apply_background(merged, background)


class TestCompositeAlpha:
    def test_semi_transparent_over_opaque_matches_porter_duff(self, node: MergeImages) -> None:
        """50% red over opaque blue must give the 'over' result, not a double-applied one.

        Pasting with the overlay as its own mask blends toward the overlay's colour and
        then writes the overlay's alpha into the base as well, which landed on
        (160, 64, 159) instead of the correct (128, 0, 127).
        """
        top = solid((255, 0, 0, 128))
        bottom = solid((0, 0, 255, 255))

        result = run(node, [top, bottom])

        assert result.getpixel((0, 0)) == (128, 0, 127, 255)

    def test_opaque_base_stays_opaque(self, node: MergeImages) -> None:
        """A semi-transparent overlay must not punch a hole in an opaque base."""
        result = run(node, [solid((255, 0, 0, 128)), solid((0, 0, 255, 255))])

        assert alpha_at(result, (0, 0)) == 255

    def test_transparent_overlay_leaves_base_untouched(self, node: MergeImages) -> None:
        """A fully transparent overlay contributes nothing, whatever sits in its RGB."""
        result = run(node, [solid((0, 255, 0, 0)), solid((0, 0, 255, 255))])

        assert result.getpixel((0, 0)) == (0, 0, 255, 255)

    def test_first_image_is_on_top(self, node: MergeImages) -> None:
        """The list reads top-down: the first image wins where it is opaque."""
        result = run(node, [solid((255, 0, 0, 255)), solid((0, 0, 255, 255))])

        assert result.getpixel((0, 0)) == (255, 0, 0, 255)

    def test_stacking_is_associative(self, node: MergeImages) -> None:
        """Three semi-transparent layers compose the same as two folded then one."""
        layers = [solid((255, 0, 0, 128)), solid((0, 255, 0, 128)), solid((0, 0, 255, 255))]

        three_at_once = run(node, layers)
        folded = run(node, [layers[0], run(node, layers[1:])])

        assert three_at_once.getpixel((0, 0)) == folded.getpixel((0, 0))


class TestPremultipliedInput:
    def test_premultiplied_input_composites_without_dark_fringe(self, node: MergeImages) -> None:
        """Premultiplied 50% red carries (128,0,0); read as premultiplied it must not darken.

        Treating it as straight alpha composited to (112, 64, 159) — the dark fringe.
        """
        top = solid((128, 0, 0, 128))
        bottom = solid((0, 0, 255, 255))

        result = run(node, [top, bottom], input_alpha=AlphaConvention.PREMULTIPLIED)

        assert result.getpixel((0, 0)) == (128, 0, 127, 255)

    def test_premultiplied_matches_equivalent_straight_input(self, node: MergeImages) -> None:
        """The same edge described either way must composite to the same pixel."""
        bottom = solid((0, 0, 255, 255))

        premultiplied = run(node, [solid((128, 0, 0, 128)), bottom], input_alpha=AlphaConvention.PREMULTIPLIED)
        straight = run(node, [solid((255, 0, 0, 128)), bottom], input_alpha=AlphaConvention.STRAIGHT)

        assert premultiplied.getpixel((0, 0)) == straight.getpixel((0, 0))

    def test_opaque_pixels_are_unchanged_by_the_convention(self, node: MergeImages) -> None:
        """At alpha 255 the two conventions agree, so the setting must be a no-op."""
        images = [solid((200, 100, 50, 255))]

        premultiplied = run(node, images, input_alpha=AlphaConvention.PREMULTIPLIED)
        straight = run(node, images, input_alpha=AlphaConvention.STRAIGHT)

        assert premultiplied.getpixel((0, 0)) == straight.getpixel((0, 0))

    def test_zero_alpha_pixel_survives_unassociation(self, node: MergeImages) -> None:
        """Dividing by a zero alpha has no answer; the pixel stays transparent, not NaN."""
        result = run(node, [solid((0, 0, 0, 0))], input_alpha=AlphaConvention.PREMULTIPLIED)

        assert alpha_at(result, (0, 0)) == 0


class TestTilingLayoutsPreserveAlpha:
    @pytest.mark.parametrize("layout", [Layout.HORIZONTAL, Layout.VERTICAL, Layout.GRID])
    def test_transparent_input_region_stays_transparent(self, node: MergeImages, layout: Layout) -> None:
        """A transparent input region must survive the layout as transparent.

        These layouts built an RGB canvas and pasted without a mask, so alpha was
        discarded outright. The pixel's own RGB is carried through untouched — that is
        what lets a downstream node re-premultiply it — so alpha is the assertion.
        """
        result = run(node, [solid((0, 255, 0, 0)), solid((0, 0, 255, 255))], layout=layout)

        assert result.mode == "RGBA"
        assert alpha_at(result, (0, 0)) == 0

    @pytest.mark.parametrize("layout", [Layout.HORIZONTAL, Layout.VERTICAL, Layout.GRID])
    def test_transparent_region_shows_no_colour_once_flattened(self, node: MergeImages, layout: Layout) -> None:
        """The colour sitting in a transparent region must never become visible.

        A transparent green pixel emitted (0, 255, 0) over a black canvas before; on a
        white background it now has to read as white.
        """
        result = run(
            node,
            [solid((0, 255, 0, 0)), solid((0, 0, 255, 255))],
            layout=layout,
            background=Background.WHITE,
        )

        assert result.getpixel((0, 0)) == (255, 255, 255)

    @pytest.mark.parametrize("layout", [Layout.HORIZONTAL, Layout.VERTICAL, Layout.GRID])
    def test_opaque_input_is_placed_unchanged(self, node: MergeImages, layout: Layout) -> None:
        """Preserving alpha must not disturb ordinary opaque tiling."""
        result = run(node, [solid((255, 0, 0, 255)), solid((0, 0, 255, 255))], layout=layout)

        assert result.getpixel((0, 0)) == (255, 0, 0, 255)

    def test_horizontal_layout_places_images_side_by_side(self, node: MergeImages) -> None:
        result = run(node, [solid((255, 0, 0, 255)), solid((0, 0, 255, 255))], layout=Layout.HORIZONTAL)

        assert result.size == (8, 4)
        assert result.getpixel((5, 0)) == (0, 0, 255, 255)

    def test_vertical_layout_stacks_images(self, node: MergeImages) -> None:
        result = run(node, [solid((255, 0, 0, 255)), solid((0, 0, 255, 255))], layout=Layout.VERTICAL)

        assert result.size == (4, 8)
        assert result.getpixel((0, 5)) == (0, 0, 255, 255)


class TestOpacity:
    def test_full_opacity_changes_nothing(self, node: MergeImages) -> None:
        images = [solid((255, 0, 0, 255)), solid((0, 0, 255, 255))]

        assert run(node, images, opacity=1.0).getpixel((0, 0)) == (255, 0, 0, 255)

    def test_half_opacity_blends_the_upper_image(self, node: MergeImages) -> None:
        """Opaque red at 50% over opaque blue is the same as compositing 128-alpha red."""
        result = run(node, [solid((255, 0, 0, 255)), solid((0, 0, 255, 255))], opacity=0.5)

        assert result.getpixel((0, 0)) == (128, 0, 127, 255)

    def test_zero_opacity_hides_the_upper_image(self, node: MergeImages) -> None:
        result = run(node, [solid((255, 0, 0, 255)), solid((0, 0, 255, 255))], opacity=0.0)

        assert result.getpixel((0, 0)) == (0, 0, 255, 255)

    def test_composite_bottom_image_is_not_faded(self, node: MergeImages) -> None:
        """Fading the backmost layer of a stack would eat into the result's own alpha."""
        result = run(node, [solid((255, 0, 0, 255)), solid((0, 0, 255, 255))], opacity=0.5)

        assert alpha_at(result, (0, 0)) == 255

    def test_opacity_scales_existing_alpha(self, node: MergeImages) -> None:
        """Opacity multiplies the image's own alpha rather than replacing it."""
        result = run(node, [solid((255, 0, 0, 128)), solid((0, 0, 255, 255))], opacity=0.5)

        # 128/255 * 0.5 -> a quarter-strength red over blue.
        assert result.getpixel((0, 0)) == (64, 0, 191, 255)

    def test_tiling_layouts_fade_toward_the_background(self, node: MergeImages) -> None:
        """No layer sits behind a tile, so opacity lowers the tile's own alpha."""
        result = run(node, [solid((255, 0, 0, 255))], layout=Layout.HORIZONTAL, opacity=0.5)

        assert result.getpixel((0, 0)) == (255, 0, 0, 128)


class TestBackground:
    def test_defaults_to_white(self, node: MergeImages) -> None:
        """Existing workflows flattened onto white, so the default keeps that output."""
        assert node.get_parameter_value("background") == Background.WHITE

    def test_transparent_background_keeps_alpha(self, node: MergeImages) -> None:
        result = run(node, [solid((255, 0, 0, 0))], background=Background.TRANSPARENT)

        assert result.mode == "RGBA"
        assert alpha_at(result, (0, 0)) == 0

    @pytest.mark.parametrize(
        ("background", "expected"),
        [(Background.WHITE, (255, 255, 255)), (Background.BLACK, (0, 0, 0))],
    )
    def test_solid_background_flattens_to_rgb(
        self, node: MergeImages, background: Background, expected: tuple[int, int, int]
    ) -> None:
        result = run(node, [solid((255, 0, 0, 0))], background=background)

        assert result.mode == "RGB"
        assert result.getpixel((0, 0)) == expected

    def test_semi_transparent_result_blends_into_the_background(self, node: MergeImages) -> None:
        """Flattening must use 'over' too, so a half-covered pixel is a real blend."""
        result = run(node, [solid((0, 0, 0, 128))], background=Background.WHITE)

        assert result.getpixel((0, 0)) == (127, 127, 127)


class TestUnknownValues:
    def test_unknown_layout_raises(self, node: MergeImages) -> None:
        with pytest.raises(ValueError, match="Unknown layout"):
            node._merge([solid((255, 0, 0, 255))], "diagonal")  # type: ignore[arg-type]

    def test_unknown_alpha_convention_raises(self, node: MergeImages) -> None:
        with pytest.raises(ValueError, match="Unknown alpha convention"):
            node._prepare_image(solid((255, 0, 0, 255)), "associated")  # type: ignore[arg-type]

    def test_unknown_background_raises(self, node: MergeImages) -> None:
        with pytest.raises(ValueError, match="Unknown background"):
            node._apply_background(solid((255, 0, 0, 255)), "chequerboard")  # type: ignore[arg-type]

    def test_composite_with_no_images_raises(self, node: MergeImages) -> None:
        with pytest.raises(ValueError, match="No images provided"):
            node._process_composite_layout([])


class TestResampling:
    def test_transparent_colour_does_not_bleed_into_opaque_edges(self, node: MergeImages) -> None:
        """Scaling a soft edge must not drag colour out of its transparent pixels.

        A straight-alpha resize averages the transparent side's RGB into the opaque
        side; premultiplying around the resample weights each pixel by its own alpha.
        """
        # Left half opaque red, right half transparent green.
        edge = Image.new("RGBA", (8, 2), (0, 255, 0, 0))
        for x in range(4):
            for y in range(2):
                edge.putpixel((x, y), (255, 0, 0, 255))

        # Force a downscale by merging against a smaller base.
        result = run(node, [edge, solid((0, 0, 0, 255), size=(4, 2))])

        # No green may appear anywhere in the composite.
        assert result.getchannel("G").getextrema()[1] == 0

    def test_soft_edge_keeps_its_colour_when_resized(self, node: MergeImages) -> None:
        """Resizing must premultiply exactly once.

        Pillow already premultiplies RGBA while resampling. Doing it again by hand
        weights colour by alpha squared and crushes low-alpha pixels toward black.
        """
        arr = np.zeros((64, 64, 4), dtype=np.uint8)
        arr[..., :3] = (200, 100, 50)
        arr[..., 3] = np.linspace(0, 255, 64).astype(np.uint8)[None, :]
        soft = Image.fromarray(arr, mode="RGBA")

        resized = np.asarray(node._resize_image(soft, 16, 16)).astype(int)

        visible = resized[..., 3] > 0
        red = resized[..., 0][visible]
        assert np.abs(red - 200).max() <= 15  # noqa: PLR2004
