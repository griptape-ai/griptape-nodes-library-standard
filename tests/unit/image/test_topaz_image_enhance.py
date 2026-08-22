from __future__ import annotations

import pytest

from griptape_nodes_library.image.topaz_image_enhance import (
    MAX_OUTPUT_DIMENSION,
    ResizeMode,
    TopazImageEnhance,
)

RESIZE_KEYS = ("output_width", "output_height", "crop_to_fill")


def _node(name: str = "TopazImageEnhance") -> TopazImageEnhance:
    return TopazImageEnhance(name=name)


def _stub_image(node: TopazImageEnhance, monkeypatch: pytest.MonkeyPatch) -> None:
    """Stand in for reading the source image and base64-encoding it.

    The real call reaches static storage; the payload tests only care about the
    resize keys sitting next to it.
    """
    node.set_parameter_value("image_input", "https://example.com/source.png")
    monkeypatch.setattr(
        TopazImageEnhance,
        "_process_input_image",
        _async_return("data:image/png;base64,AAAA"),
    )


def _async_return(value: object):  # noqa: ANN202
    async def _stub(*_args: object, **_kwargs: object) -> object:
        return value

    return _stub


def _stub_source_dimensions(monkeypatch: pytest.MonkeyPatch, width: int, height: int) -> None:
    monkeypatch.setattr(
        "griptape_nodes_library.image.topaz_image_enhance.get_image_dimensions_from_artifact",
        lambda _artifact: (width, height),
    )


# -- backward compatibility --------------------------------------------------


@pytest.mark.asyncio
async def test_default_payload_carries_no_resize_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    # The regression test for every workflow saved before resize existed: an
    # untouched node must send exactly what it sent before.
    node = _node()
    _stub_image(node, monkeypatch)

    payload = await node._build_payload()

    for key in RESIZE_KEYS:
        assert key not in payload


@pytest.mark.asyncio
async def test_resize_mode_none_carries_no_resize_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    node = _node()
    _stub_image(node, monkeypatch)
    node.set_parameter_value("resize_mode", ResizeMode.NONE)
    # Stale dimensions left over from a previous mode must not leak into the payload.
    node.set_parameter_value("output_width", 4096)
    node.set_parameter_value("output_height", 4096)

    payload = await node._build_payload()

    for key in RESIZE_KEYS:
        assert key not in payload


def test_defaults_are_none_so_the_payload_skips_them() -> None:
    # _build_payload skips a value only when it is None; a 0 default would be sent
    # and rejected by Topaz's 1-32000 range.
    node = _node()

    assert node.get_parameter_value("output_width") is None
    assert node.get_parameter_value("output_height") is None


# -- payload -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_width_and_height_emits_both_dimensions(monkeypatch: pytest.MonkeyPatch) -> None:
    node = _node()
    _stub_image(node, monkeypatch)
    node.set_parameter_value("resize_mode", ResizeMode.WIDTH_HEIGHT)
    node.set_parameter_value("output_width", 3840)
    node.set_parameter_value("output_height", 2160)

    payload = await node._build_payload()

    assert payload["output_width"] == 3840
    assert payload["output_height"] == 2160
    assert payload["crop_to_fill"] is False


@pytest.mark.asyncio
async def test_width_only_derives_height_from_the_source_aspect(monkeypatch: pytest.MonkeyPatch) -> None:
    # Topaz would scale the missing dimension itself, but then the output pixel
    # count is unknowable at request time and the proxy bills at request time.
    node = _node()
    _stub_image(node, monkeypatch)
    _stub_source_dimensions(monkeypatch, 960, 540)
    node.set_parameter_value("resize_mode", ResizeMode.WIDTH)
    node.set_parameter_value("output_width", 3840)
    node.set_parameter_value("output_height", 9999)

    payload = await node._build_payload()

    assert payload["output_width"] == 3840
    assert payload["output_height"] == 2160


@pytest.mark.asyncio
async def test_height_only_derives_width_from_the_source_aspect(monkeypatch: pytest.MonkeyPatch) -> None:
    node = _node()
    _stub_image(node, monkeypatch)
    _stub_source_dimensions(monkeypatch, 960, 540)
    node.set_parameter_value("resize_mode", ResizeMode.HEIGHT)
    node.set_parameter_value("output_height", 2160)
    node.set_parameter_value("output_width", 9999)

    payload = await node._build_payload()

    assert payload["output_width"] == 3840
    assert payload["output_height"] == 2160


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [ResizeMode.WIDTH, ResizeMode.HEIGHT, ResizeMode.PERCENTAGE])
async def test_every_resize_mode_emits_both_dimensions(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    # The proxy bills from width x height and rejects a lone dimension, so no mode
    # may emit only one.
    node = _node()
    _stub_image(node, monkeypatch)
    _stub_source_dimensions(monkeypatch, 1000, 750)
    node.set_parameter_value("resize_mode", mode)
    node.set_parameter_value("output_width", 2000)
    node.set_parameter_value("output_height", 1500)

    payload = await node._build_payload()

    assert payload["output_width"] > 0
    assert payload["output_height"] > 0


@pytest.mark.asyncio
async def test_percentage_scales_the_source(monkeypatch: pytest.MonkeyPatch) -> None:
    node = _node()
    _stub_image(node, monkeypatch)
    _stub_source_dimensions(monkeypatch, 960, 540)
    node.set_parameter_value("resize_mode", ResizeMode.PERCENTAGE)
    node.set_parameter_value("percentage", 200)

    payload = await node._build_payload()

    assert payload["output_width"] == 1920
    assert payload["output_height"] == 1080


@pytest.mark.asyncio
async def test_percentage_below_100_downscales(monkeypatch: pytest.MonkeyPatch) -> None:
    node = _node()
    _stub_image(node, monkeypatch)
    _stub_source_dimensions(monkeypatch, 1000, 500)
    node.set_parameter_value("resize_mode", ResizeMode.PERCENTAGE)
    node.set_parameter_value("percentage", 50)

    payload = await node._build_payload()

    assert payload["output_width"] == 500
    assert payload["output_height"] == 250


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [ResizeMode.WIDTH, ResizeMode.HEIGHT, ResizeMode.PERCENTAGE])
async def test_source_dependent_modes_raise_when_the_source_cannot_be_read(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    # get_image_dimensions_from_artifact returns (0, 0) on failure. Silently skipping
    # the resize would ship an un-upscaled image that looks like a success.
    node = _node()
    _stub_image(node, monkeypatch)
    _stub_source_dimensions(monkeypatch, 0, 0)
    node.set_parameter_value("resize_mode", mode)
    node.set_parameter_value("output_width", 2000)
    node.set_parameter_value("output_height", 1500)

    with pytest.raises(ValueError, match="could not read the source image"):
        await node._build_payload()


@pytest.mark.asyncio
async def test_width_and_height_does_not_need_the_source(monkeypatch: pytest.MonkeyPatch) -> None:
    # The escape hatch the other modes' error message points at.
    node = _node()
    _stub_image(node, monkeypatch)
    _stub_source_dimensions(monkeypatch, 0, 0)
    node.set_parameter_value("resize_mode", ResizeMode.WIDTH_HEIGHT)
    node.set_parameter_value("output_width", 2000)
    node.set_parameter_value("output_height", 1500)

    payload = await node._build_payload()

    assert (payload["output_width"], payload["output_height"]) == (2000, 1500)


@pytest.mark.asyncio
async def test_crop_to_fill_is_emitted_when_true(monkeypatch: pytest.MonkeyPatch) -> None:
    node = _node()
    _stub_image(node, monkeypatch)
    node.set_parameter_value("resize_mode", ResizeMode.WIDTH_HEIGHT)
    node.set_parameter_value("output_width", 1000)
    node.set_parameter_value("output_height", 1000)
    node.set_parameter_value("crop_to_fill", True)

    payload = await node._build_payload()

    assert payload["crop_to_fill"] is True


@pytest.mark.asyncio
async def test_crop_to_fill_is_not_emitted_without_a_dimension(monkeypatch: pytest.MonkeyPatch) -> None:
    node = _node()
    _stub_image(node, monkeypatch)
    node.set_parameter_value("crop_to_fill", True)

    payload = await node._build_payload()

    assert "crop_to_fill" not in payload


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["denoise", "sharpen", "lighting", "matting"])
async def test_non_resizable_operations_never_emit_resize_keys(monkeypatch: pytest.MonkeyPatch, operation: str) -> None:
    # Topaz silently ignores unknown fields, so sending them here would bill for
    # pixels the endpoint never produces.
    node = _node()
    _stub_image(node, monkeypatch)
    node.set_parameter_value("operation", operation)
    node.set_parameter_value("output_width", 4096)
    node.set_parameter_value("output_height", 4096)

    payload = await node._build_payload()

    for key in RESIZE_KEYS:
        assert key not in payload


# -- validation --------------------------------------------------------------


def test_default_node_validates_clean() -> None:
    node = _node()

    assert node.validate_before_node_run() is None


def _force_values(node: TopazImageEnhance, monkeypatch: pytest.MonkeyPatch, **values: object) -> None:
    """Bypass the parameters' own validation to reach the node's guards.

    `min_val`/`max_val` clamp an out-of-range dimension and the `Options` trait
    rejects an off-list operation, so `set_parameter_value` can never deliver
    either. The guards still exist because a workflow saved against an older build
    loads with `initial_setup=True`, which skips every hook; this is how they get
    exercised.
    """
    original = node.get_parameter_value
    monkeypatch.setattr(
        node,
        "get_parameter_value",
        lambda name, *a, **kw: values.get(name, original(name, *a, **kw)),
    )


@pytest.mark.parametrize("value", [0, -1, MAX_OUTPUT_DIMENSION + 1])
def test_out_of_range_dimensions_are_rejected(monkeypatch: pytest.MonkeyPatch, value: int) -> None:
    node = _node()
    _force_values(
        node,
        monkeypatch,
        resize_mode=ResizeMode.WIDTH_HEIGHT,
        output_width=value,
        output_height=1024,
    )

    errors = node.validate_before_node_run() or []

    assert any("output_width must be between" in str(error) for error in errors)


def test_missing_dimension_is_rejected() -> None:
    node = _node()
    node.set_parameter_value("resize_mode", ResizeMode.WIDTH_HEIGHT)
    node.set_parameter_value("output_width", 1024)

    errors = node.validate_before_node_run() or []

    assert any("output_height is required" in str(error) for error in errors)


def test_resize_on_a_non_resizable_operation_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    # after_value_set does not run during workflow load, so a stale combination can
    # reach run time. It must fail loudly rather than silently drop the resize.
    node = _node()
    _force_values(
        node,
        monkeypatch,
        operation="denoise",
        resize_mode=ResizeMode.WIDTH_HEIGHT,
        output_width=1024,
        output_height=1024,
    )

    errors = node.validate_before_node_run() or []

    assert any("cannot resize" in str(error) for error in errors)


def test_unknown_resize_mode_raises_rather_than_defaulting() -> None:
    # The wildcard case must surface an unexpected value, not silently emit nothing.
    node = _node()

    with pytest.raises(ValueError, match="Unknown resize mode"):
        node._resolve_output_size({"resize_mode": "4x"})


def test_boolean_dimensions_are_rejected() -> None:
    # True is 1 in Python; the int check has to reject bools explicitly.
    node = _node()

    with pytest.raises(ValueError, match="must be an integer"):
        node._validated_dimension(True, "output_width")  # noqa: FBT003


# -- credit meter ------------------------------------------------------------

# Topaz's published tables, reproduced row for row. These prove the formula
# rather than restating it.
# https://developer.topazlabs.com/image-models/gigapixel/standard-2.md
GIGAPIXEL_TABLE = [(1, 1), (4, 1), (8, 1), (16, 1), (24, 1), (32, 2), (40, 2), (50, 3), (64, 3), (100, 5)]
# https://developer.topazlabs.com/image-models/wonder/standard-max.md
WONDER_TABLE = [(1, 1), (4, 1), (8, 2), (16, 4), (24, 6), (32, 8), (40, 10), (50, 13), (64, 16), (100, 25)]


@pytest.mark.parametrize(("megapixels", "expected"), GIGAPIXEL_TABLE)
def test_gigapixel_credits_match_the_published_table(megapixels: int, expected: int) -> None:
    assert TopazImageEnhance._credits_for_pixels(megapixels * 1_000_000, 24) == expected


@pytest.mark.parametrize(("megapixels", "expected"), WONDER_TABLE)
def test_wonder_credits_match_the_published_table(megapixels: int, expected: int) -> None:
    assert TopazImageEnhance._credits_for_pixels(megapixels * 1_000_000, 4) == expected


def test_credits_round_up_past_the_boundary() -> None:
    # The case per-megapixel billing gets wrong: Topaz rounds credits, not megapixels.
    assert TopazImageEnhance._credits_for_pixels(24 * 1_000_000, 24) == 1
    assert TopazImageEnhance._credits_for_pixels(25 * 1_000_000, 24) == 2


def test_credits_never_fall_below_one() -> None:
    assert TopazImageEnhance._credits_for_pixels(1, 24) == 1
