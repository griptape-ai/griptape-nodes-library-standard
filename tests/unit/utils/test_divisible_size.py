"""Tests for snapping dimensions to a multiple of a divisor."""

from __future__ import annotations

import pytest

from griptape_nodes_library.utils.divisible_size import divisible_dimensions, round_to_multiple


@pytest.mark.parametrize(
    ("rounding", "expected"),
    [("round", 192), ("floor", 128), ("ceil", 192)],
)
def test_round_to_multiple(rounding: str, expected: int) -> None:
    # 160 / 64 = 2.5, so round goes up on the tie.
    assert round_to_multiple(160, 64, rounding) == expected


def test_never_rounds_to_zero() -> None:
    assert round_to_multiple(10, 64, "floor") == 64


def test_preserving_aspect_ratio_scales_from_the_larger_side() -> None:
    # 4970 -> 4992 (x1.00443); 3727 x 1.00443 = 3743.5 -> 3712.
    assert divisible_dimensions(4970, 3727, 128, "round", preserve_aspect_ratio=True, anchor_dimension="auto") == (
        4992,
        3712,
    )


def test_auto_anchors_on_height_for_portrait() -> None:
    assert divisible_dimensions(3727, 4970, 128, "round", preserve_aspect_ratio=True, anchor_dimension="auto") == (
        3712,
        4992,
    )


def test_explicit_anchor() -> None:
    # Height 3727 -> 3712 (x0.99598); 4970 x 0.99598 = 4950.0, floored to 4864.
    assert divisible_dimensions(4970, 3727, 128, "floor", preserve_aspect_ratio=True, anchor_dimension="height") == (
        4864,
        3712,
    )


def test_without_aspect_ratio_each_side_rounds_on_its_own() -> None:
    assert divisible_dimensions(4970, 3727, 128, "ceil", preserve_aspect_ratio=False, anchor_dimension="auto") == (
        4992,
        3840,
    )


def test_unknown_rounding_raises() -> None:
    with pytest.raises(ValueError, match="Unknown rounding"):
        round_to_multiple(100, 8, "nearest")
