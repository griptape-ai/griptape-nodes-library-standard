"""Snap image and video dimensions to a multiple of a divisor, as some models require."""

import math
from enum import StrEnum


class Rounding(StrEnum):
    ROUND = "round"
    FLOOR = "floor"
    CEIL = "ceil"


class AnchorDimension(StrEnum):
    AUTO = "auto"
    WIDTH = "width"
    HEIGHT = "height"


def round_to_multiple(value: float, divisor: int, rounding: str) -> int:
    """Round value to a multiple of divisor, never below divisor itself.

    Halfway values round up, as FFmpeg's expressions and the editor preview round them.
    """
    quotient = value / divisor
    match rounding:
        case Rounding.ROUND:
            steps = math.floor(quotient + 0.5)
        case Rounding.FLOOR:
            steps = math.floor(quotient)
        case Rounding.CEIL:
            steps = math.ceil(quotient)
        case _:
            msg = f"Unknown rounding: {rounding!r}"
            raise ValueError(msg)
    return max(1, steps) * divisor


def divisible_dimensions(  # noqa: PLR0913
    width: int,
    height: int,
    divisor: int,
    rounding: str,
    *,
    preserve_aspect_ratio: bool,
    anchor_dimension: str,
) -> tuple[int, int]:
    """Return (width, height) with both sides a multiple of divisor.

    Without preserve_aspect_ratio each side is rounded on its own. With it, the anchor side is rounded,
    the other side is scaled by the anchor's exact scale, and that result is rounded.
    """
    if not preserve_aspect_ratio:
        return round_to_multiple(width, divisor, rounding), round_to_multiple(height, divisor, rounding)

    match anchor_dimension:
        case AnchorDimension.AUTO:
            anchor_is_width = width >= height
        case AnchorDimension.WIDTH:
            anchor_is_width = True
        case AnchorDimension.HEIGHT:
            anchor_is_width = False
        case _:
            msg = f"Unknown anchor dimension: {anchor_dimension!r}"
            raise ValueError(msg)

    if anchor_is_width:
        new_width = round_to_multiple(width, divisor, rounding)
        return new_width, round_to_multiple(height * new_width / width, divisor, rounding)
    new_height = round_to_multiple(height, divisor, rounding)
    return round_to_multiple(width * new_height / height, divisor, rounding), new_height
