import re
from collections.abc import Iterable
from typing import Any

from griptape_nodes.exe_types.core_types import ParameterMode
from griptape_nodes.exe_types.node_types import DataNode
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString

# Matches width:height with optional decimals and optional spaces around ':'.
_ASPECT_RATIO_TOKEN_PATTERN = re.compile(r"\d+(?:\.\d+)?\s*:\s*\d+(?:\.\d+)?")


def closest_aspect_ratio(target: str, aspect_ratios: list[str]) -> str:
    def ratio_to_float(ratio: str) -> float:
        w, h = map(float, ratio.split(":"))
        return w / h

    target_value = ratio_to_float(target)
    closest = min(aspect_ratios, key=lambda r: abs(ratio_to_float(r) - target_value))
    return closest


def _is_valid_aspect_ratio_string(value: str) -> bool:
    try:
        parts = value.split(":")
        if len(parts) != 2:
            return False
        _, height = map(float, parts)
        if height == 0:
            return False
    except ValueError:
        return False
    return True


def _normalized_valid_target(aspect_ratio_value: Any) -> str | None:
    if aspect_ratio_value is None:
        return None
    if not isinstance(aspect_ratio_value, str):
        return None
    target = aspect_ratio_value.strip()
    if not target:
        return None
    if not _is_valid_aspect_ratio_string(target):
        return None
    return target


def _valid_aspect_ratio_candidates(raw_items: Iterable[Any]) -> list[str]:
    result: list[str] = []
    for item in raw_items:
        if not isinstance(item, str):
            continue
        candidate = item.strip()
        if not candidate:
            continue
        if not _is_valid_aspect_ratio_string(candidate):
            continue
        result.append(candidate)
    return result


def _aspect_ratio_list_from_value(raw: Any) -> list[str]:
    """Build candidate aspect ratios from a string (comma/newline/space separated) or an iterable of strings."""
    if raw is None:
        return []
    if isinstance(raw, str):
        items = _ASPECT_RATIO_TOKEN_PATTERN.findall(raw)
        return _valid_aspect_ratio_candidates(items)
    if isinstance(raw, Iterable):
        return _valid_aspect_ratio_candidates(raw)
    return []


class GetClosestAspectRatio(DataNode):
    """Pick the candidate aspect ratio whose numeric value is nearest to the target."""

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)

        self.add_parameter(
            ParameterString(
                name="aspect_ratio",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                default_value="16:9",
                tooltip="Target aspect ratio as width:height (e.g. 16:9)",
                placeholder_text="16:9",
            )
        )

        self.add_parameter(
            ParameterString(
                name="aspect_ratios",
                default_value="1:1, 3:2, 2:3, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9",
                tooltip=(
                    "Candidate aspect ratios separated by commas, newlines, or spaces (e.g. 1:1 3:2 or 1:1, 2:3)"
                ),
            )
        )

        self.add_parameter(
            ParameterString(
                name="closest_aspect_ratio",
                allowed_modes={ParameterMode.OUTPUT},
                default_value="",
                placeholder_text="The closest aspect ratio",
                tooltip="The candidate closest to the target by absolute difference in width/height ratio",
            )
        )

    def process(self) -> None:
        target = _normalized_valid_target(self.get_parameter_value("aspect_ratio"))
        if target is None:
            self.parameter_output_values["closest_aspect_ratio"] = ""
            return

        candidates = _aspect_ratio_list_from_value(self.get_parameter_value("aspect_ratios"))
        if not candidates:
            self.parameter_output_values["closest_aspect_ratio"] = ""
            return

        self.parameter_output_values["closest_aspect_ratio"] = closest_aspect_ratio(target, candidates)
