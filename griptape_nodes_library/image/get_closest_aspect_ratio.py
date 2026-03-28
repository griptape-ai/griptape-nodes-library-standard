import re
from collections.abc import Iterable
from typing import Any

from griptape_nodes.exe_types.core_types import ParameterMode
from griptape_nodes.exe_types.node_types import SuccessFailureNode
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


def _require_valid_aspect_ratio_token(token: str, *, in_list: bool) -> str:
    candidate = token.strip()
    ctx = "aspect_ratios" if in_list else "aspect_ratio"
    if not candidate:
        raise ValueError(f"Empty aspect ratio in {ctx}")
    if not _is_valid_aspect_ratio_string(candidate):
        raise ValueError(f"Invalid aspect ratio in {ctx}: {candidate!r} (expected width:height, e.g. 16:9)")
    return candidate


def _parse_target_aspect_ratio(aspect_ratio_value: Any) -> str:
    if aspect_ratio_value is None:
        raise ValueError("aspect_ratio is required")
    if not isinstance(aspect_ratio_value, str):
        msg = f"aspect_ratio must be a string, got {type(aspect_ratio_value).__name__}"
        raise ValueError(msg)
    return _require_valid_aspect_ratio_token(aspect_ratio_value, in_list=False)


def _parse_candidate_aspect_ratios_string(text: str) -> list[str]:
    if not text.strip():
        raise ValueError("aspect_ratios is empty")
    matches = _ASPECT_RATIO_TOKEN_PATTERN.findall(text)
    if not matches:
        raise ValueError(f"No valid aspect ratio tokens in aspect_ratios: {text!r}")
    depleted = _ASPECT_RATIO_TOKEN_PATTERN.sub("", text)
    junk = re.sub(r"[\s,;]+", "", depleted)
    if junk:
        raise ValueError(f"Invalid text in aspect_ratios: {junk!r}")
    return [_require_valid_aspect_ratio_token(m, in_list=True) for m in matches]


def _parse_candidate_aspect_ratios_iterable(items: Iterable[Any]) -> list[str]:
    out: list[str] = []
    for i, item in enumerate(items):
        if item is None or (isinstance(item, str) and not item.strip()):
            continue
        if not isinstance(item, str):
            msg = f"aspect_ratios item at index {i} must be a string, got {type(item).__name__}"
            raise ValueError(msg)
        out.append(_require_valid_aspect_ratio_token(item, in_list=True))
    if not out:
        raise ValueError("aspect_ratios contains no valid entries")
    return out


def _parse_aspect_ratio_candidates(raw: Any) -> list[str]:
    """Parse aspect_ratios from a delimiter string or an iterable; raises ValueError if invalid."""
    if raw is None:
        raise ValueError("aspect_ratios is required")
    if isinstance(raw, str):
        return _parse_candidate_aspect_ratios_string(raw)
    if isinstance(raw, Iterable):
        return _parse_candidate_aspect_ratios_iterable(raw)
    msg = f"aspect_ratios must be a string or iterable, got {type(raw).__name__}"
    raise ValueError(msg)


class GetClosestAspectRatio(SuccessFailureNode):
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

        self._create_status_parameters(
            result_details_tooltip="Details about the closest aspect ratio result",
            result_details_placeholder="Details on the comparison will be presented here.",
            parameter_group_initially_collapsed=True,
        )

    def process(self) -> None:
        self._clear_execution_status()

        try:
            target = _parse_target_aspect_ratio(self.get_parameter_value("aspect_ratio"))
            candidates = _parse_aspect_ratio_candidates(self.get_parameter_value("aspect_ratios"))
            chosen = closest_aspect_ratio(target, candidates)
        except ValueError as e:
            self.set_parameter_value("closest_aspect_ratio", "")
            self.parameter_output_values["closest_aspect_ratio"] = ""
            self._set_status_results(was_successful=False, result_details=f"FAILURE: {e}")
            self._handle_failure_exception(e)
            return

        self.set_parameter_value("closest_aspect_ratio", chosen)
        self.parameter_output_values["closest_aspect_ratio"] = chosen
        self._set_status_results(
            was_successful=True,
            result_details=f"SUCCESS: Closest aspect ratio to {target} is {chosen}",
        )
