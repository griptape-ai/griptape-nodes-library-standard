import json
from typing import Any

from griptape_nodes.exe_types.core_types import (
    Parameter,
    ParameterMode,
)
from griptape_nodes.exe_types.node_types import ControlNode
from griptape_nodes.exe_types.param_types.parameter_int import ParameterInt


def flatten(items: list[Any], depth: int) -> list[Any]:
    """Flatten nested lists into one list, keeping item order.

    A depth of 0 flattens every level; a positive depth flattens that many levels.
    Only lists are expanded, so strings, dicts and other values stay whole items.
    """
    flat: list[Any] = []
    for item in items:
        if isinstance(item, list) and depth != 1:
            flat.extend(flatten(item, max(depth - 1, 0)))
        elif isinstance(item, list):
            flat.extend(item)
        else:
            flat.append(item)
    return flat


class FlattenList(ControlNode):
    """FlattenList Node that takes a list of nested lists and returns a single flat list."""

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)
        self.items = Parameter(
            name="items",
            tooltip="List to flatten. It can contain other lists at any depth.",
            input_types=["list", "json"],
            allowed_modes={ParameterMode.INPUT},
        )
        self.add_parameter(self.items)

        self.depth = ParameterInt(
            name="depth",
            tooltip="How many levels of nesting to flatten. 0 flattens every level.",
            default_value=0,
            min_val=0,
            allowed_modes={ParameterMode.PROPERTY, ParameterMode.INPUT},
        )
        self.add_parameter(self.depth)

        self.output = Parameter(
            name="output",
            tooltip="Flat list with every item in its original order",
            output_type="list",
            allowed_modes={ParameterMode.OUTPUT},
        )
        self.add_parameter(self.output)

    def process(self) -> None:
        items = self.get_parameter_value("items")
        depth = max(self.get_parameter_value("depth") or 0, 0)

        self.parameter_output_values["output"] = flatten(self._as_list(items), depth)

    def _as_list(self, value: Any) -> list[Any]:
        """Normalize the input to a list, parsing JSON text and wrapping a single value."""
        if value is None:
            return []
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return [value]
            return parsed if isinstance(parsed, list) else [parsed]
        if isinstance(value, list):
            return value
        return [value]
