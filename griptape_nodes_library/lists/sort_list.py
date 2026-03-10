from copy import deepcopy
from typing import Any

from griptape_nodes.exe_types.core_types import (
    Parameter,
    ParameterMode,
)
from griptape_nodes.exe_types.node_types import ControlNode
from griptape_nodes.traits.options import Options


class SortList(ControlNode):
    """SortList Node that takes a list and sorts it in ascending or descending order."""

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)
        self.items = Parameter(
            name="items",
            tooltip="List of items to sort",
            input_types=["list"],
            allowed_modes={ParameterMode.INPUT},
        )
        self.add_parameter(self.items)

        self.sort_order = Parameter(
            name="sort_order",
            tooltip="Sort order for the list",
            input_types=["str"],
            allowed_modes={ParameterMode.PROPERTY},
            default_value="asc",
        )
        self.add_parameter(self.sort_order)
        self.sort_order.add_trait(Options(choices=["asc", "desc"]))

        self.output = Parameter(
            name="output",
            tooltip="Sorted list",
            output_type="list",
            allowed_modes={ParameterMode.OUTPUT},
        )
        self.add_parameter(self.output)

    def process(self) -> None:
        list_values = self.get_parameter_value("items")
        if not list_values or not isinstance(list_values, list):
            return

        sort_order = self.get_parameter_value("sort_order") or "asc"
        reverse = sort_order == "desc"

        sorted_list = sorted(deepcopy(list_values), reverse=reverse)
        self.parameter_output_values["output"] = sorted_list
