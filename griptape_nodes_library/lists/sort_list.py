import ast
import json
from copy import deepcopy
from typing import Any

from griptape_nodes.exe_types.core_types import (
    Parameter,
    ParameterMode,
)
from griptape_nodes.exe_types.node_types import ControlNode
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.traits.options import Options


class SortList(ControlNode):
    """SortList Node that takes a list and sorts it in ascending or descending order.

    When the list contains dictionaries, a Key parameter appears allowing you to
    select which key to sort by.
    """

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)
        self.items = Parameter(
            name="items",
            tooltip="List of items to sort",
            input_types=["list", "json"],
            allowed_modes={ParameterMode.INPUT},
        )
        self.add_parameter(self.items)

        self.sort_order = ParameterString(
            name="sort_order",
            tooltip="Sort order for the list",
            default_value="asc",
            allowed_modes={ParameterMode.PROPERTY},
            traits={Options(choices=["asc", "desc"])},
        )
        self.add_parameter(self.sort_order)

        self.key = ParameterString(
            name="key",
            tooltip="Key to sort by (when list contains dictionaries)",
            default_value="",
            allowed_modes={ParameterMode.PROPERTY},
            traits={Options(choices=[])},
        )
        self.add_parameter(self.key)
        self.hide_parameter_by_name("key")

        self.output = Parameter(
            name="output",
            tooltip="Sorted list",
            output_type="list",
            allowed_modes={ParameterMode.OUTPUT},
        )
        self.add_parameter(self.output)

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        if parameter.name == "items":
            self._update_key_parameter_visibility()
        return super().after_value_set(parameter, value)

    def _parse_dict_like(self, s: str) -> dict[Any, Any] | None:
        """Try to parse a string as a dict (JSON or Python literal syntax)."""
        if not isinstance(s, str) or not s.strip():
            return None
        s = s.strip()
        # Try JSON first (double quotes)
        try:
            parsed = json.loads(s)
            return parsed if isinstance(parsed, dict) else None
        except (json.JSONDecodeError, TypeError):
            pass
        # Try ast.literal_eval for Python literal syntax (single quotes)
        try:
            parsed = ast.literal_eval(s)
            return parsed if isinstance(parsed, dict) else None
        except (ValueError, SyntaxError, TypeError):
            pass
        return None

    def _parse_list_input(self, list_values: Any) -> list[Any] | None:
        """Parse and normalize the input to a list. Preserves original item types."""
        if list_values is None:
            return None
        if isinstance(list_values, str):
            try:
                parsed = json.loads(list_values)
                return parsed if isinstance(parsed, list) else None
            except (json.JSONDecodeError, TypeError):
                return None
        if isinstance(list_values, list):
            return list_values
        return None

    def _is_list_of_dicts(self, list_values: list[Any]) -> bool:
        """Check if the list contains only dictionaries (or string representations of dicts)."""
        if not list_values:
            return False
        for item in list_values:
            if isinstance(item, dict):
                continue
            if isinstance(item, str) and self._parse_dict_like(item) is not None:
                continue
            return False
        return True

    def _get_dict_keys(self, list_values: list[Any]) -> list[str]:
        """Get union of all keys from dictionaries in the list."""
        keys: set[str] = set()
        for item in list_values:
            if isinstance(item, dict):
                keys.update(item.keys())
            elif isinstance(item, str):
                parsed = self._parse_dict_like(item)
                if parsed:
                    keys.update(parsed.keys())
        return sorted(keys)

    def _get_sort_key_for_item(self, item: Any, key: str) -> str:
        """Extract sort key from item (dict or string representation of dict)."""
        if isinstance(item, dict):
            return str(item.get(key, ""))
        if isinstance(item, str):
            parsed = self._parse_dict_like(item)
            if parsed:
                return str(parsed.get(key, ""))
        return str(item)

    def _update_key_parameter_visibility(self) -> None:
        """Show/hide key parameter and update dropdown based on list content."""
        raw_values = self.get_parameter_value("items")
        list_values = self._parse_list_input(raw_values)
        if not list_values:
            self.hide_parameter_by_name("key")
            return

        if self._is_list_of_dicts(list_values):
            keys = self._get_dict_keys(list_values)
            if keys:
                self.show_parameter_by_name("key")
                current_key = self.get_parameter_value("key")
                default_key = (
                    current_key if current_key and current_key in keys else keys[0]
                )
                self._update_option_choices("key", keys, default_key)
            else:
                self.hide_parameter_by_name("key")
        else:
            self.hide_parameter_by_name("key")

    def process(self) -> None:
        raw_values = self.get_parameter_value("items")
        list_values = self._parse_list_input(raw_values)
        if not list_values:
            return

        self._update_key_parameter_visibility()

        sort_order = self.get_parameter_value("sort_order") or "asc"
        reverse = sort_order == "desc"
        key = self.get_parameter_value("key")

        list_copy = deepcopy(list_values)

        if self._is_list_of_dicts(list_copy) and key:
            sorted_list = sorted(
                list_copy,
                key=lambda x: self._get_sort_key_for_item(x, key),
                reverse=reverse,
            )
        else:
            sorted_list = sorted(list_copy, reverse=reverse)

        self.parameter_output_values["output"] = sorted_list
