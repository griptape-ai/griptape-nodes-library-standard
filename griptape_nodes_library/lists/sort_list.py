import ast
import json
from copy import deepcopy
from typing import Any, NamedTuple

from griptape_nodes.exe_types.core_types import (
    Parameter,
    ParameterMode,
)
from griptape_nodes.exe_types.node_types import ControlNode
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.traits.options import Options
from json_repair import repair_json


class _SortKey(NamedTuple):
    """Sort key: type_order (0=numeric, 1=string) and value for correct ordering."""

    type_order: int
    value: float | str


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

    def process(self) -> None:
        raw_values = self.get_parameter_value("items")
        list_values = self._parse_list_input(raw_values)
        if not list_values:
            self.parameter_output_values["output"] = []
            return

        self._update_key_parameter_visibility()

        sort_order = self.get_parameter_value("sort_order") or "asc"
        reverse = sort_order == "desc"
        key = self.get_parameter_value("key")
        # Strip Options display suffix if present (e.g. "age [42]" or "age (str)" -> "age")
        if key and isinstance(key, str):
            for sep in (" [", " (", " -"):
                if sep in key:
                    key = key.split(sep)[0].strip()
                    break

        list_copy = deepcopy(list_values)

        # Always use a key function - artifacts (ImageUrlArtifact, VideoUrlArtifact, etc.)
        # and other non-comparable types don't support default < comparison
        if key:
            dict_key = key
        else:
            dict_key = None
        sorted_list = sorted(
            list_copy,
            key=lambda x: self._get_sort_key_for_item(x, dict_key),
            reverse=reverse,
        )

        self.parameter_output_values["output"] = sorted_list

    def _update_key_parameter_visibility(self) -> None:
        """Show/hide key parameter and update dropdown based on list content."""
        raw_values = self.get_parameter_value("items")
        list_values = self._parse_list_input(raw_values)
        if not list_values:
            self.hide_parameter_by_name("key")
            return

        if not self._is_list_of_dicts(list_values):
            self.hide_parameter_by_name("key")
            return

        keys = self._get_dict_keys(list_values)
        if not keys:
            self.hide_parameter_by_name("key")
            return

        self.show_parameter_by_name("key")
        current_key = self.get_parameter_value("key")
        if current_key and current_key in keys:
            default_key = current_key
        else:
            default_key = keys[0]
        self._update_option_choices("key", keys, default_key)

    def _parse_dict_like(self, s: str) -> dict[Any, Any] | None:
        """Try to parse a string as a dict (JSON or Python literal syntax)."""
        if not isinstance(s, str) or not s.strip():
            return None
        s = s.strip()
        try:
            parsed = json.loads(s)
            if isinstance(parsed, dict):
                return parsed
            return None
        except (json.JSONDecodeError, TypeError):
            pass
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, dict):
                return parsed
            return None
        except (ValueError, SyntaxError, TypeError):
            pass
        return None

    def _parse_list_input(self, list_values: Any) -> list[Any] | None:
        """Parse and normalize the input to a list. Preserves original item types.
        Handles JSON strings (from connections/serialization) and ensures proper types."""
        if list_values is None:
            return None
        if isinstance(list_values, str):
            try:
                parsed = json.loads(list_values)
                if isinstance(parsed, list):
                    return parsed
            except (json.JSONDecodeError, TypeError):
                pass
            try:
                parsed = repair_json(list_values)
                if isinstance(parsed, list):
                    return parsed
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
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

    def _get_sort_key_for_item(self, item: Any, dict_key: str | None) -> _SortKey:
        """Extract a sortable key for any item. Handles dicts, artifacts, and primitives.
        Returns _SortKey with type_order 0 for numeric values, 1 for strings."""

        def _key_for_val(val: Any) -> _SortKey:
            """Use numeric sort when value is a number (including string '42')."""
            if isinstance(val, (int, float)):
                return _SortKey(0, float(val))
            if isinstance(val, str) and val.strip():
                try:
                    return _SortKey(0, float(val))
                except (ValueError, TypeError):
                    pass
            str_val = str(val) if val is not None else ""
            return _SortKey(1, str_val)

        # Dict with key specified - extract that field
        if dict_key and isinstance(item, dict):
            return _key_for_val(item.get(dict_key, ""))
        if dict_key and isinstance(item, str):
            parsed = self._parse_dict_like(item)
            if parsed:
                return _key_for_val(parsed.get(dict_key, ""))

        # Artifacts (ImageUrlArtifact, VideoUrlArtifact, AudioUrlArtifact, etc.) - sort by value (URL)
        # name is often a UUID and would produce incorrect sort order
        if hasattr(item, "value"):
            val = getattr(item, "value", None)
            if val is not None and str(val).strip():
                return _SortKey(1, str(val))

        # Dict without key - deterministic string for stable sort
        if isinstance(item, dict):
            return _SortKey(1, json.dumps(item, sort_keys=True))

        return _SortKey(1, str(item))
