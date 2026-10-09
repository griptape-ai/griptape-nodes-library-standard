import json
from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import DataNode
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString


class GroupByKey(DataNode):
    """Group a list of dictionaries into a dictionary of lists.

    With Group By empty, every key/value pair is collected under its key:
    [{"a": 1}, {"b": 2}, {"a": 3}] becomes {"a": [1, 3], "b": [2]}.
    With Group By set, each dictionary is collected whole under the value of that field.
    Groups keep the order their keys first appear in.
    """

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)

        self.add_parameter(
            Parameter(
                name="items",
                input_types=["list", "dict"],
                type="list",
                default_value=[],
                allowed_modes={ParameterMode.INPUT},
                tooltip="A list of dictionaries, such as the results of a For Each loop.",
            )
        )

        self.add_parameter(
            ParameterString(
                name="group_by",
                default_value="",
                placeholder_text="Leave empty to group every key/value pair",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                tooltip="Optional. A field name to group whole dictionaries by, such as 'shot'. "
                "Leave empty to collect each key/value pair under its own key.",
            )
        )

        self.add_parameter(
            Parameter(
                name="output",
                output_type="dict",
                default_value={},
                allowed_modes={ParameterMode.OUTPUT},
                tooltip="A dictionary mapping each key to the list of values collected under it.",
            )
        )

    @staticmethod
    def _decode_json(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    def _group(self) -> dict[str, list[Any]]:
        raw_items = self.get_parameter_value("items") or []
        items = self._decode_json(raw_items)
        if isinstance(items, dict):
            items = [items]
        if not isinstance(items, list):
            msg = f"{self.name}: items is a {type(items).__name__}, not a list of dictionaries."
            raise TypeError(msg)
        field = (self.get_parameter_value("group_by") or "").strip()

        groups: dict[str, list[Any]] = {}
        for index, raw_item in enumerate(items):
            item = self._decode_json(raw_item)
            if not isinstance(item, dict):
                msg = f"{self.name}: item {index} is a {type(item).__name__}, not a dictionary."
                raise TypeError(msg)
            if not field:
                for key, value in item.items():
                    groups.setdefault(str(key), []).append(value)
                continue
            if field not in item:
                msg = f"{self.name}: item {index} has no '{field}' field to group by."
                raise KeyError(msg)
            groups.setdefault(str(item[field]), []).append(item)
        return groups

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        # Values from upstream can arrive after this node resolves, so regroup on every change.
        # Bad input is reported by process() when the node runs, not while wiring.
        if parameter.name in ("items", "group_by"):
            try:
                self.parameter_output_values["output"] = self._group()
            except (TypeError, KeyError):
                self.parameter_output_values["output"] = {}
        return super().after_value_set(parameter, value)

    def process(self) -> None:
        self.parameter_output_values["output"] = self._group()
