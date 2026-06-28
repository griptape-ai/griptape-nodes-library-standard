import copy
import re
from io import StringIO
from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import ControlNode
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.exe_types.param_types.parameter_yaml import ParameterYaml
from ruamel.yaml import YAML

_yaml = YAML()
_yaml.preserve_quotes = True


class YamlReplace(ControlNode):
    """Replace a value in YAML using dot notation path."""

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)

        self.add_parameter(
            ParameterYaml(
                name="yaml",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                default_value="",
                tooltip="Input YAML data to modify",
            )
        )

        self.add_parameter(
            ParameterString(
                name="path",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                default_value="",
                tooltip="Dot notation path to replace (e.g., 'user.name', 'items[0].title')",
                placeholder_text="ex: user.name, items[0].title",
            )
        )

        self.add_parameter(
            ParameterYaml(
                name="replacement_value",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                default_value="",
                tooltip="The new value to put at the specified path (YAML scalar, mapping, or sequence)",
            )
        )

        self.add_parameter(
            ParameterYaml(
                name="output",
                tooltip="The modified YAML with the replacement applied",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

    def _parse_array_index(self, part: str) -> tuple[str | None, int | None]:
        match = re.match(r"^(.+)\[(\d+)\]$", part)
        return (match.group(1), int(match.group(2))) if match else (None, None)

    def _ensure_dict_has_list(self, current: Any, key: str) -> Any:
        if isinstance(current, dict):
            if key not in current:
                current[key] = []
            return current[key]
        return current

    def _ensure_list_has_index(self, current: Any, index: int) -> Any:
        if isinstance(current, list):
            while len(current) <= index:
                current.append({} if index < len(current) else None)
            return current[index]
        return None

    def _handle_array_part(self, current: Any, key: str, index: int) -> Any:
        current = self._ensure_dict_has_list(current, key)
        return self._ensure_list_has_index(current, index)

    def _split_path_into_parts(self, path: str) -> list[str]:
        return re.split(r"\.(?![^\[]*\])", path)

    def _is_valid_container(self, obj: Any) -> bool:
        return isinstance(obj, (dict, list))

    def _handle_path_part(self, current: Any, part: str) -> Any:
        key, index = self._parse_array_index(part)

        if key and index is not None:
            current = self._handle_array_part(current, key, index)
            if current is None:
                return None
        elif isinstance(current, dict):
            if part not in current:
                current[part] = {}
            current = current[part]
        else:
            return None

        return current

    def _navigate_to_parent_container(self, data: Any, path_parts: list[str]) -> tuple[Any, str]:
        current = data
        for part in path_parts[:-1]:
            if not self._is_valid_container(current):
                return None, ""
            current = self._handle_path_part(current, part)
            if current is None:
                return None, ""
        return current, path_parts[-1]

    def _set_value_in_container(self, container: Any, final_part: str, new_value: Any) -> None:
        key, index = self._parse_array_index(final_part)
        if key and index is not None:
            container = self._handle_array_part(container, key, index)
            if isinstance(container, list):
                container[index] = new_value
        elif isinstance(container, dict):
            container[final_part] = new_value

    def _set_value_at_path(self, data: Any, path: str, new_value: Any) -> Any:
        if not path:
            return new_value
        result = copy.deepcopy(data)
        path_parts = self._split_path_into_parts(path)
        parent_container, final_part = self._navigate_to_parent_container(result, path_parts)
        if parent_container is None:
            return result
        self._set_value_in_container(parent_container, final_part, new_value)
        return result

    def _to_yaml_string(self, data: Any) -> str:
        stream = StringIO()
        _yaml.dump(data, stream)
        return stream.getvalue()

    def _perform_replacement(self) -> None:
        yaml_str = self.get_parameter_value("yaml")
        path = self.get_parameter_value("path")
        replacement_str = self.get_parameter_value("replacement_value")

        data = _yaml.load(yaml_str) if yaml_str else {}
        replacement = _yaml.load(replacement_str) if replacement_str else None

        result_data = self._set_value_at_path(data, path, replacement)
        result_str = self._to_yaml_string(result_data)

        self.set_parameter_value("output", result_str)
        self.publish_update_to_parameter("output", result_str)

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        if parameter.name in ["yaml", "path", "replacement_value"]:
            self._perform_replacement()
        return super().after_value_set(parameter, value)

    def process(self) -> None:
        self._perform_replacement()
