import logging
from io import StringIO
from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import ControlNode
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.exe_types.param_types.parameter_yaml import ParameterYaml
from ruamel.yaml import YAML

from griptape_nodes_library.utils.dot_path import set_value_at_path

logger = logging.getLogger("griptape_nodes")

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

        result_data = set_value_at_path(data, path, replacement)
        result_str = self._to_yaml_string(result_data)

        self.set_parameter_value("output", result_str)
        self.publish_update_to_parameter("output", result_str)

    def _preview_replacement(self) -> None:
        """Update the output while the user edits. A path that cannot be followed clears the output.

        Raising here would leave the new input stored on a node that still looks resolved, so the
        error is caught. process() raises the same error on a real run.
        """
        try:
            self._perform_replacement()
        except ValueError as e:
            self.set_parameter_value("output", None)
            self.publish_update_to_parameter("output", None)
            logger.debug("%s: skipped preview: %s", self.name, e)

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        if parameter.name in ["yaml", "path", "replacement_value"]:
            self._preview_replacement()
        return super().after_value_set(parameter, value)

    def process(self) -> None:
        self._perform_replacement()
