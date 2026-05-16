from io import StringIO
from typing import Any

import jmespath
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import DataNode
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.exe_types.param_types.parameter_yaml import ParameterYaml
from griptape_nodes.retained_mode.events.parameter_events import SetParameterValueRequest
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from ruamel.yaml import YAML

_yaml = YAML()
_yaml.preserve_quotes = True


class YamlExtractValue(DataNode):
    """Extract values from YAML using JMESPath expressions."""

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)

        self.add_parameter(
            ParameterYaml(
                name="yaml",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                default_value="",
                tooltip="Input YAML data to extract from",
            )
        )

        path_param = ParameterString(
            name="path",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            default_value="",
            tooltip="JMESPath expression to extract data (e.g., 'user.name', 'items[0].title', '[*].name' for all names)",
            placeholder_text="ex: user.name, items[0].title, [*].name",
        )
        path_param.set_badge(
            variant="help",
            title="JMESPath Syntax",
            message="`user.name` — nested key\n`items[0].title` — array index\n`[*].name` — all names\n\n[JMESPath Docs](https://jmespath.org/)",
        )
        self.add_parameter(path_param)

        self.add_parameter(
            ParameterString(
                name="output",
                tooltip="The extracted value(s)",
                allowed_modes={ParameterMode.OUTPUT},
                multiline=True
            )
        )

    def _serialize_result(self, result: Any) -> str:
        if isinstance(result, (dict, list)):
            stream = StringIO()
            _yaml.dump(result, stream)
            return stream.getvalue().strip()
        return str(result) if result is not None else ""

    def _perform_extraction(self) -> None:
        yaml_str = self.get_parameter_value("yaml")
        path = self.get_parameter_value("path")

        if not yaml_str:
            result = {}
        else:
            try:
                data = _yaml.load(StringIO(yaml_str) if not isinstance(yaml_str, str) else yaml_str)
            except Exception as e:
                msg = f"{self.name}: Invalid YAML provided. Failed to parse: {e}. Input was: {str(yaml_str)[:200]!r}"
                raise ValueError(msg) from e

            if not path:
                result = data
            else:
                try:
                    result = jmespath.search(path, data)
                except (ValueError, TypeError) as e:
                    msg = f"{self.name}: Invalid JMESPath expression '{path}': {e}"
                    raise ValueError(msg) from e

                if result is None:
                    result = {}

        result_str = self._serialize_result(result)
        GriptapeNodes.handle_request(
            SetParameterValueRequest(parameter_name="output", value=result_str, node_name=self.name)
        )
        self.publish_update_to_parameter("output", result_str)

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        if parameter.name in ["yaml", "path"]:
            self._perform_extraction()
        return super().after_value_set(parameter, value)

    def process(self) -> None:
        self._perform_extraction()
