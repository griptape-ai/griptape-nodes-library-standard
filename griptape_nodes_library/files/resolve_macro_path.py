from typing import Any

from griptape_nodes.exe_types.core_types import ParameterMode
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString

from griptape_nodes_library.files.resolve_macro_path_base import BaseResolveMacroPath


class ResolveMacroPath(BaseResolveMacroPath):
    """Resolve a macro path (e.g. {inputs}/file.txt) to an absolute filesystem path."""

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)

        self.add_parameter(
            ParameterString(
                name="path",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                default_value="",
                tooltip="Macro path to resolve. Accepts any artifact type.",
                placeholder_text="Example: {inputs}/my_file.png",
            )
        )

        self.add_parameter(
            ParameterString(
                name="resolved_path",
                allow_input=False,
                allow_property=False,
                default_value="",
                tooltip="The resolved absolute path.",
                placeholder_text="Example: /Users/me/project/inputs/my_file.png",
            )
        )

        self._create_status_parameters(
            result_details_tooltip="Details about the macro path resolution.",
            result_details_placeholder="Results will appear here after the node runs.",
        )

    def process(self) -> None:
        self._clear_execution_status()
        try:
            raw = self.get_parameter_value("path")
            path_str = self._extract_path_string(raw)
            resolved = self._resolve_one(path_str) if path_str else ""

            self.parameter_output_values["resolved_path"] = resolved
            self._set_status_results(was_successful=True, result_details=f"Resolved: {resolved}")
        except Exception as exc:
            self._set_status_results(was_successful=False, result_details=str(exc))
            self.parameter_output_values["resolved_path"] = ""
            self._handle_failure_exception(exc)
