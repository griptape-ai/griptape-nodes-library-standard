from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterList, ParameterMode

from griptape_nodes_library.files.resolve_macro_path_base import BaseResolveMacroPath


class ResolveMacroPaths(BaseResolveMacroPath):
    """Resolve a list of macro paths (e.g. {inputs}/file.txt) to absolute filesystem paths."""

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)

        self.add_parameter(
            ParameterList(
                name="paths",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                input_types=["any"],
                default_value=[],
                ui_options={"hide_property": True},
                tooltip="One or more macro paths to resolve. Accepts any artifact type.",
            )
        )

        self.add_parameter(
            Parameter(
                name="resolved_paths",
                type="list",
                allowed_modes={ParameterMode.OUTPUT},
                default_value=[],
                tooltip="All resolved absolute paths, in the same order as the inputs.",
            )
        )

        self._create_status_parameters(
            result_details_tooltip="Details about the macro path resolution.",
            result_details_placeholder="Results will appear here after the node runs.",
        )

    def process(self) -> None:
        self._clear_execution_status()
        try:
            raw_values = self.get_parameter_list_value("paths")

            resolved: list[str] = []
            for raw in raw_values:
                path_str = self._extract_path_string(raw)
                if path_str:
                    resolved.append(self._resolve_one(path_str))

            self.parameter_output_values["resolved_paths"] = resolved
            self._set_status_results(
                was_successful=True,
                result_details=f"Resolved {len(resolved)} path(s).",
            )
        except Exception as exc:
            self._set_status_results(was_successful=False, result_details=str(exc))
            self.parameter_output_values["resolved_paths"] = []
            self._handle_failure_exception(exc)
