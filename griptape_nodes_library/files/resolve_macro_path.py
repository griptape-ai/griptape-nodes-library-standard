from typing import Any

from griptape.artifacts import UrlArtifact
from griptape_nodes.common.macro_parser import ParsedMacro
from griptape_nodes.exe_types.core_types import Parameter, ParameterList, ParameterMode
from griptape_nodes.exe_types.node_types import SuccessFailureNode
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.retained_mode.events.project_events import (
    GetPathForMacroRequest,
    GetPathForMacroResultFailure,
    GetPathForMacroResultSuccess,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes


class ResolveMacroPath(SuccessFailureNode):
    """Resolve one or more macro paths (e.g. {inputs}/file.txt) to absolute filesystem paths."""

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)

        self.paths_input = ParameterList(
            name="paths",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            input_types=["any"],
            default_value=[],
            ui_options={"hide_property": True},
            tooltip="One or more macro paths to resolve. Accepts a single path or a list of paths.",
        )
        self.add_parameter(self.paths_input)

        self.add_parameter(
            ParameterString(
                name="resolved_path",
                allow_input=False,
                allow_property=False,
                default_value="",
                tooltip="The first resolved absolute path. Convenient when only one path is provided.",
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
            raw_values = self.get_parameter_list_value(self.paths_input.name)

            resolved: list[str] = []
            for raw in raw_values:
                path_str = self._extract_path_string(raw)
                if path_str:
                    resolved.append(self._resolve_one(path_str))

            first = resolved[0] if resolved else ""

            self.parameter_output_values["resolved_path"] = first
            self.parameter_output_values["resolved_paths"] = resolved

            self._set_status_results(
                was_successful=True,
                result_details=f"Resolved {len(resolved)} path(s).",
            )
        except Exception as exc:
            self._set_status_results(was_successful=False, result_details=str(exc))
            self.parameter_output_values["resolved_path"] = ""
            self.parameter_output_values["resolved_paths"] = []
            self._handle_failure_exception(exc)

    def _extract_path_string(self, value: Any) -> str | None:
        """Extract a plain string from a str, UrlArtifact, or artifact-like object."""
        if value is None:
            return None
        if isinstance(value, str):
            raw = value
        elif isinstance(value, UrlArtifact):
            raw = str(value.value)
        elif hasattr(value, "value"):
            raw = str(value.value)
        else:
            raw = str(value)
        cleaned = GriptapeNodes.OSManager().sanitize_path_string(raw)
        return cleaned if cleaned else None

    def _resolve_one(self, path_str: str) -> str:
        """Resolve a single macro path to an absolute path. Raises on malformed or unresolvable macros."""
        result = GriptapeNodes.handle_request(GetPathForMacroRequest(parsed_macro=ParsedMacro(path_str), variables={}))
        if isinstance(result, GetPathForMacroResultSuccess):
            return str(result.resolved_path)
        if isinstance(result, GetPathForMacroResultFailure):
            msg = f"Could not resolve macro path '{path_str}': {result.failure_reason.value}"
            raise ValueError(msg)
        msg = f"Unexpected result resolving macro path '{path_str}'"
        raise ValueError(msg)
