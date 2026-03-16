"""SaveToProject node - save a file to the project using situation-based path resolution."""

import logging
from pathlib import Path, PureWindowsPath
from typing import Any

from griptape.artifacts import UrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import SuccessFailureNode
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.files.file import File
from griptape_nodes.retained_mode.events.project_events import (
    AttemptMapAbsolutePathToProjectRequest,
    AttemptMapAbsolutePathToProjectResultSuccess,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

logger = logging.getLogger("griptape_nodes")


def _extract_source_path(value: Any) -> str | None:
    """Extract a file path string from various input types."""
    if isinstance(value, UrlArtifact):
        return str(value.value)
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and "value" in value:
        return str(value["value"])
    return None


class SaveToProject(SuccessFailureNode):
    """Save a file to the project using situation-based path resolution."""

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)

        self.add_parameter(
            Parameter(
                name="source",
                type="any",
                input_types=["any"],
                default_value=None,
                allowed_modes={ParameterMode.INPUT},
                tooltip="The file to save (path or UrlArtifact).",
            )
        )

        self._file_param = ProjectFileParameter(
            node=self,
            name="output_file",
            default_filename="output",
        )
        self._file_param.add_parameter()

        self.add_parameter(
            Parameter(
                name="saved_url",
                type="UrlArtifact",
                default_value=None,
                allowed_modes={ParameterMode.OUTPUT},
                tooltip="The resolved macro path of the saved file.",
            )
        )

        self._create_status_parameters(
            result_details_tooltip="Details about the save operation result",
            result_details_placeholder="Details on the save attempt will be presented here.",
        )

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        if parameter.name == "source":
            source_path = _extract_source_path(value)
            if source_path:
                self._update_default_filename(source_path)
        return super().after_value_set(parameter, value)

    def validate_before_node_run(self) -> list[Exception] | None:
        """Validate that required parameters are provided."""
        exceptions = []

        source_value = self.get_parameter_value("source")
        source_path = _extract_source_path(source_value)
        if not source_path:
            exceptions.append(
                ValueError(f"{self.name} requires a source file to save. Please connect or set the 'source' parameter.")
            )

        parent_exceptions = super().validate_before_node_run()
        if parent_exceptions:
            exceptions.extend(parent_exceptions)

        return exceptions or None

    async def aprocess(self) -> None:
        self._clear_execution_status()

        source_value = self.get_parameter_value("source")
        source_path = _extract_source_path(source_value)

        if not source_path:
            self._set_status_results(was_successful=False, result_details="No source file provided.")
            self._handle_failure_exception(RuntimeError("No source file provided."))
            return

        try:
            content = File(source_path).read_bytes()
        except Exception as e:
            msg = f"Failed to read source file '{source_path}': {e}"
            logger.error(msg)
            self._set_status_results(was_successful=False, result_details=msg)
            self._handle_failure_exception(RuntimeError(msg))
            return

        try:
            destination = self._file_param.build_file()
            saved_file = destination.write_bytes(content)
            saved_path = Path(saved_file.resolve())
        except Exception as e:
            msg = f"Failed to write destination file: {e}"
            logger.error(msg)
            self._set_status_results(was_successful=False, result_details=msg)
            self._handle_failure_exception(RuntimeError(msg))
            return

        # Map the saved absolute path back to a macro path
        url_artifact = self._resolve_url(saved_path)
        self.parameter_output_values["saved_url"] = url_artifact

        self._set_status_results(
            was_successful=True,
            result_details=f"File saved successfully to {saved_path}",
        )

    def _update_default_filename(self, source_path: str) -> None:
        """Update the default output filename extension based on the source path."""
        output_file_value = self.get_parameter_value("output_file")
        if output_file_value and output_file_value != "output":
            return

        # PureWindowsPath handles both forward slashes and backslashes as separators,
        # so it correctly extracts suffixes from any path style on any platform.
        suffix = PureWindowsPath(source_path).suffix
        new_default = f"output{suffix}" if suffix else "output"
        self.set_parameter_value("output_file", new_default)

    def _resolve_url(self, saved_path: Path) -> UrlArtifact:
        """Map an absolute path back to a macro path UrlArtifact."""
        result = GriptapeNodes.handle_request(AttemptMapAbsolutePathToProjectRequest(absolute_path=saved_path))
        if isinstance(result, AttemptMapAbsolutePathToProjectResultSuccess) and result.mapped_path is not None:
            return UrlArtifact(result.mapped_path)
        return UrlArtifact(str(saved_path))
