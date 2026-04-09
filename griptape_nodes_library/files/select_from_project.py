from pathlib import Path
from typing import Any

from griptape.artifacts import UrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import DataNode, NodeDependencies
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.retained_mode.events.project_events import (
    AttemptMapAbsolutePathToProjectRequest,
    AttemptMapAbsolutePathToProjectResultSuccess,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.file_system_picker import FileSystemPicker


class SelectFromProject(DataNode):
    """Select a file or directory from the project and resolve it to a macro path."""

    def __init__(
        self,
        name: str,
        metadata: dict[Any, Any] | None = None,
    ) -> None:
        super().__init__(name, metadata)

        self.selected_path_input = ParameterString(
            name="selected_path",
            default_value="",
            tooltip="The file or directory to select.",
        )
        self.selected_path_input.add_trait(
            FileSystemPicker(
                allow_files=True,
                allow_directories=True,
            )
        )
        self.add_parameter(self.selected_path_input)

        self.add_parameter(
            Parameter(
                name="project_path",
                type="UrlArtifact",
                default_value=None,
                tooltip="The resolved project macro path.",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

        self.add_parameter(
            Parameter(
                name="in_project",
                type="bool",
                default_value=False,
                tooltip="Whether the selected path is within the project.",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

    def get_node_dependencies(self) -> NodeDependencies | None:
        deps = super().get_node_dependencies()
        if deps is None:
            deps = NodeDependencies()
        value = self.get_parameter_value("selected_path")
        if value and isinstance(value, str):
            deps.static_files.add(value)
        return deps

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        if parameter.name == "selected_path":
            file_path_str = str(value) if value is not None else ""
            self._update_project_path(file_path_str)

        return super().after_value_set(parameter, value)

    async def aprocess(self) -> None:
        file_path_str = self.get_parameter_value("selected_path") or ""
        self._update_project_path(str(file_path_str))

    def _resolve_project_path(self, file_path_str: str) -> tuple[UrlArtifact | None, bool]:
        """Resolve a file path to a UrlArtifact, mapping to a macro path if one exists.

        Returns a tuple of (resolved artifact, whether the path was in the project).
        """
        if not file_path_str:
            return None, False

        result = GriptapeNodes.handle_request(AttemptMapAbsolutePathToProjectRequest(absolute_path=Path(file_path_str)))

        if isinstance(result, AttemptMapAbsolutePathToProjectResultSuccess) and result.mapped_path is not None:
            return UrlArtifact(result.mapped_path), True

        return UrlArtifact(file_path_str), False

    def _update_project_path(self, file_path_str: str) -> None:
        """Compute the project path output from the given file path and keep both parameters in sync."""
        resolved, in_project = self._resolve_project_path(file_path_str)
        mapped_path = resolved.value if resolved is not None else ""

        # Direct assignment bypasses hooks to avoid re-triggering after_value_set
        self.parameter_values["selected_path"] = mapped_path
        self.publish_update_to_parameter("selected_path", mapped_path)

        self.parameter_output_values["project_path"] = resolved
        self.publish_update_to_parameter("project_path", resolved)

        self.parameter_output_values["in_project"] = in_project
        self.publish_update_to_parameter("in_project", in_project)
