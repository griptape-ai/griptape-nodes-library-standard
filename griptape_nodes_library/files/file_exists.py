from typing import Any

from griptape_nodes.exe_types.node_types import DataNode
from griptape_nodes.exe_types.param_types.parameter_bool import ParameterBool
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.retained_mode.events.os_events import (
    GetFileInfoRequest,
    GetFileInfoResultSuccess,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.file_system_picker import FileSystemPicker


class FileExists(DataNode):
    """Check whether a file or directory exists at a given path."""

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)

        self.path_input = ParameterString(
            name="path",
            default_value="",
            tooltip="Path to check for existence.",
            placeholder_text="Example: my_folder/file.txt",
        )
        self.path_input.add_trait(
            FileSystemPicker(
                allow_files=True,
                allow_directories=True,
                multiple=False,
            )
        )
        self.add_parameter(self.path_input)

        self.add_parameter(
            ParameterBool(
                name="exists",
                allow_input=False,
                allow_property=False,
                default_value=False,
                tooltip="True if the path exists, False otherwise.",
            )
        )

        self.add_parameter(
            ParameterBool(
                name="is_directory",
                allow_input=False,
                allow_property=False,
                default_value=False,
                tooltip="True if the path exists and is a directory.",
            )
        )

    def process(self) -> None:
        path_str = self.get_parameter_value("path") or ""
        path_str = GriptapeNodes.OSManager().sanitize_path_string(str(path_str))

        exists = False
        is_directory = False

        if path_str:
            result = GriptapeNodes.handle_request(GetFileInfoRequest(path=path_str, workspace_only=False))
            if isinstance(result, GetFileInfoResultSuccess) and result.file_entry is not None:
                exists = True
                is_directory = result.file_entry.is_dir

        self.parameter_output_values["exists"] = exists
        self.parameter_output_values["is_directory"] = is_directory
        self.publish_update_to_parameter("exists", exists)
        self.publish_update_to_parameter("is_directory", is_directory)
