from __future__ import annotations

from pathlib import Path
from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.retained_mode.events.os_events import (
    CreateFileRequest,
    CreateFileResultFailure,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.file_system_picker import FileSystemPicker

from griptape_nodes_library.files.file_operation_base import FileOperationBaseNode


class CreateFolder(FileOperationBaseNode):
    """Create a folder at the specified path."""

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)

        self.folder_path = ParameterString(
            name="folder_path",
            tooltip="Path of the folder to create.",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            default_value="",
            placeholder_text="Enter folder path",
        )
        self.folder_path.add_trait(
            FileSystemPicker(
                allow_files=False,
                allow_directories=True,
                multiple=False,
            )
        )
        self.add_parameter(self.folder_path)

        self.create_parents = Parameter(
            name="create_parents",
            type="bool",
            input_types=["bool"],
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            default_value=True,
            tooltip="Whether to create missing parent directories.",
        )
        self.add_parameter(self.create_parents)

        self.fail_if_already_exists = Parameter(
            name="fail_if_already_exists",
            type="bool",
            input_types=["bool"],
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            default_value=False,
            tooltip="Whether to fail if the target folder already exists.",
        )
        self.add_parameter(self.fail_if_already_exists)

        self.created_path_output = Parameter(
            name="created_path",
            allow_input=False,
            allow_property=False,
            output_type="str",
            default_value="",
            tooltip="The folder path that was created (or already existed).",
        )
        self.add_parameter(self.created_path_output)

        self.already_existed_output = Parameter(
            name="already_existed",
            allow_input=False,
            allow_property=False,
            output_type="bool",
            default_value=False,
            tooltip="True if the folder already existed before this node ran.",
        )
        self.add_parameter(self.already_existed_output)

        self._create_status_parameters(
            result_details_tooltip="Details about the folder creation result",
            result_details_placeholder="Details on the create folder attempt will be presented here.",
            parameter_group_initially_collapsed=True,
        )

    def process(self) -> None:
        """Create a folder at the requested path."""
        self._clear_execution_status()

        folder_path_raw = self.get_parameter_value(self.folder_path.name)
        create_parents = bool(self.get_parameter_value(self.create_parents.name))
        fail_if_already_exists = bool(self.get_parameter_value(self.fail_if_already_exists.name))

        folder_path = self._extract_value_from_artifact(folder_path_raw)
        folder_path = GriptapeNodes.OSManager().sanitize_path_string(folder_path)

        self.set_parameter_value(self.created_path_output.name, "")
        self.set_parameter_value(self.already_existed_output.name, False)

        if not folder_path:
            msg = f"{self.name} attempted to create folder but folder_path is empty. Failed due to no path provided"
            self._set_status_results(was_successful=False, result_details=msg)
            return

        existing_path = self._check_path_exists(folder_path)
        if existing_path.exists:
            if not existing_path.is_directory:
                msg = f"{self.name} attempted to create folder but path exists and is not a directory: {folder_path}"
                self._set_status_results(was_successful=False, result_details=msg)
                return

            if fail_if_already_exists:
                msg = f"{self.name} attempted to create folder but it already exists and fail_if_already_exists is True: {folder_path}"
                self._set_status_results(was_successful=False, result_details=msg)
                return

            self.set_parameter_value(self.created_path_output.name, folder_path)
            self.set_parameter_value(self.already_existed_output.name, True)
            self.parameter_output_values[self.created_path_output.name] = folder_path
            self.parameter_output_values[self.already_existed_output.name] = True
            self._set_status_results(
                was_successful=True,
                result_details=f"Folder already exists: {folder_path}",
            )
            return

        if not create_parents:
            parent_path = self._check_path_exists(str(Path(folder_path).parent))
            if not parent_path.exists:
                msg = (
                    f"{self.name} attempted to create folder but parent directory does not exist and "
                    f"create_parents is False: {folder_path}"
                )
                self._set_status_results(was_successful=False, result_details=msg)
                return
            if not parent_path.is_directory:
                msg = f"{self.name} attempted to create folder but parent path is not a directory: {folder_path}"
                self._set_status_results(was_successful=False, result_details=msg)
                return

        create_result = GriptapeNodes.handle_request(
            CreateFileRequest(
                path=folder_path,
                is_directory=True,
                workspace_only=False,
            )
        )
        if isinstance(create_result, CreateFileResultFailure):
            failure_reason = (
                create_result.failure_reason.value
                if hasattr(create_result.failure_reason, "value")
                else "Unknown error"
            )
            error_details = f" - {create_result.result_details}" if create_result.result_details else ""
            msg = f"{self.name} failed to create folder '{folder_path}': {failure_reason}{error_details}"
            self._set_status_results(was_successful=False, result_details=msg)
            return

        self.set_parameter_value(self.created_path_output.name, folder_path)
        self.set_parameter_value(self.already_existed_output.name, False)
        self.parameter_output_values[self.created_path_output.name] = folder_path
        self.parameter_output_values[self.already_existed_output.name] = False
        self._set_status_results(was_successful=True, result_details=f"Folder created successfully: {folder_path}")
