import logging
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

from griptape_nodes.exe_types.core_types import Parameter
from griptape_nodes.exe_types.node_types import SuccessFailureNode
from griptape_nodes.exe_types.param_types.parameter_bool import ParameterBool
from griptape_nodes.exe_types.param_types.parameter_int import ParameterInt
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.retained_mode.events.os_events import (
    ListDirectoryRequest,
    ListDirectoryResultFailure,
    ListDirectoryResultSuccess,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.file_system_picker import FileSystemPicker
from griptape_nodes.traits.options import Options

logger = logging.getLogger("griptape_nodes")
LIST_OPTIONS = [
    "List files and folders",
    "List files only",
    "List folders only",
]


class ListFiles(SuccessFailureNode):
    def __init__(
        self,
        name: str,
        metadata: dict[Any, Any] | None = None,
    ) -> None:
        super().__init__(name, metadata)

        # Add input parameters
        self.directory_path = ParameterString(
            name="directory_path",
            allow_output=False,
            default_value="",
            tooltip="The directory path to list files from.",
        )
        self.directory_path.add_trait(
            FileSystemPicker(
                allow_files=False,
                allow_directories=True,
                multiple=False,
            )
        )

        self.match_pattern = ParameterString(
            name="match_pattern",
            default_value="",
            tooltip="Optional shell-style glob for entry names (e.g. *.jpg, report_*.txt). Empty includes all. Applies to both files and folders when listed.",
        )
        self.match_pattern_case_sensitive = ParameterBool(
            name="match_pattern_case_sensitive",
            default_value=True,
            tooltip="When match_pattern is set: if True, letter case must match; if False, matching is case-insensitive (e.g. *.jpg matches .JPG). Ignored when match_pattern is empty.",
        )
        self.show_hidden = ParameterBool(
            name="show_hidden",
            allow_output=False,
            default_value=False,
            tooltip="Whether to show hidden files/folders.",
        )

        self.list_options = ParameterString(
            name="list_options",
            allow_output=False,
            default_value=LIST_OPTIONS[0],
            tooltip="The options for the list files and folders.",
            traits={Options(choices=LIST_OPTIONS)},
        )

        self.recursive = ParameterBool(
            name="recursive",
            allow_output=False,
            default_value=False,
            tooltip="If True, walk subdirectories and include matches at every depth. If False, only the directory_path level is listed.",
        )

        self.use_absolute_paths = ParameterBool(
            name="use_absolute_paths",
            allow_output=False,
            default_value=False,
            tooltip="Whether to return absolute paths. If False, returns paths as provided by the system (may be relative or absolute).",
        )

        self.add_parameter(self.directory_path)
        self.add_parameter(self.match_pattern)
        self.add_parameter(self.match_pattern_case_sensitive)
        self.add_parameter(self.list_options)
        self.add_parameter(self.recursive)
        self.add_parameter(self.show_hidden)
        self.add_parameter(self.use_absolute_paths)

        # Add output parameters
        self.add_parameter(
            Parameter(
                name="file_paths",
                allow_input=False,
                allow_property=False,
                output_type="list",
                default_value=[],
                tooltip="Paths for matched entries. When recursive is True, includes nested files/folders under directory_path.",
            )
        )

        self.add_parameter(
            Parameter(
                name="file_names",
                allow_input=False,
                allow_property=False,
                output_type="list",
                default_value=[],
                tooltip="Base names for matched entries only (no parent path). Same for nested matches when recursive is True.",
            )
        )

        self.add_parameter(
            ParameterInt(
                name="file_count",
                allow_input=False,
                allow_property=False,
                default_value=0,
                tooltip="Total number of files found.",
            )
        )
        self._create_status_parameters(
            result_details_tooltip="Details about the list file result",
            result_details_placeholder="Details on the list file attempt will be presented here.",
        )

    @staticmethod
    def _directory_visit_key(directory_path: str | None) -> str:
        """Stable key for visited dirs (avoids repeated work on symlink cycles)."""
        if not directory_path:
            return "\x00<list_directory_root>\x00"
        try:
            return str(Path(directory_path).resolve())
        except (OSError, RuntimeError):
            return str(Path(directory_path).absolute())

    def _name_matches_pattern(self, name: str, pattern: str, *, case_sensitive: bool) -> bool:
        if case_sensitive:
            return fnmatchcase(name, pattern)
        return fnmatchcase(name.casefold(), pattern.casefold())

    def _filter_entries(
        self,
        entries: list,
        *,
        include_files: bool,
        include_folders: bool,
        match_pattern: str,
        match_pattern_case_sensitive: bool,
    ) -> list:
        """Filter entries based on include_files, include_folders, and optional fnmatch pattern on entry.name."""
        pattern = match_pattern.strip()
        use_pattern = bool(pattern)
        filtered_entries = []
        for entry in entries:
            if entry.is_dir:
                if include_folders and (
                    not use_pattern
                    or self._name_matches_pattern(entry.name, pattern, case_sensitive=match_pattern_case_sensitive)
                ):
                    filtered_entries.append(entry)
            elif include_files and (
                not use_pattern
                or self._name_matches_pattern(entry.name, pattern, case_sensitive=match_pattern_case_sensitive)
            ):
                filtered_entries.append(entry)
        return filtered_entries

    def _convert_paths(self, entries: list, *, use_absolute_paths: bool) -> list[str]:
        """Extract paths from entries, optionally converting to absolute paths."""
        if use_absolute_paths:
            return [entry.absolute_path for entry in entries]
        return [entry.path for entry in entries]

    def _collect_entries_recursive(
        self,
        root_directory_path: str | None,
        *,
        show_hidden: bool,
        include_files: bool,
        include_folders: bool,
        match_pattern: str,
        match_pattern_case_sensitive: bool,
    ) -> tuple[list, str | None]:
        """Walk subdirectories depth-first. Returns (filtered entries, error_message)."""
        collected: list = []
        stack: list[str | None] = [root_directory_path]
        visited: set[str] = set()

        while stack:
            current = stack.pop()
            visit_key = self._directory_visit_key(current)
            if visit_key in visited:
                continue
            visited.add(visit_key)

            request = ListDirectoryRequest(
                directory_path=current,
                show_hidden=show_hidden,
                workspace_only=False,
            )
            result = GriptapeNodes.handle_request(request)

            if isinstance(result, ListDirectoryResultFailure):
                error_msg = getattr(result, "error_message", "Unknown error occurred")
                return [], str(error_msg)
            if not isinstance(result, ListDirectoryResultSuccess):
                return [], "unexpected result type from directory listing"

            # Descend into subdirs in a stable order (reverse so first listed dir is processed next on pop).
            subdirs: list[str] = []
            for entry in result.entries:
                collected.extend(
                    self._filter_entries(
                        [entry],
                        include_files=include_files,
                        include_folders=include_folders,
                        match_pattern=match_pattern,
                        match_pattern_case_sensitive=match_pattern_case_sensitive,
                    )
                )
                if entry.is_dir:
                    subdirs.append(entry.path)
            for d in reversed(subdirs):
                stack.append(d)

        return collected, None

    def process(self) -> None:
        self._clear_execution_status()
        directory_path = self.get_parameter_value("directory_path")
        # Clean directory path to remove newlines/carriage returns that cause Windows errors
        if directory_path:
            directory_path = GriptapeNodes.OSManager().sanitize_path_string(directory_path)
        show_hidden = self.get_parameter_value("show_hidden")
        match_pattern = self.get_parameter_value("match_pattern") or ""
        match_pattern_case_sensitive = self.get_parameter_value("match_pattern_case_sensitive")
        list_options = self.get_parameter_value("list_options")
        recursive = self.get_parameter_value("recursive")
        use_absolute_paths = self.get_parameter_value("use_absolute_paths")

        # Determine include_files and include_folders based on list_options
        if list_options == LIST_OPTIONS[0]:  # "List files and folders"
            include_files = True
            include_folders = True
        elif list_options == LIST_OPTIONS[1]:  # "List files only"
            include_files = True
            include_folders = False
        elif list_options == LIST_OPTIONS[2]:  # "List folders only"
            include_files = False
            include_folders = True
        else:
            # Fallback to default if invalid option
            msg = f"{self.name}: Invalid list options: {list_options}, listing all files and folders"
            logger.warning(msg)
            include_files = True
            include_folders = True

        root = directory_path if directory_path else None

        if recursive:
            filtered_entries, list_error = self._collect_entries_recursive(
                root,
                show_hidden=show_hidden,
                include_files=include_files,
                include_folders=include_folders,
                match_pattern=match_pattern,
                match_pattern_case_sensitive=match_pattern_case_sensitive,
            )
            if list_error:
                msg = f"{self.name} failed to list directory: {list_error}"
                self._set_status_results(was_successful=False, result_details=f"Failure: {msg}")
                return
        else:
            request = ListDirectoryRequest(
                directory_path=root,
                show_hidden=show_hidden,
                workspace_only=False,  # Allow system-wide browsing
            )
            result = GriptapeNodes.handle_request(request)

            if isinstance(result, ListDirectoryResultFailure):
                error_msg = getattr(result, "error_message", "Unknown error occurred")
                msg = f"{self.name} failed to list directory: {error_msg}"
                self._set_status_results(was_successful=False, result_details=f"Failure: {msg}")
                return

            if not isinstance(result, ListDirectoryResultSuccess):
                msg = f"{self.name} received unexpected result type from directory listing"
                self._set_status_results(was_successful=False, result_details=f"Failure: {msg}")
                return

            filtered_entries = self._filter_entries(
                result.entries,
                include_files=include_files,
                include_folders=include_folders,
                match_pattern=match_pattern,
                match_pattern_case_sensitive=match_pattern_case_sensitive,
            )
        file_paths = self._convert_paths(filtered_entries, use_absolute_paths=use_absolute_paths)
        file_names = [entry.name for entry in filtered_entries]

        # Set output values
        self.set_parameter_value("file_paths", file_paths)
        self.set_parameter_value("file_names", file_names)
        self.set_parameter_value("file_count", len(file_paths))
        self._set_status_results(was_successful=True, result_details="Success")
