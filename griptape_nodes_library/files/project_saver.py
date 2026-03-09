"""ProjectSaver node - save a source file to a project location using situation-based path resolution."""

import logging
from pathlib import Path
from typing import Any

from griptape.artifacts import UrlArtifact  # used in _resolve_source_path isinstance check

from griptape_nodes.common.macro_parser import ParsedMacro
from griptape_nodes.common.project_templates.situation import SituationFilePolicy
from griptape_nodes.exe_types.core_types import (
    NodeMessageResult,
    Parameter,
    ParameterGroup,
    ParameterMessage,
    ParameterMode,
)
from griptape_nodes.exe_types.node_types import ControlNode
from griptape_nodes.exe_types.param_types.parameter_bool import ParameterBool
from griptape_nodes.exe_types.param_types.parameter_button import ParameterButton
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.files.file import File, FileDestination
from griptape_nodes.files.path_utils import FilenameParts
from griptape_nodes.files.project_file import FALLBACK_MACRO_TEMPLATE, SITUATION_TO_FILE_POLICY
from griptape_nodes.retained_mode.events.connection_events import (
    ListConnectionsForNodeRequest,
    ListConnectionsForNodeResultSuccess,
)
from griptape_nodes.retained_mode.events.os_events import ExistingFilePolicy
from griptape_nodes.retained_mode.events.project_events import (
    AttemptMapAbsolutePathToProjectRequest,
    AttemptMapAbsolutePathToProjectResultSuccess,
    GetAllSituationsForProjectRequest,
    GetAllSituationsForProjectResultSuccess,
    GetPathForMacroRequest,
    GetPathForMacroResultSuccess,
    GetSituationRequest,
    GetSituationResultSuccess,
    MacroPath,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.button import Button, ButtonDetailsMessagePayload
from griptape_nodes.traits.file_system_picker import FileSystemPicker
from griptape_nodes.traits.options import Options
from griptape_nodes_library.filesystem.file_output_settings import ClassifiedPath, PathResolutionScenario
from griptape_nodes_library.utils.image_utils import resolve_localhost_url_to_path

logger = logging.getLogger("griptape_nodes")

_IF_FILE_EXISTS_CHOICES = [
    {"name": "Increment Version", "option_value": SituationFilePolicy.CREATE_NEW},
    {"name": "Overwrite Existing", "option_value": SituationFilePolicy.OVERWRITE},
    {"name": "Abort / Error", "option_value": SituationFilePolicy.FAIL},
]
_IF_FILE_EXISTS_DISPLAY_NAMES = [choice["name"] for choice in _IF_FILE_EXISTS_CHOICES]
_POLICY_VALUE_TO_DISPLAY_NAME: dict[str, str] = {
    choice["option_value"]: choice["name"] for choice in _IF_FILE_EXISTS_CHOICES
}
_DISPLAY_NAME_TO_POLICY_VALUE: dict[str, str] = {
    choice["name"]: choice["option_value"] for choice in _IF_FILE_EXISTS_CHOICES
}


class ProjectSaver(ControlNode):
    """Save a source file to a project location using situation-based path resolution."""

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)

        self._updating_lock = False

        self._available_situations = self._fetch_available_situations()
        self._create_parameters()
        self._load_project_situation()

    def _fetch_available_situations(self) -> list[str]:
        """Fetch available situations from the project manager."""
        request = GetAllSituationsForProjectRequest()
        result = GriptapeNodes.handle_request(request)

        if not isinstance(result, GetAllSituationsForProjectResultSuccess):
            logger.error("%s: Failed to fetch situations from project", self.name)
            return []

        return sorted(result.situations.keys())

    def _create_parameters(self) -> None:
        """Create all parameters for the node."""
        self.source = Parameter(
            name="source",
            type="any",
            default_value=None,
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            tooltip="Source file path or localhost URL to save",
        )
        self.add_parameter(self.source)

        default_situation = (
            "save_node_output"
            if "save_node_output" in self._available_situations
            else (self._available_situations[0] if self._available_situations else "")
        )
        self.situation = ParameterString(
            name="situation",
            default_value=default_situation,
            allowed_modes={ParameterMode.PROPERTY},
            tooltip="Select the file save situation template to use for path resolution",
            traits={Options(choices=self._available_situations)},
            settable=True,
        )
        self.add_parameter(self.situation)

        with ParameterGroup(name="Situation Options") as situation_group:
            self.macro = ParameterString(
                name="macro",
                default_value="",
                tooltip="Macro template for output path resolution",
                settable=True,
            )

            self.if_file_exists = ParameterString(
                name="if_file_exists",
                default_value=_POLICY_VALUE_TO_DISPLAY_NAME[SituationFilePolicy.CREATE_NEW],
                tooltip="Policy for handling existing files when writing",
                allowed_modes={ParameterMode.PROPERTY},
                traits={Options(choices=_IF_FILE_EXISTS_DISPLAY_NAMES)},
                settable=True,
            )

            self.auto_create_path = ParameterBool(
                name="auto_create_path",
                default_value=True,
                tooltip="Whether to create parent directories automatically when saving",
                allowed_modes={ParameterMode.PROPERTY},
                settable=True,
            )

            ParameterButton(
                name="reset_situation",
                label="Reset to Default",
                variant="default",
                icon="refresh-cw",
                on_click=self._on_reset_situation_clicked,
            )

        self.add_node_element(situation_group)

        self.absolute_path_warning = ParameterMessage(
            name="absolute_path_warning",
            variant="warning",
            value="The file path specified could not be found within a directory defined within the current project. This will affect portability.",
            ui_options={"hide": True},
        )
        self.add_node_element(self.absolute_path_warning)

        self.filename = ParameterString(
            name="filename",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            tooltip="Destination filename with extension (leave empty to derive from source filename)",
            traits={
                FileSystemPicker(
                    allow_files=True,
                    allow_directories=False,
                    allow_create=True,
                    workspace_only=False,
                )
            },
        )
        self.add_parameter(self.filename)

        self.file_path = Parameter(
            name="file_path",
            type="str",
            default_value=None,
            allowed_modes={ParameterMode.OUTPUT},
            tooltip="Macro path of the saved file",
        )
        self.add_parameter(self.file_path)

    def _load_project_situation(self) -> None:
        """Load situation template from project and set macro default."""
        situation_name = self.get_parameter_value(self.situation.name)
        result = GriptapeNodes.handle_request(GetSituationRequest(situation_name=situation_name))

        if isinstance(result, GetSituationResultSuccess):
            macro_template = result.situation.macro
            on_collision = result.situation.policy.on_collision
            create_dirs = result.situation.policy.create_dirs
        else:
            logger.error("%s: Failed to load situation '%s', using fallback macro template", self.name, situation_name)
            macro_template = FALLBACK_MACRO_TEMPLATE
            on_collision = SituationFilePolicy.CREATE_NEW
            create_dirs = True

        self.set_parameter_value(self.macro.name, macro_template, initial_setup=True)
        display_name = _POLICY_VALUE_TO_DISPLAY_NAME.get(
            on_collision, _POLICY_VALUE_TO_DISPLAY_NAME[SituationFilePolicy.CREATE_NEW]
        )
        self.set_parameter_value(self.if_file_exists.name, display_name, initial_setup=True)
        self.set_parameter_value(self.auto_create_path.name, create_dirs, initial_setup=True)

    def _classify_path(self, file_name_value: str) -> ClassifiedPath | str:
        """Classify the user's filename input into one of three scenarios."""
        parsed_macro = ParsedMacro(file_name_value)
        parse_result = GriptapeNodes.handle_request(GetPathForMacroRequest(parsed_macro=parsed_macro, variables={}))

        if not isinstance(parse_result, GetPathForMacroResultSuccess):
            return "Failed to parse macro"

        resolved = parse_result.resolved_path

        if not resolved.is_absolute():
            return ClassifiedPath(
                scenario=PathResolutionScenario.RELATIVE_PATH,
                normalized_path=file_name_value,
            )

        map_result = GriptapeNodes.handle_request(AttemptMapAbsolutePathToProjectRequest(absolute_path=resolved))

        if isinstance(map_result, AttemptMapAbsolutePathToProjectResultSuccess) and map_result.mapped_path:
            return ClassifiedPath(
                scenario=PathResolutionScenario.ABSOLUTE_PATH_INSIDE_PROJECT,
                normalized_path=map_result.mapped_path,
            )

        return ClassifiedPath(
            scenario=PathResolutionScenario.ABSOLUTE_PATH_OUTSIDE_PROJECT,
            normalized_path=str(resolved),
        )

    def _get_file_policy(self) -> ExistingFilePolicy:
        """Map the current if_file_exists parameter value to an ExistingFilePolicy."""
        display_name = self.get_parameter_value(self.if_file_exists.name)
        policy_value = _DISPLAY_NAME_TO_POLICY_VALUE.get(display_name, SituationFilePolicy.CREATE_NEW)
        return SITUATION_TO_FILE_POLICY.get(policy_value, ExistingFilePolicy.CREATE_NEW)

    def _build_file_from_template(self, macro_template: str, variables: dict[str, str | int]) -> FileDestination:
        """Build a FileDestination with a MacroPath from a template and variables."""
        macro_path = MacroPath(ParsedMacro(macro_template), variables)
        create_dirs = bool(self.get_parameter_value(self.auto_create_path.name))
        return FileDestination(macro_path, existing_file_policy=self._get_file_policy(), create_parents=create_dirs)

    def _resolve_source_path(self, value: Any) -> str | None:
        """Extract a readable absolute file path from the source value.

        Handles UrlArtifact objects and plain strings. Converts localhost URLs
        to absolute paths by resolving workspace-relative paths against the
        configured workspace directory.
        """
        if value is None:
            return None

        if isinstance(value, UrlArtifact):
            raw = value.value
        else:
            raw = str(value)

        if not raw:
            return None

        resolved = resolve_localhost_url_to_path(raw)

        # resolve_localhost_url_to_path returns a workspace-relative path for
        # localhost URLs. Convert it to an absolute path.
        resolved_path = Path(resolved)
        if not resolved_path.is_absolute():
            workspace_dir = Path(GriptapeNodes.ConfigManager().get_config_value("workspace_directory"))
            resolved = str(workspace_dir / resolved)

        return resolved

    def _get_source_node_name(self) -> str:
        """Return the name of the upstream node connected to the source input.

        Falls back to this node's own name when no connection exists.
        """
        result = GriptapeNodes.handle_request(ListConnectionsForNodeRequest(node_name=self.name))
        if isinstance(result, ListConnectionsForNodeResultSuccess):
            for conn in result.incoming_connections:
                if conn.target_parameter_name == self.source.name:
                    return conn.source_node_name
        return self.name

    def _build_file_destination(self, source_path: str) -> FileDestination | None:
        """Build a FileDestination from the current situation/macro/filename configuration.

        If filename is empty, derives the destination name from the source filename.
        """
        file_name_value = self.get_parameter_value(self.filename.name) or ""

        if not file_name_value:
            # Derive filename from source path
            file_name_value = Path(source_path).name

        classified = self._classify_path(file_name_value)
        if isinstance(classified, str):
            logger.error("%s: Could not classify path '%s': %s", self.name, file_name_value, classified)
            return None

        if classified.scenario == PathResolutionScenario.RELATIVE_PATH:
            macro_template = self.get_parameter_value(self.macro.name)
            if not macro_template:
                logger.error("%s: No macro template available", self.name)
                return None

            filename_path = Path(classified.normalized_path)
            parts = FilenameParts.from_filename(filename_path.name)
            variables: dict[str, str | int] = {
                "file_name_base": parts.stem,
                "file_extension": parts.extension,
                "node_name": self._get_source_node_name(),
            }
            return self._build_file_from_template(macro_template, variables)

        if classified.scenario == PathResolutionScenario.ABSOLUTE_PATH_INSIDE_PROJECT:
            return self._build_file_from_template(classified.normalized_path, {})

        # ABSOLUTE_PATH_OUTSIDE_PROJECT
        create_dirs = bool(self.get_parameter_value(self.auto_create_path.name))
        return FileDestination(
            classified.normalized_path,
            existing_file_policy=self._get_file_policy(),
            create_parents=create_dirs,
        )

    def after_value_set(self, parameter: Parameter, value: Any, *, initial_setup: bool = False) -> None:  # noqa: ARG002
        """React to situation changes by reloading the situation template."""
        if initial_setup or self._updating_lock:
            return

        if parameter.name == self.situation.name:
            self._updating_lock = True
            try:
                self._load_project_situation()
                self.publish_update_to_parameter(self.macro.name, self.get_parameter_value(self.macro.name))
                self.publish_update_to_parameter(
                    self.if_file_exists.name, self.get_parameter_value(self.if_file_exists.name)
                )
            finally:
                self._updating_lock = False

    def process(self) -> None:
        """Read the source file and write it to the configured project location."""
        source_value = self.get_parameter_value(self.source.name)
        source_path = self._resolve_source_path(source_value)

        if not source_path:
            msg = f"{self.name}: No source file provided"
            raise ValueError(msg)

        source_bytes = File(source_path).read_bytes()

        file_destination = self._build_file_destination(source_path)
        if file_destination is None:
            msg = f"{self.name}: Could not build file destination"
            raise ValueError(msg)

        written_file = file_destination.write_bytes(source_bytes)

        # Map the written absolute path back to a macro path
        written_path = Path(written_file.resolve())
        map_result = GriptapeNodes.handle_request(
            AttemptMapAbsolutePathToProjectRequest(absolute_path=written_path)
        )

        if isinstance(map_result, AttemptMapAbsolutePathToProjectResultSuccess) and map_result.mapped_path:
            output_path = map_result.mapped_path
        else:
            output_path = str(written_path)

        self.parameter_output_values[self.file_path.name] = output_path

        # Show or hide the warning based on whether the path is inside the project
        outside_project = not (
            isinstance(map_result, AttemptMapAbsolutePathToProjectResultSuccess) and map_result.mapped_path
        )
        self.absolute_path_warning.ui_options = {"hide": not outside_project}

    def _on_reset_situation_clicked(
        self,
        button: Button,  # noqa: ARG002
        button_details: ButtonDetailsMessagePayload,
    ) -> NodeMessageResult:
        """Reset macro to the situation's default template."""
        self._load_project_situation()

        self.publish_update_to_parameter(self.macro.name, self.get_parameter_value(self.macro.name))
        self.publish_update_to_parameter(self.if_file_exists.name, self.get_parameter_value(self.if_file_exists.name))
        self.publish_update_to_parameter(
            self.auto_create_path.name, self.get_parameter_value(self.auto_create_path.name)
        )

        return NodeMessageResult(
            success=True,
            details="Situation parameters reset to defaults",
            response=button_details,
            altered_workflow_state=True,
        )
