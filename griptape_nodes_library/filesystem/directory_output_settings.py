"""DirectoryOutputSettings node - configure folder save paths using macro expansion."""

import logging
import pathlib
from typing import Any

from griptape_nodes.common.macro_parser import SEQUENCE_VARIABLE_NAME, MacroSyntaxError, ParsedMacro
from griptape_nodes.common.project_templates.situation import BuiltInSituation, SituationFilePolicy
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import BaseNode
from griptape_nodes.exe_types.param_types.parameter_bool import ParameterBool
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.files.directory import DirectoryDestination
from griptape_nodes.files.path_utils import parse_file_uri
from griptape_nodes.files.situation_resolver import SITUATION_TO_FILE_POLICY
from griptape_nodes.retained_mode.events.os_events import ExistingFilePolicy
from griptape_nodes.retained_mode.events.project_events import (
    GetPathForMacroRequest,
    GetPathForMacroResultSuccess,
    GetSituationRequest,
    GetSituationResultSuccess,
    MacroPath,
    UnresolvedSequenceSlotBehavior,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.file_system_picker import FileSystemPicker
from griptape_nodes.traits.options import Options

from griptape_nodes_library.utils.situation_utils import build_situation_data, fetch_situations

logger = logging.getLogger("griptape_nodes")

_IF_DIR_EXISTS_CHOICES = [
    {"name": "Increment Version", "option_value": SituationFilePolicy.CREATE_NEW},
    {"name": "Overwrite Existing", "option_value": SituationFilePolicy.OVERWRITE},
    {"name": "Abort / Error", "option_value": SituationFilePolicy.FAIL},
]
_IF_DIR_EXISTS_DISPLAY_NAMES = [choice["name"] for choice in _IF_DIR_EXISTS_CHOICES]
_POLICY_VALUE_TO_DISPLAY_NAME: dict[str, str] = {
    choice["option_value"]: choice["name"] for choice in _IF_DIR_EXISTS_CHOICES
}
_DISPLAY_NAME_TO_POLICY_VALUE: dict[str, str] = {
    choice["name"]: choice["option_value"] for choice in _IF_DIR_EXISTS_CHOICES
}


def _explicit_local_path(value: str) -> str | None:
    """Return the local path `value` names explicitly, or None for a plain folder name.

    An absolute path, a `~`/`~/` path, or a `file://` URI names an exact location rather than a
    name to nest inside the situation's folder.
    """
    local_path_from_uri = parse_file_uri(value)
    if local_path_from_uri is not None:
        return local_path_from_uri
    if value == "~" or value.startswith(("~/", "~\\")):
        return str(pathlib.Path(value).expanduser())
    if pathlib.Path(value).is_absolute():
        return value
    return None


def _has_macro_variable(value: str) -> bool:
    """Return True if `value` contains at least one `{...}` macro variable."""
    try:
        return bool(ParsedMacro(value).get_variables())
    except MacroSyntaxError:
        return False


def _has_version_slot(value: str) -> bool:
    """Return True if `value` contains a `{###}` version slot."""
    try:
        return any(variable.name == SEQUENCE_VARIABLE_NAME for variable in ParsedMacro(value).get_variables())
    except MacroSyntaxError:
        return False


def _relative_variables(value: str) -> dict[str, str | int]:
    """Build the dir_name/sub_dirs macro variables for a plain relative folder name."""
    path = pathlib.Path(value)
    variables: dict[str, str | int] = {"dir_name": path.name}
    parent = str(path.parent)
    if parent and parent != ".":
        variables["sub_dirs"] = parent
    return variables


class DirectoryOutputSettings(BaseNode):
    """Configure folder save paths using situation templates and macro expansion.

    Exposes a DirectoryDestination via the directory_destination property (computed on
    demand from current parameter values) and outputs a resolved path string on the
    directory_destination parameter for display. Downstream nodes retrieve the
    DirectoryDestination directly via the DirectoryDestinationProvider protocol rather
    than deserializing it from the wire.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._create_parameters()
        self._load_situation_defaults()
        self._refresh_preview()

    @property
    def directory_destination(self) -> DirectoryDestination | None:
        """Build a DirectoryDestination from current parameter values.

        Computed fresh on every access rather than cached so consumers always see the
        latest configuration without depending on this node having been processed in the
        current session (a cached attribute would be None until `process()` ran, which
        breaks downstream reads on a freshly loaded workflow).
        """
        resolved = self._resolve_location_and_policy()
        if resolved is None:
            return None
        location, policy = resolved
        create_dirs = bool(self.get_parameter_value(self.auto_create_path.name))
        return DirectoryDestination(location, existing_dir_policy=policy, create_parents=create_dirs)

    def _create_parameters(self) -> None:
        """Create all parameters for the node."""
        situation_names, situation_descriptions = fetch_situations()
        self.situation = ParameterString(
            name="situation",
            default_value=BuiltInSituation.SAVE_OUTPUT_DIRECTORY,
            allowed_modes={ParameterMode.PROPERTY},
            tooltip="Select the situation template to use for folder path resolution",
            traits={Options(choices=situation_names)},
            settable=True,
        )
        self.add_parameter(self.situation)
        self.situation.update_ui_options(
            {
                "data": build_situation_data(situation_names, situation_descriptions),
                "dropdown_row_subtitles": True,
            }
        )

        self.dirname = ParameterString(
            name="dirname",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            tooltip="Folder name or path (supports macros like {workflow_name}_renders)",
            # The engine substitutes workflow variables into "{name}" tokens before a run, which
            # would resolve {outputs} and drop an unset {sub_dirs?:/}. The node resolves them itself.
            allow_variable_substitution=False,
            traits={
                FileSystemPicker(
                    allow_files=False,
                    allow_directories=True,
                    allow_create=True,
                    workspace_only=False,
                )
            },
        )
        self.add_parameter(self.dirname)

        self.macro = ParameterString(
            name="macro",
            default_value="",
            allowed_modes={ParameterMode.PROPERTY},
            tooltip="Macro template for output folder resolution",
            allow_variable_substitution=False,
            settable=True,
        )
        self.add_parameter(self.macro)

        self.if_dir_exists = ParameterString(
            name="if_dir_exists",
            default_value=_POLICY_VALUE_TO_DISPLAY_NAME[SituationFilePolicy.CREATE_NEW],
            allowed_modes={ParameterMode.PROPERTY},
            tooltip="Policy for handling an existing folder when creating it",
            traits={Options(choices=_IF_DIR_EXISTS_DISPLAY_NAMES)},
            settable=True,
        )
        self.add_parameter(self.if_dir_exists)

        self.auto_create_path = ParameterBool(
            name="auto_create_path",
            default_value=True,
            allowed_modes={ParameterMode.PROPERTY},
            tooltip="Whether to create parent directories automatically when saving",
            settable=True,
        )
        self.add_parameter(self.auto_create_path)

        self.directory_destination_parameter = Parameter(
            name="directory_destination",
            type="str",
            default_value=None,
            allowed_modes={ParameterMode.OUTPUT},
            tooltip="Resolved folder path for downstream nodes",
            output_type="str",
        )
        self.add_parameter(self.directory_destination_parameter)

    def _load_situation_defaults(self) -> None:
        """Load the macro template and policy from the selected situation."""
        situation_name = self.get_parameter_value(self.situation.name)
        result = GriptapeNodes.handle_request(GetSituationRequest(situation_name=situation_name))

        if isinstance(result, GetSituationResultSuccess):
            macro_template = result.situation.macro
            on_collision = result.situation.policy.on_collision
            create_dirs = result.situation.policy.create_dirs
        else:
            logger.error("%s: Failed to load situation '%s'; leaving macro empty", self.name, situation_name)
            macro_template = ""
            on_collision = SituationFilePolicy.CREATE_NEW
            create_dirs = True

        self.set_parameter_value(self.macro.name, macro_template, initial_setup=True)
        display_name = _POLICY_VALUE_TO_DISPLAY_NAME.get(
            on_collision, _POLICY_VALUE_TO_DISPLAY_NAME[SituationFilePolicy.CREATE_NEW]
        )
        self.set_parameter_value(self.if_dir_exists.name, display_name, initial_setup=True)
        self.set_parameter_value(self.auto_create_path.name, create_dirs, initial_setup=True)

    def _get_dir_policy(self) -> ExistingFilePolicy:
        """Map the current if_dir_exists parameter value to an ExistingFilePolicy."""
        display_name = self.get_parameter_value(self.if_dir_exists.name)
        policy_value = _DISPLAY_NAME_TO_POLICY_VALUE.get(display_name, SituationFilePolicy.CREATE_NEW)
        return SITUATION_TO_FILE_POLICY.get(policy_value, ExistingFilePolicy.CREATE_NEW)

    def _resolve_location_and_policy(self) -> tuple[str | MacroPath, ExistingFilePolicy] | None:
        """Classify the current dirname value into a location (and its write policy), or None.

        Returns:
            None when dirname is empty, or when dirname is a plain folder name and no macro
            template is available to nest it under.
        """
        dirname_value = self.get_parameter_value(self.dirname.name)
        if not dirname_value:
            return None
        dirname_value = str(dirname_value)

        explicit_path = _explicit_local_path(dirname_value)
        if explicit_path is not None or _has_macro_variable(dirname_value):
            # The user named the destination, so use it as given rather than nesting it inside
            # the situation's folder. Only a `{###}` slot versions it, as a typed-in Directory
            # value does; without one, DirectoryDestination would append `_1`, `_2`, ...
            location = explicit_path if explicit_path is not None else dirname_value
            if _has_version_slot(location):
                return location, self._get_dir_policy()
            return location, ExistingFilePolicy.OVERWRITE

        macro_template = self.get_parameter_value(self.macro.name)
        if not macro_template:
            return None

        macro_path = MacroPath(ParsedMacro(macro_template), _relative_variables(dirname_value))
        return macro_path, self._get_dir_policy()

    def _refresh_preview(self) -> None:
        """Resolve the current configuration and update the directory_destination preview output."""
        resolved = self._resolve_location_and_policy()
        if resolved is None:
            self.set_parameter_value(self.directory_destination_parameter.name, "")
            return

        location, _policy = resolved
        if isinstance(location, MacroPath):
            parsed_macro = location.parsed_macro
            variables = location.variables
        else:
            parsed_macro = ParsedMacro(location)
            variables = {}

        # Preview-only resolve: render an unbound `{###}` as `###` instead of failing.
        resolve_result = GriptapeNodes.handle_request(
            GetPathForMacroRequest(
                parsed_macro=parsed_macro,
                variables=variables,
                unresolved_sequence_slot_behavior=UnresolvedSequenceSlotBehavior.RENDER_SEQUENCE_PATTERN,
            )
        )
        if not isinstance(resolve_result, GetPathForMacroResultSuccess):
            logger.error("%s: Failed to resolve macro: %s", self.name, parsed_macro.template)
            return

        self.set_parameter_value(self.directory_destination_parameter.name, str(resolve_result.absolute_path))

    def after_value_set(self, parameter: Parameter, value: Any, *, initial_setup: bool = False) -> None:  # noqa: ARG002
        """React to parameter changes by re-resolving the preview."""
        if initial_setup:
            return
        if parameter.name == self.situation.name:
            self._load_situation_defaults()
            for name in (self.macro.name, self.if_dir_exists.name, self.auto_create_path.name):
                self.publish_update_to_parameter(name, self.get_parameter_value(name))
            self._refresh_preview()
            return
        if parameter.name in (
            self.dirname.name,
            self.macro.name,
            self.if_dir_exists.name,
            self.auto_create_path.name,
        ):
            self._refresh_preview()

    def process(self) -> None:
        """Resolve the preview and set the output parameter value."""
        self._refresh_preview()
