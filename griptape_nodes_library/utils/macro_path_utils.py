from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

from griptape_nodes.common.macro_parser import MacroSyntaxError, ParsedMacro
from griptape_nodes.exe_types.core_types import ParameterMessage
from griptape_nodes.exe_types.param_types.parameter_button import ParameterButton
from griptape_nodes.files.file import File
from griptape_nodes.files.project_file import ProjectFileDestination
from griptape_nodes.retained_mode.events.project_events import (
    AttemptMapAbsolutePathToProjectRequest,
    AttemptMapAbsolutePathToProjectResultSuccess,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes, logger

_WARNING_TEXT_EXTERNAL = (
    "This file is outside the project and may not be accessible when this workflow is shared or run headlessly. "
    "Click 'Copy to Project' to copy it into the project."
)
_WARNING_TEXT_INCOMING = (
    "This file is outside the project and may not be accessible when this workflow is shared or run headlessly. "
    "To use a project path, update the upstream node that is providing this value."
)


@dataclass
class MacroPathResult:
    resolved_path: str
    is_external: bool  # True if path is outside project or is a URL


def resolve_to_macro_path(path: str) -> MacroPathResult:
    """Attempt to resolve a path to a project macro path.

    If the path exists on disk and is inside the project, returns the macro form.
    If the path exists on disk but is outside the project, or if the path does not
    exist on disk (e.g. a remote URL), returns the original path with is_external=True
    so the caller can prompt the user to copy it into the project.
    """
    # Already a macro path (e.g. "{inputs}/image.png") — treat as in-project
    try:
        parsed = ParsedMacro(path)
        if parsed.get_variables():
            return MacroPathResult(resolved_path=path, is_external=False)
    except MacroSyntaxError:
        pass

    resolved = Path(path).resolve()
    if resolved.exists():
        result = GriptapeNodes.handle_request(AttemptMapAbsolutePathToProjectRequest(absolute_path=resolved))
        if isinstance(result, AttemptMapAbsolutePathToProjectResultSuccess) and result.mapped_path is not None:
            return MacroPathResult(resolved_path=result.mapped_path, is_external=False)
        logger.debug(f"Failed to map path to project macro: '{path}' (result: {result})")
        # Path exists on disk but is outside the project
        return MacroPathResult(resolved_path=path, is_external=True)

    # Path does not exist on disk — treat as external URL
    return MacroPathResult(resolved_path=path, is_external=True)


def create_external_file_controls(on_click_callback: Any) -> tuple[ParameterMessage, ParameterButton]:
    """Create the external warning message and copy button, initially hidden."""
    warning = ParameterMessage(
        name="external_file_warning",
        variant="warning",
        value=_WARNING_TEXT_EXTERNAL,
        hide=True,
    )
    button = ParameterButton(
        name="copy_to_project",
        label="Copy to Project",
        variant="default",
        icon="folder-input",
        hide=True,
        on_click=on_click_callback,
    )
    return warning, button


def update_external_file_controls(
    result: MacroPathResult,
    warning: ParameterMessage,
    button: ParameterButton,
    node_name: str,
    artifact_param_name: str,
) -> None:
    """Show/hide warning and copy button based on macro path resolution result.

    When the artifact is externally fed via an incoming connection, the copy button is
    hidden since copying won't persist — the connected value overwrites it on the next run.
    """
    if result.is_external:
        connections = GriptapeNodes.FlowManager().get_connections()
        target_connections = connections.incoming_index.get(node_name)
        has_incoming = bool(target_connections and target_connections.get(artifact_param_name))
        warning.hide = False
        if has_incoming:
            warning.value = _WARNING_TEXT_INCOMING
            button.hide = True
        else:
            warning.value = _WARNING_TEXT_EXTERNAL
            button.hide = False
    else:
        warning.hide = True
        button.hide = True


def copy_external_file_to_project(
    path: str,
    artifact_class: type,
    default_filename: str,
    node_name: str,
    parameter_name: str,
) -> tuple[Any, str]:
    """Copy an external file into the project's inputs folder.

    Returns a tuple of (new_artifact, macro_path).
    """
    content = File(path).read_bytes()
    filename = PurePosixPath(urlparse(path).path).name or default_filename
    dest = ProjectFileDestination(
        filename=filename,
        situation="copy_external_file",
        node_name=node_name,
        parameter_name=parameter_name,
    )
    saved = dest.write_bytes(content)
    macro_path = saved.location
    return artifact_class(macro_path), macro_path
