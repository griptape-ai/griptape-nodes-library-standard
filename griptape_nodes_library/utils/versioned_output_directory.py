"""Shared 'directory' parameter for nodes that save a group of files into one versioned folder.

The directory is a template such as ``{outputs}/frames_v{###}``. ``{###}`` is replaced by the next
free version number, so each run writes a new folder (frames_v001, frames_v002, ...).
"""

import os
import pathlib
import re
from collections.abc import Callable
from typing import Any

from griptape_nodes.exe_types.core_types import NodeError, NodeMessageResult, Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import BaseNode
from griptape_nodes.files.directory import DirectoryDestination
from griptape_nodes.files.file import File
from griptape_nodes.retained_mode.events.connection_events import (
    ListConnectionsForNodeRequest,
    ListConnectionsForNodeResultSuccess,
)
from griptape_nodes.retained_mode.events.destination_events import (
    GetConnectedDirectoryDestinationRequest,
    GetConnectedDirectoryDestinationResultSuccess,
    GetConnectedFileDestinationRequest,
    GetConnectedFileDestinationResultSuccess,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.retained_mode.retained_mode import RetainedMode
from griptape_nodes.traits.button import Button, ButtonDetailsMessagePayload
from griptape_nodes.traits.file_system_picker import FileSystemPicker

from griptape_nodes_library.utils.macro_path_utils import resolve_to_macro_path

DIRECTORY_PARAMETER = "directory"

_VERSION_TOKEN_RE = re.compile(r"\{(#+)\}")


def build_directory_parameter(
    *,
    default_directory: str,
    tooltip: str,
    on_configure_clicked: Callable[[Button, ButtonDetailsMessagePayload], NodeMessageResult],
    on_connection_removed: Callable[..., None],
) -> Parameter:
    """Build the 'directory' parameter, with a gear button that wires up a DirectoryOutputSettings node."""
    parameter = Parameter(
        name=DIRECTORY_PARAMETER,
        type="str",
        default_value=default_directory,
        input_types=["str"],
        allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
        tooltip=tooltip,
        ui_options={"placeholder_text": default_directory},
        # The engine treats any "{name" in a value as a workflow variable and unresolves the
        # node before every run. The node resolves {outputs} and {###} itself.
        allow_variable_substitution=False,
        traits={
            Button(
                icon="cog",
                size="icon",
                variant="secondary",
                tooltip="Create and connect a Directory Output Settings node",
                on_click=on_configure_clicked,
            ),
            FileSystemPicker(allow_files=False, allow_directories=True, multiple=False),
        },
    )
    parameter.on_incoming_connection_removed.append(on_connection_removed)
    return parameter


def normalize_directory_value(node: BaseNode, value: Any) -> None:
    """Rewrite a typed-in absolute path as a project macro path, when it lies inside the project."""
    if value and not str(value).startswith("{"):
        result = resolve_to_macro_path(str(value))
        if not result.is_external and result.resolved_path != str(value):
            node.set_parameter_value(DIRECTORY_PARAMETER, result.resolved_path)
            node.publish_update_to_parameter(DIRECTORY_PARAMETER, result.resolved_path)


def clear_directory_value(node: BaseNode) -> None:
    node.set_parameter_value(DIRECTORY_PARAMETER, "")
    node.publish_update_to_parameter(DIRECTORY_PARAMETER, "")


def configure_directory_with_settings_node(
    node: BaseNode, button_details: ButtonDetailsMessagePayload
) -> NodeMessageResult:
    """Create a DirectoryOutputSettings node and connect it to the node's 'directory' parameter."""
    conn_result = GriptapeNodes.handle_request(ListConnectionsForNodeRequest(node_name=node.name))
    if isinstance(conn_result, ListConnectionsForNodeResultSuccess):
        if any(c.target_parameter_name == DIRECTORY_PARAMETER for c in conn_result.incoming_connections):
            return NodeMessageResult(
                success=False,
                details="'directory' already has an incoming connection.",
                response=button_details,
                altered_workflow_state=False,
            )

    create_result = RetainedMode.create_node_relative_to(
        reference_node_name=node.name,
        new_node_type="DirectoryOutputSettings",
        offset_side="left",
        offset_x=-750,
        offset_y=0,
        lock=False,
    )
    if not isinstance(create_result, str):
        return NodeMessageResult(
            success=False,
            details="Failed to create a Directory Output Settings node.",
            response=button_details,
            altered_workflow_state=False,
        )

    configure_node_name = create_result
    configure_node = GriptapeNodes.ObjectManager().attempt_get_object_by_name(configure_node_name)
    if configure_node is not None:
        current_dir = node.get_parameter_value(DIRECTORY_PARAMETER) or ""
        if current_dir:
            configure_node.set_parameter_value("dirname", current_dir)
            configure_node.publish_update_to_parameter("dirname", current_dir)

    connection_result = RetainedMode.connect(
        source=f"{configure_node_name}.directory_destination",
        destination=f"{node.name}.{DIRECTORY_PARAMETER}",
    )
    if not connection_result.succeeded():
        return NodeMessageResult(
            success=False,
            details=f"Failed to connect {configure_node_name}.directory_destination to 'directory'.",
            response=button_details,
            altered_workflow_state=True,
        )
    return NodeMessageResult(
        success=True,
        details=f"Created and connected {configure_node_name}",
        response=button_details,
        altered_workflow_state=True,
    )


def resolve_output_dir(node: BaseNode, default_directory: str) -> pathlib.Path:
    """Return the folder to write into, taking the next free version when the template has ``{###}``.

    Raises:
        NodeError: If a Directory Output Settings node is connected to 'directory' but can't be
            read, or if a File Output Settings node is connected to 'directory' instead.
        DirectoryError: If the Directory Output Settings node's destination can't be created.
    """
    settings_folder = _settings_node_folder(node)
    if settings_folder is not None:
        return settings_folder
    value = (node.get_parameter_value(DIRECTORY_PARAMETER) or default_directory).strip().rstrip("/")
    if not value:
        value = default_directory
    # "~/renders" names a home folder, not a folder called "~" inside {outputs}.
    value = os.path.expanduser(value)  # noqa: PTH111
    if _VERSION_TOKEN_RE.search(value):
        return resolve_versioned_dir(value)
    # Treat bare relative paths (no leading / or macro {) as relative to {outputs}
    if not value.startswith(("/", "{")):
        value = f"{{outputs}}/{value}"
    return pathlib.Path(File(value).resolve())


def _settings_node_folder(node: BaseNode) -> pathlib.Path | None:
    """The folder a Directory Output Settings node connected to 'directory' names, or None when none is.

    Asked by request rather than looked up by name, so it is answered where the settings node lives
    and works when the node runs in a worker.

    Raises:
        NodeError: If the connected settings node's destination can't be read, or if a File Output
            Settings node is connected instead of a Directory Output Settings node.
        DirectoryError: If the destination can't be created.
    """
    if node.workflow is None:
        # A node in no workflow has nothing connected to it.
        return None
    directory_result = GriptapeNodes.handle_request(
        GetConnectedDirectoryDestinationRequest(node_name=node.name, parameter_name=DIRECTORY_PARAMETER)
    )
    if not isinstance(directory_result, GetConnectedDirectoryDestinationResultSuccess):
        raise NodeError(str(directory_result.result_details))
    if directory_result.destination is not None:
        directory = DirectoryDestination.from_spec(directory_result.destination, node.workflow).create()
        return directory.resolve()

    # No Directory Output Settings node connected. A File Output Settings node names a file,
    # not a folder, so its destination's parent is not what the user meant.
    file_result = GriptapeNodes.handle_request(
        GetConnectedFileDestinationRequest(node_name=node.name, parameter_name=DIRECTORY_PARAMETER)
    )
    if not isinstance(file_result, GetConnectedFileDestinationResultSuccess) or file_result.destination is not None:
        msg = (
            "'directory' is connected to a File Output Settings node, which names a file, not a folder. "
            "Disconnect it and use the gear button to connect a Directory Output Settings node."
        )
        raise NodeError(msg)
    return None


def resolve_versioned_dir(template: str) -> pathlib.Path:
    m = _VERSION_TOKEN_RE.search(template)
    if not m:
        msg = f"No version token in template: {template!r}"
        raise ValueError(msg)
    num_digits = len(m.group(1))
    prefix_template = template[: m.start()].rstrip("/")

    # Resolve any macros in the prefix by substituting zeros as a stand-in
    test_value = prefix_template + "0" * num_digits
    if not test_value.startswith(("/", "{")):
        test_value = f"{{outputs}}/{test_value}"
    resolved_test = pathlib.Path(File(test_value).resolve())

    parent = resolved_test.parent
    name_prefix = resolved_test.name[:-num_digits]  # e.g. "frames_v"

    next_version = 1
    if parent.exists():
        for entry in parent.iterdir():
            if not entry.is_dir() or not entry.name.startswith(name_prefix):
                continue
            version_str = entry.name[len(name_prefix) :]
            try:
                next_version = max(next_version, int(version_str) + 1)
            except ValueError:
                pass

    return parent / f"{name_prefix}{next_version:0{num_digits}d}"
