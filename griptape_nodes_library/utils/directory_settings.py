"""Shared check for a 'directory' parameter fed by the wrong kind of output settings node."""

from griptape_nodes.exe_types.core_types import NodeError
from griptape_nodes.exe_types.node_types import BaseNode
from griptape_nodes.retained_mode.events.destination_events import (
    GetConnectedFileDestinationRequest,
    GetConnectedFileDestinationResultSuccess,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

DIRECTORY_PARAMETER = "directory"


def reject_connected_file_destination(node: BaseNode) -> None:
    """Raise if a File Output Settings node is connected to 'directory'.

    ``ProjectDirectoryParameter`` ignores it and would read its resolved file path, passed in as the
    parameter's value, as an absolute folder.

    Raises:
        NodeError: If a File Output Settings node is connected to 'directory'.
    """
    # A node in no workflow has nothing connected to it.
    if node.workflow is None:
        return
    result = GriptapeNodes.handle_request(
        GetConnectedFileDestinationRequest(node_name=node.name, parameter_name=DIRECTORY_PARAMETER)
    )
    if not isinstance(result, GetConnectedFileDestinationResultSuccess) or result.destination is not None:
        msg = (
            "'directory' is connected to a File Output Settings node, which names a file, not a folder. "
            "Disconnect it and use the gear button to connect a Directory Output Settings node."
        )
        raise NodeError(msg)
