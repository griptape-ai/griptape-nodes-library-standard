"""Helpers for keeping a node's outgoing connections valid when an output's type changes."""

from griptape_nodes.exe_types.core_types import ParameterType
from griptape_nodes.exe_types.node_types import BaseNode
from griptape_nodes.retained_mode.events.connection_events import DeleteConnectionRequest
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes, logger


def remove_incompatible_outgoing_connections(node: BaseNode, parameter_name: str, new_type: str) -> None:
    """Delete outgoing connections from node.parameter_name whose target no longer accepts new_type."""
    connections = GriptapeNodes.FlowManager().get_connections()
    connection_ids = connections.outgoing_index.get(node.name, {}).get(parameter_name, [])

    # Deleting a connection removes it from the index being read, so iterate over a copy.
    for connection_id in list(connection_ids):
        connection = connections.connections[connection_id]
        target_param = connection.target_parameter
        target_node = connection.target_node

        is_compatible = any(
            ParameterType.are_types_compatible(new_type, input_type) for input_type in target_param.input_types
        )
        if is_compatible:
            continue

        logger.info(
            "Removing incompatible connection: %s '%s' %s (%s) to '%s.%s' (accepts: %s)",
            node.__class__.__name__,
            node.name,
            parameter_name,
            new_type,
            target_node.name,
            target_param.name,
            target_param.input_types,
        )
        GriptapeNodes.handle_request(
            DeleteConnectionRequest(
                source_node_name=node.name,
                source_parameter_name=parameter_name,
                target_node_name=target_node.name,
                target_parameter_name=target_param.name,
            )
        )
