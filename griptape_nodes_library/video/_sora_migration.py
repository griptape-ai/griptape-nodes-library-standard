"""Rebuild a retired Sora node as a video node that can still run.

OpenAI removed the Videos API and every Sora 2 model on 2026-09-24 without shipping a
replacement, so `SoraVideoGeneration` cannot succeed under any configuration. It stays in
the library only so saved workflows still load; this module is the escape hatch that turns
one into a working node without the artist rewiring the graph by hand.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from griptape_nodes.retained_mode.events.connection_events import (
    CreateConnectionRequest,
    CreateConnectionResultSuccess,
    DeleteConnectionRequest,
)
from griptape_nodes.retained_mode.events.node_events import (
    CreateNodeRequest,
    CreateNodeResultSuccess,
    DeleteNodeRequest,
    GetFlowForNodeRequest,
    GetFlowForNodeResultSuccess,
    GetNodeMetadataRequest,
    GetNodeMetadataResultSuccess,
)
from griptape_nodes.retained_mode.events.parameter_events import SetParameterValueRequest
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

if TYPE_CHECKING:
    from griptape_nodes.exe_types.node_types import BaseNode

logger = logging.getLogger("griptape_nodes")

__all__ = ["SEEDANCE_TARGET", "VEO_TARGET", "MigrationOutcome", "MigrationTarget", "migrate_sora_node"]

# Sora's own defaults, standing in for a value the node never had set.
_SORA_DEFAULT_OUTPUT_FILE = "sora_video.mp4"
_SORA_DEFAULT_SECONDS = 4
_SORA_DEFAULT_SIZE = "720x1280"

# Sora sizes expressed as the aspect ratio and resolution the targets take instead.
_SORA_SIZE_GEOMETRY: dict[str, tuple[str, str]] = {
    "1280x720": ("16:9", "720p"),
    "720x1280": ("9:16", "720p"),
    "1792x1024": ("16:9", "1080p"),
    "1024x1792": ("9:16", "1080p"),
}


@dataclass(frozen=True)
class MigrationTarget:
    """A node type a retired Sora node can be rebuilt as.

    Attributes:
        node_type: Class name registered in the library, as `CreateNodeRequest` takes it.
        display_name: Target's library display name, for messages the artist reads.
        parameter_renames: Sora parameter name -> target parameter name, for the ones that
            carry the same meaning under a different name. Parameters absent here migrate
            under their own name when the target also has one.
    """

    node_type: str
    display_name: str
    parameter_renames: dict[str, str]


@dataclass
class MigrationOutcome:
    """What `migrate_sora_node` managed to carry over.

    Attributes:
        new_node_name: Name the engine assigned the replacement node.
        display_name: Replacement node's library display name.
        dropped_connections: Human-readable connections that could not be recreated,
            because the target has no counterpart parameter or refused the type.
        notes: Value translations an artist would want to know about, such as a duration
            the target cannot reproduce.
    """

    new_node_name: str
    display_name: str
    dropped_connections: list[str]
    notes: list[str]

    def summary(self) -> str:
        """Render the outcome as the message shown after the button click."""
        lines = [f"Replaced this node with '{self.new_node_name}' ({self.display_name})."]
        if self.notes:
            lines.append("")
            lines.extend(f"- {note}" for note in self.notes)
        if self.dropped_connections:
            lines.append("")
            lines.append("Connections that could not be carried over -- reconnect these by hand:")
            lines.extend(f"- {dropped}" for dropped in self.dropped_connections)
        return "\n".join(lines)


def _sora_seconds(sora_values: dict[str, Any]) -> int:
    """Read Sora's duration as a plain int.

    The dropdown hands back a single-element list rather than a scalar often enough that the
    Sora node unwraps it before every submission; do the same here.
    """
    seconds = sora_values.get("seconds")
    if isinstance(seconds, list):
        seconds = seconds[0] if seconds else None
    if seconds is None:
        return _SORA_DEFAULT_SECONDS
    try:
        return int(seconds)
    except (TypeError, ValueError):
        return _SORA_DEFAULT_SECONDS


def _sora_geometry(sora_values: dict[str, Any]) -> tuple[str, str]:
    """Read Sora's size as the aspect ratio and resolution the targets take instead."""
    size = sora_values.get("size")
    if not isinstance(size, str):
        size = _SORA_DEFAULT_SIZE
    return _SORA_SIZE_GEOMETRY.get(size, _SORA_SIZE_GEOMETRY[_SORA_DEFAULT_SIZE])


def _veo_translated_values(sora_values: dict[str, Any]) -> dict[str, Any]:
    """Translate Sora's seconds/size onto Veo 3.1's parameters."""
    aspect_ratio, resolution = _sora_geometry(sora_values)
    # Veo tops out at 8 seconds, and only offers 8 at 1080p.
    duration = "8" if resolution == "1080p" else str(min(_sora_seconds(sora_values), 8))
    return {
        "aspect_ratio": aspect_ratio,
        "resolution": resolution,
        "duration_seconds": duration,
        # Sora 2 always produced audio, so match it rather than the target's own default.
        "generate_audio": True,
    }


def _seedance_translated_values(sora_values: dict[str, Any]) -> dict[str, Any]:
    """Translate Sora's seconds/size/start_frame onto Seedance 2.0's parameters."""
    aspect_ratio, resolution = _sora_geometry(sora_values)
    return {
        "ratio": aspect_ratio,
        "resolution": resolution,
        # Seedance accepts 4-15 seconds, so every Sora duration survives intact.
        "duration": _sora_seconds(sora_values),
        # Seedance ignores frame inputs unless the mode asks for them, so the mode has to
        # follow whether Sora was given a start frame.
        "input_mode": "First/Last Frame" if sora_values.get("start_frame") else "Text Only",
        # Sora 2 always produced audio, so match it rather than the target's own default.
        "generate_audio": True,
    }


# Neither target renames `model` onto its own `model_id`, deliberately. A Sora model id means
# nothing to another provider, so carrying the value or a connection across would leave the
# dropdown holding "sora-2". The target keeps its own default instead, and a connection that fed
# Sora's model dropdown is reported as dropped for the artist to reconsider.
VEO_TARGET = MigrationTarget(
    node_type="Veo3VideoGeneration",
    display_name="Veo 3.1 Video Generation",
    # Veo names prompt, start_frame, and every output exactly as Sora does.
    parameter_renames={},
)

SEEDANCE_TARGET = MigrationTarget(
    node_type="Seedance20VideoGeneration",
    display_name="Seedance 2.0 Video Generation",
    parameter_renames={"start_frame": "first_frame"},
)

# Which Sora values are worth translating, and how, per target. Kept beside the targets
# rather than on them so `MigrationTarget` stays a plain description of a node.
_VALUE_TRANSLATORS = {
    VEO_TARGET.node_type: _veo_translated_values,
    SEEDANCE_TARGET.node_type: _seedance_translated_values,
}


def migrate_sora_node(source_node: BaseNode, target: MigrationTarget) -> MigrationOutcome:
    """Replace `source_node` with a `target` node holding its values and connections.

    The replacement lands at the Sora node's canvas position and the Sora node is deleted,
    so one click leaves a graph that runs. Connections the target cannot accept are reported
    on the outcome rather than failing the whole migration -- a mostly-rewired graph the
    artist can finish beats an untouched dead one.

    Args:
        source_node: The retired Sora node to replace.
        target: The node type to rebuild it as.

    Returns:
        What was carried over, including anything the artist has to reconnect by hand.

    Raises:
        RuntimeError: If the replacement node could not be created, in which case the
            Sora node and its connections are left exactly as they were.
    """
    source_name = source_node.name

    flow_result = GriptapeNodes.handle_request(GetFlowForNodeRequest(node_name=source_name))
    if not isinstance(flow_result, GetFlowForNodeResultSuccess):
        msg = (
            f"Attempted to replace '{source_name}' with a {target.display_name} node. "
            f"Failed because the flow containing '{source_name}' could not be found."
        )
        raise RuntimeError(msg)

    metadata_result = GriptapeNodes.handle_request(GetNodeMetadataRequest(node_name=source_name))
    metadata = metadata_result.metadata if isinstance(metadata_result, GetNodeMetadataResultSuccess) else None

    # Snapshot before touching anything: deleting the Sora node cascades its connections away.
    incoming, outgoing = _snapshot_connections(source_name)
    sora_values = _read_source_values(source_node)

    create_result = GriptapeNodes.handle_request(
        CreateNodeRequest(
            node_type=target.node_type,
            override_parent_flow_name=flow_result.flow_name,
            metadata=metadata,
            create_error_proxy_on_failure=False,
        )
    )
    if not isinstance(create_result, CreateNodeResultSuccess):
        msg = (
            f"Attempted to replace '{source_name}' with a {target.display_name} node. "
            f"Failed because the {target.display_name} node could not be created."
        )
        raise RuntimeError(msg)

    new_name = create_result.node_name
    notes = _apply_values(new_name, source_node, sora_values, target)

    # Free the downstream input slots before reconnecting: an input parameter holds one
    # incoming connection, so the Sora node has to let go before the replacement can take over.
    for source_param, target_node_name, target_param in outgoing:
        GriptapeNodes.handle_request(
            DeleteConnectionRequest(
                source_node_name=source_name,
                source_parameter_name=source_param,
                target_node_name=target_node_name,
                target_parameter_name=target_param,
            )
        )

    dropped = _reconnect(new_name, source_node, incoming, outgoing, target)

    GriptapeNodes.handle_request(DeleteNodeRequest(node_name=source_name))

    return MigrationOutcome(
        new_node_name=new_name,
        display_name=target.display_name,
        dropped_connections=dropped,
        notes=notes,
    )


def _snapshot_connections(
    node_name: str,
) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    """Record the node's connections as plain names, surviving the node's deletion.

    Returns:
        Incoming as (upstream node, upstream parameter, this node's parameter), and
        outgoing as (this node's parameter, downstream node, downstream parameter).
    """
    connections = GriptapeNodes.FlowManager().get_connections()

    incoming = [
        (
            connections.connections[connection_id].source_node.name,
            connections.connections[connection_id].source_parameter.name,
            parameter_name,
        )
        for parameter_name, connection_ids in connections.incoming_index.get(node_name, {}).items()
        for connection_id in connection_ids
    ]
    outgoing = [
        (
            parameter_name,
            connections.connections[connection_id].target_node.name,
            connections.connections[connection_id].target_parameter.name,
        )
        for parameter_name, connection_ids in connections.outgoing_index.get(node_name, {}).items()
        for connection_id in connection_ids
    ]
    return incoming, outgoing


def _read_source_values(source_node: BaseNode) -> dict[str, Any]:
    """Read the Sora values a migration cares about, skipping untouched defaults."""
    values: dict[str, Any] = {}
    for name in ("prompt", "seconds", "size", "start_frame", "output_file"):
        if source_node.get_parameter_by_name(name) is None:
            continue
        values[name] = source_node.get_parameter_value(name)
    return values


def _apply_values(
    new_name: str,
    source_node: BaseNode,
    sora_values: dict[str, Any],
    target: MigrationTarget,
) -> list[str]:
    """Set the target's parameters from the Sora node's values.

    Returns:
        Notes about translations the artist would want to know about.
    """
    notes: list[str] = []
    values: dict[str, Any] = {}

    # Values that mean the same thing on both nodes, under the same name or a renamed one.
    for name in ("prompt", "start_frame"):
        if name not in sora_values or sora_values[name] in (None, ""):
            continue
        values[target.parameter_renames.get(name, name)] = sora_values[name]

    # A default output filename is Sora's, not the artist's, so let the target name its own.
    output_file = sora_values.get("output_file")
    if output_file and output_file != _SORA_DEFAULT_OUTPUT_FILE:
        values["output_file"] = output_file

    values.update(_VALUE_TRANSLATORS[target.node_type](sora_values))

    seconds = _sora_seconds(sora_values)
    migrated_duration = values.get("duration_seconds") or values.get("duration")
    if str(migrated_duration) != str(seconds):
        notes.append(
            f"Duration changed from {seconds}s to {migrated_duration}s, the closest {target.display_name} supports."
        )

    for parameter_name, value in values.items():
        result = GriptapeNodes.handle_request(
            SetParameterValueRequest(node_name=new_name, parameter_name=parameter_name, value=value)
        )
        logger.debug("Migrating %s: set %s.%s -> %s", source_node.name, new_name, parameter_name, result)

    return notes


def _reconnect(
    new_name: str,
    source_node: BaseNode,
    incoming: list[tuple[str, str, str]],
    outgoing: list[tuple[str, str, str]],
    target: MigrationTarget,
) -> list[str]:
    """Rebuild the Sora node's connections on the replacement.

    Returns:
        Descriptions of the connections that could not be rebuilt.
    """
    dropped: list[str] = []

    for upstream_node, upstream_param, sora_param in incoming:
        new_param = target.parameter_renames.get(sora_param, sora_param)
        result = GriptapeNodes.handle_request(
            CreateConnectionRequest(
                source_node_name=upstream_node,
                source_parameter_name=upstream_param,
                target_node_name=new_name,
                target_parameter_name=new_param,
            )
        )
        if not isinstance(result, CreateConnectionResultSuccess):
            dropped.append(f"{upstream_node}.{upstream_param} -> {source_node.name}.{sora_param}")

    for sora_param, downstream_node, downstream_param in outgoing:
        new_param = target.parameter_renames.get(sora_param, sora_param)
        result = GriptapeNodes.handle_request(
            CreateConnectionRequest(
                source_node_name=new_name,
                source_parameter_name=new_param,
                target_node_name=downstream_node,
                target_parameter_name=downstream_param,
            )
        )
        if not isinstance(result, CreateConnectionResultSuccess):
            dropped.append(f"{source_node.name}.{sora_param} -> {downstream_node}.{downstream_param}")

    return dropped
