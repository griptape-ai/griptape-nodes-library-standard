"""Tests for migrating a retired Sora node onto a video node that still works (issue #614).

OpenAI removed the Videos API and every Sora 2 model on 2026-09-24 without naming a successor,
so ``SoraVideoGeneration`` survives only to keep saved workflows loadable. These tests cover the
graph surgery behind its migrate buttons: the replacement has to inherit the values, the
connections, and the canvas position, and the Sora node has to be gone afterwards.
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

import pytest
from griptape_nodes.retained_mode.events.connection_events import CreateConnectionRequest, CreateConnectionResultSuccess
from griptape_nodes.retained_mode.events.flow_events import (
    CreateFlowRequest,
    CreateFlowResultSuccess,
    DeleteFlowRequest,
)
from griptape_nodes.retained_mode.events.node_events import CreateNodeRequest, CreateNodeResultSuccess
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

from griptape_nodes_library.video._sora_migration import SEEDANCE_TARGET, VEO_TARGET, migrate_sora_node
from griptape_nodes_library.video.sora_video_generation import SoraVideoGeneration

FLOW_NAME = "canvas"
SORA_POSITION = {"position": {"x": 137.0, "y": 42.0}}


@pytest.fixture
def flow(griptape_nodes: GriptapeNodes) -> Generator[str, None, None]:  # noqa: ARG001
    """Create a fresh top-level flow (under an ambient test workflow) for each test."""
    context_manager = GriptapeNodes.ContextManager()
    context_manager.push_workflow(workflow_name="test_sora_migration_workflow")
    try:
        result = GriptapeNodes.handle_request(CreateFlowRequest(parent_flow_name=None, flow_name=FLOW_NAME))
        assert isinstance(result, CreateFlowResultSuccess)
        yield FLOW_NAME
        GriptapeNodes.handle_request(DeleteFlowRequest(flow_name=FLOW_NAME))
    finally:
        context_manager.pop_workflow()


def _create(node_type: str, flow: str, metadata: dict | None = None) -> str:
    result = GriptapeNodes.handle_request(
        CreateNodeRequest(node_type=node_type, override_parent_flow_name=flow, metadata=metadata)
    )
    assert isinstance(result, CreateNodeResultSuccess), f"could not create {node_type}"
    return result.node_name


def _connect(source_node: str, source_param: str, target_node: str, target_param: str) -> None:
    result = GriptapeNodes.handle_request(
        CreateConnectionRequest(
            source_node_name=source_node,
            source_parameter_name=source_param,
            target_node_name=target_node,
            target_parameter_name=target_param,
        )
    )
    assert isinstance(result, CreateConnectionResultSuccess), (
        f"could not connect {source_node}.{source_param} -> {target_node}.{target_param}"
    )


def _node(name: str) -> Any:
    return GriptapeNodes.NodeManager().get_node_by_name(name)


def _node_exists(name: str) -> bool:
    return GriptapeNodes.ObjectManager().attempt_get_object_by_name(name) is not None


def _incoming(node_name: str) -> set[tuple[str, str, str]]:
    """Return the node's incoming connections as (upstream node, upstream param, own param)."""
    connections = GriptapeNodes.FlowManager().get_connections()
    return {
        (
            connections.connections[connection_id].source_node.name,
            connections.connections[connection_id].source_parameter.name,
            parameter_name,
        )
        for parameter_name, ids in connections.incoming_index.get(node_name, {}).items()
        for connection_id in ids
    }


def _outgoing(node_name: str) -> set[tuple[str, str, str]]:
    """Return the node's outgoing connections as (own param, downstream node, downstream param)."""
    connections = GriptapeNodes.FlowManager().get_connections()
    return {
        (
            parameter_name,
            connections.connections[connection_id].target_node.name,
            connections.connections[connection_id].target_parameter.name,
        )
        for parameter_name, ids in connections.outgoing_index.get(node_name, {}).items()
        for connection_id in ids
    }


@pytest.fixture
def sora(flow: str) -> SoraVideoGeneration:
    """A Sora node carrying a position and a full set of artist-set values."""
    name = _create("SoraVideoGeneration", flow, metadata=dict(SORA_POSITION))
    node = _node(name)
    node.set_parameter_value("prompt", "a lighthouse in fog")
    node.set_parameter_value("model", "sora-2-pro")
    node.set_parameter_value("seconds", 12)
    node.set_parameter_value("size", "1280x720")
    return node


class TestReplacesTheNode:
    def test_veo_replacement_takes_over_and_sora_is_gone(self, sora: SoraVideoGeneration) -> None:
        sora_name = sora.name

        outcome = migrate_sora_node(sora, VEO_TARGET)

        assert type(_node(outcome.new_node_name)).__name__ == "Veo3VideoGeneration"
        assert not _node_exists(sora_name), "the retired Sora node should be deleted"

    def test_replacement_lands_at_the_sora_position(self, sora: SoraVideoGeneration) -> None:
        outcome = migrate_sora_node(sora, VEO_TARGET)

        assert _node(outcome.new_node_name).metadata["position"] == SORA_POSITION["position"]

    def test_seedance_replacement_takes_over(self, sora: SoraVideoGeneration) -> None:
        sora_name = sora.name

        outcome = migrate_sora_node(sora, SEEDANCE_TARGET)

        assert type(_node(outcome.new_node_name)).__name__ == "Seedance20VideoGeneration"
        assert not _node_exists(sora_name)


class TestCarriesOverValues:
    def test_veo_translates_prompt_size_and_duration(self, sora: SoraVideoGeneration) -> None:
        outcome = migrate_sora_node(sora, VEO_TARGET)
        veo = _node(outcome.new_node_name)

        assert veo.get_parameter_value("prompt") == "a lighthouse in fog"
        assert veo.get_parameter_value("aspect_ratio") == "16:9"
        assert veo.get_parameter_value("resolution") == "720p"
        # Sora allowed 12s; Veo stops at 8, so the value is clamped and the artist is told.
        assert veo.get_parameter_value("duration_seconds") == "8"
        assert any("12s to 8s" in note for note in outcome.notes)

    def test_seedance_keeps_a_duration_veo_cannot_reach(self, sora: SoraVideoGeneration) -> None:
        outcome = migrate_sora_node(sora, SEEDANCE_TARGET)
        seedance = _node(outcome.new_node_name)

        assert seedance.get_parameter_value("duration") == 12
        assert seedance.get_parameter_value("ratio") == "16:9"
        assert outcome.notes == [], "12s survives intact, so there is nothing to warn about"

    def test_generated_audio_matches_sora_rather_than_the_target_default(self, sora: SoraVideoGeneration) -> None:
        # Seedance defaults generate_audio off, but Sora 2 always produced audio.
        outcome = migrate_sora_node(sora, SEEDANCE_TARGET)

        assert _node(outcome.new_node_name).get_parameter_value("generate_audio") is True

    def test_sora_model_id_is_not_carried_onto_another_provider(self, sora: SoraVideoGeneration) -> None:
        outcome = migrate_sora_node(sora, VEO_TARGET)

        assert "sora" not in str(_node(outcome.new_node_name).get_parameter_value("model_id")).lower()

    def test_default_output_filename_is_not_carried_over(self, sora: SoraVideoGeneration) -> None:
        outcome = migrate_sora_node(sora, VEO_TARGET)

        assert _node(outcome.new_node_name).get_parameter_value("output_file") != "sora_video.mp4"

    def test_customised_output_filename_survives(self, sora: SoraVideoGeneration) -> None:
        sora.set_parameter_value("output_file", "shot_042.mp4")

        outcome = migrate_sora_node(sora, VEO_TARGET)

        assert _node(outcome.new_node_name).get_parameter_value("output_file") == "shot_042.mp4"

    def test_seedance_input_mode_follows_whether_a_start_frame_was_set(self, sora: SoraVideoGeneration) -> None:
        sora.set_parameter_value("start_frame", "https://example.invalid/frame.png")

        outcome = migrate_sora_node(sora, SEEDANCE_TARGET)
        seedance = _node(outcome.new_node_name)

        # Seedance ignores frame inputs unless the mode asks for them.
        assert seedance.get_parameter_value("input_mode") == "First/Last Frame"
        # The image parameter wraps a bare URL in an artifact, so compare what it points at.
        assert seedance.get_parameter_value("first_frame").value == "https://example.invalid/frame.png"

    def test_seedance_stays_text_only_without_a_start_frame(self, sora: SoraVideoGeneration) -> None:
        outcome = migrate_sora_node(sora, SEEDANCE_TARGET)

        assert _node(outcome.new_node_name).get_parameter_value("input_mode") == "Text Only"


class TestCarriesOverConnections:
    def test_data_connections_move_to_the_replacement(self, sora: SoraVideoGeneration, flow: str) -> None:
        upstream = _create("DisplayText", flow)
        downstream = _create("DisplayVideo", flow)
        _connect(upstream, "text", sora.name, "prompt")
        _connect(sora.name, "video_url", downstream, "video")

        outcome = migrate_sora_node(sora, VEO_TARGET)

        assert (upstream, "text", "prompt") in _incoming(outcome.new_node_name)
        assert ("video_url", downstream, "video") in _outgoing(outcome.new_node_name)
        assert outcome.dropped_connections == []

    def test_control_connections_move_to_the_replacement(self, sora: SoraVideoGeneration, flow: str) -> None:
        upstream = _create("DisplayText", flow)
        downstream = _create("DisplayVideo", flow)
        _connect(upstream, "exec_out", sora.name, "exec_in")
        _connect(sora.name, "exec_out", downstream, "exec_in")

        outcome = migrate_sora_node(sora, VEO_TARGET)

        assert (upstream, "exec_out", "exec_in") in _incoming(outcome.new_node_name)
        assert ("exec_out", downstream, "exec_in") in _outgoing(outcome.new_node_name)
        assert outcome.dropped_connections == []

    def test_start_frame_connection_follows_the_seedance_rename(self, sora: SoraVideoGeneration, flow: str) -> None:
        upstream = _create("DisplayText", flow)
        _connect(upstream, "text", sora.name, "start_frame")

        outcome = migrate_sora_node(sora, SEEDANCE_TARGET)

        assert (upstream, "text", "first_frame") in _incoming(outcome.new_node_name)
        assert outcome.dropped_connections == []

    def test_a_connection_with_no_counterpart_is_reported_not_silently_lost(
        self, sora: SoraVideoGeneration, flow: str
    ) -> None:
        upstream = _create("DisplayText", flow)
        # Neither target has a `size` parameter; it becomes an aspect ratio plus a resolution.
        _connect(upstream, "text", sora.name, "size")

        outcome = migrate_sora_node(sora, VEO_TARGET)

        assert outcome.dropped_connections == [f"{upstream}.text -> {sora.name}.size"]
        assert "reconnect these by hand" in outcome.summary()

    def test_downstream_input_is_not_left_holding_the_deleted_node(self, sora: SoraVideoGeneration, flow: str) -> None:
        """An input takes one connection, so the replacement can only claim it once Sora lets go."""
        downstream = _create("DisplayVideo", flow)
        _connect(sora.name, "video_url", downstream, "video")

        outcome = migrate_sora_node(sora, VEO_TARGET)

        incoming = _incoming(downstream)
        assert incoming == {(outcome.new_node_name, "video_url", "video")}
