"""Unit tests for ExtractFrames' Sequence output.

Frame files are empty placeholders in ``tmp_path``. The scan only looks at filenames,
so no ffmpeg or real video is needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from griptape_nodes.exe_types.core_types import NodeError
from griptape_nodes.retained_mode.events.connection_events import (
    CreateConnectionRequest,
    ListConnectionsForNodeRequest,
    ListConnectionsForNodeResultSuccess,
)
from griptape_nodes.retained_mode.events.flow_events import CreateFlowRequest
from griptape_nodes.retained_mode.events.node_events import CreateNodeRequest
from griptape_nodes.retained_mode.events.parameter_events import SetParameterValueRequest
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.button import Button, ButtonDetailsMessagePayload

from griptape_nodes_library.utils.directory_settings import reject_connected_file_destination
from griptape_nodes_library.video.extract_frames import ExtractFrames

if TYPE_CHECKING:
    from pathlib import Path


def _write_frames(directory: Path, numbers: list[int], *, prefix: str = "frames", padding: int = 4) -> None:
    for n in numbers:
        (directory / f"{prefix}.{str(n).zfill(padding)}.png").touch()


def _resolve_output_dir(node: ExtractFrames) -> Path:
    """Mirror what `process()` does to turn the 'directory' parameter into a folder."""
    reject_connected_file_destination(node)
    return node._directory_param.build_directory().create().resolve()


@pytest.fixture
def node() -> ExtractFrames:
    return ExtractFrames(name="test_extract_frames")


def _create_node(node_type: str, name: str, **values: Any) -> None:
    """Create a node through the engine's own request path, so it's a real member of the open workflow."""
    result = GriptapeNodes.handle_request(CreateNodeRequest(node_type=node_type, node_name=name))
    assert result.succeeded(), result.result_details
    for parameter_name, value in values.items():
        result = GriptapeNodes.handle_request(
            SetParameterValueRequest(node_name=name, parameter_name=parameter_name, value=value)
        )
        assert result.succeeded(), result.result_details


def _connect(source: str, source_parameter: str, target: str, target_parameter: str) -> None:
    result = GriptapeNodes.handle_request(
        CreateConnectionRequest(
            source_node_name=source,
            source_parameter_name=source_parameter,
            target_node_name=target,
            target_parameter_name=target_parameter,
        )
    )
    assert result.succeeded(), result.result_details


def _get_extract_frames(name: str = "ef") -> ExtractFrames:
    """Look up the node the engine created.

    Not an `isinstance` check: the engine loads library node classes through its own dynamic
    import, so the resulting object's class is a distinct (if identical) class from the one this
    test module imports directly.
    """
    found = GriptapeNodes.ObjectManager().attempt_get_object_by_name(name)
    assert found is not None
    assert type(found).__name__ == "ExtractFrames"
    return found


def _click_directory_gear(node: ExtractFrames) -> str:
    """Click the 'directory' gear button and return the name of the settings node it connected."""
    result = node._directory_param._on_configure_button_clicked(
        Button(), ButtonDetailsMessagePayload(label="", variant="secondary", size="icon", state="normal")
    )
    assert result.success, result.details
    connections = GriptapeNodes.handle_request(ListConnectionsForNodeRequest(node_name=node.name))
    assert isinstance(connections, ListConnectionsForNodeResultSuccess)
    (connection,) = [c for c in connections.incoming_connections if c.target_parameter_name == "directory"]
    return connection.source_node_name


@pytest.fixture
def real_graph(open_workflow: None, tmp_path: Path) -> ExtractFrames:
    """A real ExtractFrames node in a real flow, with outputs routed under tmp_path.

    `DirectoryDestination.create()` does real engine work (next-version lookup, mkdir, mapping the
    created path back to project-macro form) through the node's bound `WorkflowRuntime`, which only
    a node actually built by the engine -- not a stand-in object -- has.
    """
    GriptapeNodes.ConfigManager().workspace_path = tmp_path
    result = GriptapeNodes.handle_request(CreateFlowRequest(parent_flow_name=None, flow_name="TestFlow"))
    assert result.succeeded(), result.result_details
    _create_node("ExtractFrames", "ef")
    return _get_extract_frames()


class TestScanOutputSequence:
    def test_sparse_selection_keeps_gaps(self, node: ExtractFrames, tmp_path: Path) -> None:
        frames = [1, 4, 5, 6, 7, 8, 9, 11]
        _write_frames(tmp_path, frames)

        sequence = node._scan_output_sequence(tmp_path, "frames", 4, "png", frames)

        assert sequence is not None
        assert sequence.pattern == "frames.####.png"
        assert sequence.padding == 4
        assert sorted(sequence.present_numbers) == frames
        assert [entry.number for entry in sequence.entries] == frames
        assert sequence.entries[0].path == str(tmp_path / "frames.0001.png")

    def test_single_frame_is_a_one_item_sequence(self, node: ExtractFrames, tmp_path: Path) -> None:
        _write_frames(tmp_path, [7])

        sequence = node._scan_output_sequence(tmp_path, "frames", 4, "png", [7])

        assert sequence is not None
        assert [entry.number for entry in sequence.entries] == [7]

    def test_frames_outside_requested_range_are_excluded(self, node: ExtractFrames, tmp_path: Path) -> None:
        # Frames 1 and 20 are left over from an earlier run into the same directory.
        _write_frames(tmp_path, [1, 5, 6, 7, 20])

        sequence = node._scan_output_sequence(tmp_path, "frames", 4, "png", [5, 6, 7])

        assert sequence is not None
        assert sorted(sequence.present_numbers) == [5, 6, 7]

    def test_other_prefixes_are_ignored(self, node: ExtractFrames, tmp_path: Path) -> None:
        _write_frames(tmp_path, [1, 2, 3])
        _write_frames(tmp_path, [1, 2, 3], prefix="other")

        sequence = node._scan_output_sequence(tmp_path, "frames", 4, "png", [1, 2, 3])

        assert sequence is not None
        assert all(entry.path.startswith(str(tmp_path / "frames.")) for entry in sequence.entries)

    def test_empty_directory_returns_none(self, node: ExtractFrames, tmp_path: Path) -> None:
        assert node._scan_output_sequence(tmp_path, "frames", 4, "png", [1, 2, 3]) is None


class TestSequenceOutputWiring:
    def test_perform_extraction_sets_sequence_output(
        self, node: ExtractFrames, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        frames = [2, 4, 6]

        def fake_extract_frames(**_kwargs: object) -> list[Path]:
            _write_frames(tmp_path, frames)
            return [tmp_path / f"frames.{str(n).zfill(4)}.png" for n in frames]

        monkeypatch.setattr(node, "_extract_frames", fake_extract_frames)

        node._perform_extraction("unused.mp4", frames, tmp_path)

        sequence = node.parameter_output_values["sequence"]
        assert sequence is not None
        assert [entry.number for entry in sequence.entries] == frames
        assert node.parameter_output_values["output_frames"] == [entry.path for entry in sequence.entries]

    def test_safe_defaults_clear_sequence_output(self, node: ExtractFrames) -> None:
        node.parameter_output_values["sequence"] = object()

        node._set_safe_defaults()

        assert node.parameter_output_values["sequence"] is None


class TestOutputDirectory:
    """'directory' is a folder name placed by the project's situation, or an absolute path."""

    def test_a_typed_folder_name_creates_a_versioned_folder(self, real_graph: ExtractFrames, tmp_path: Path) -> None:
        real_graph.set_parameter_value("directory", "shots")

        # Each resolve actually creates the folder, so the next one takes the next version.
        assert _resolve_output_dir(real_graph) == tmp_path / "outputs" / "shots_v001"
        assert _resolve_output_dir(real_graph) == tmp_path / "outputs" / "shots_v002"

    def test_the_default_folder_name_is_versioned(self, real_graph: ExtractFrames, tmp_path: Path) -> None:
        assert _resolve_output_dir(real_graph) == tmp_path / "outputs" / "frames_v001"
        assert _resolve_output_dir(real_graph) == tmp_path / "outputs" / "frames_v002"

    def test_an_absolute_path_is_used_as_given(self, real_graph: ExtractFrames, tmp_path: Path) -> None:
        real_graph.set_parameter_value("directory", str(tmp_path / "frames"))

        assert _resolve_output_dir(real_graph) == tmp_path / "frames"

    def test_a_node_in_no_workflow_uses_the_directory_value(self, node: ExtractFrames, tmp_path: Path) -> None:
        """A node built outside any workflow (not yet added to one) has nothing connected to it."""
        node.set_parameter_value("directory", str(tmp_path / "frames"))

        assert _resolve_output_dir(node) == tmp_path / "frames"

    def test_old_default_directory_value_is_migrated_on_load(self, real_graph: ExtractFrames) -> None:
        """A saved workflow holding the pre-redesign default is rewritten to the new default."""
        result = GriptapeNodes.handle_request(
            SetParameterValueRequest(
                node_name=real_graph.name,
                parameter_name="directory",
                value="{outputs}/frames_v{###}",
                initial_setup=True,
            )
        )

        assert result.succeeded(), result.result_details
        assert real_graph.get_parameter_value("directory") == "frames"

    def test_an_unrelated_template_value_is_not_migrated(self, real_graph: ExtractFrames) -> None:
        result = GriptapeNodes.handle_request(
            SetParameterValueRequest(
                node_name=real_graph.name,
                parameter_name="directory",
                value="{outputs}/custom_v{###}",
                initial_setup=True,
            )
        )

        assert result.succeeded(), result.result_details
        assert real_graph.get_parameter_value("directory") == "{outputs}/custom_v{###}"


class TestOutputDirectoryFromSettings:
    """A settings node wired to 'directory' is read by request, so a worker run honors it too."""

    def test_the_gear_button_connects_a_directory_settings_node_that_names_the_folder(
        self, real_graph: ExtractFrames, tmp_path: Path
    ) -> None:
        settings_name = _click_directory_gear(real_graph)

        settings = GriptapeNodes.ObjectManager().attempt_get_object_by_name(settings_name)
        assert settings is not None
        assert type(settings).__name__ == "DirectoryOutputSettings"
        assert settings.get_parameter_value("situation") == "save_output_directory"
        assert settings.get_parameter_value("dirname") == "frames"
        assert _resolve_output_dir(real_graph) == tmp_path / "outputs" / "frames_v001"

    def test_directory_settings_with_a_plain_name_creates_a_versioned_folder(
        self, real_graph: ExtractFrames, tmp_path: Path
    ) -> None:
        _create_node("DirectoryOutputSettings", "dos", dirname="shots")
        _connect("dos", "directory_destination", real_graph.name, "directory")

        # Each resolve actually creates the folder, so the next one takes the next version.
        assert _resolve_output_dir(real_graph) == tmp_path / "outputs" / "shots_v001"
        assert _resolve_output_dir(real_graph) == tmp_path / "outputs" / "shots_v002"

    def test_directory_settings_with_a_hash_template_is_versioned(
        self, real_graph: ExtractFrames, tmp_path: Path
    ) -> None:
        _create_node("DirectoryOutputSettings", "dos", dirname="{outputs}/frames_v{###}")
        _connect("dos", "directory_destination", real_graph.name, "directory")

        assert _resolve_output_dir(real_graph) == tmp_path / "outputs" / "frames_v001"
        assert _resolve_output_dir(real_graph) == tmp_path / "outputs" / "frames_v002"

    def test_a_file_settings_node_with_a_filename_raises(self, real_graph: ExtractFrames) -> None:
        _create_node("FileOutputSettings", "fos", filename="frame.png")
        _connect("fos", "file_destination", real_graph.name, "directory")

        with pytest.raises(NodeError, match="names a file, not a folder"):
            _resolve_output_dir(real_graph)

    def test_a_file_settings_node_without_a_filename_also_raises(self, real_graph: ExtractFrames) -> None:
        _create_node("FileOutputSettings", "fos")
        _connect("fos", "file_destination", real_graph.name, "directory")

        with pytest.raises(NodeError, match="names a file, not a folder"):
            _resolve_output_dir(real_graph)


class TestProcessFailures:
    """Run-time failures must fail the node, or route down Failed when it is wired."""

    def test_missing_video_raises_when_failed_is_not_wired(self, node: ExtractFrames) -> None:
        node._has_outgoing_connections = lambda _param: False  # type: ignore[method-assign]

        with pytest.raises(ValueError, match="Video Input"):
            next(node.process())

        assert node._execution_succeeded is False

    def test_missing_video_routes_to_failed_when_wired(self, node: ExtractFrames) -> None:
        node._has_outgoing_connections = lambda _param: True  # type: ignore[method-assign]

        with pytest.raises(StopIteration):
            next(node.process())

        assert node._execution_succeeded is False

    def _stub_until_the_folder(self, node: ExtractFrames, monkeypatch: pytest.MonkeyPatch) -> None:
        """Get process() past the video and frame checks, then hit a Directory Output Settings node
        with nothing configured, so reading its destination fails.
        """
        monkeypatch.setattr(node, "_resolve_video_for_ffmpeg", lambda _raw: "clip.mp4")
        monkeypatch.setattr(node, "_build_frame_list", lambda _url: [1, 2])
        _create_node("DirectoryOutputSettings", "dos")
        _connect("dos", "directory_destination", node.name, "directory")

    def test_a_settings_failure_raises_when_failed_is_not_wired(
        self, real_graph: ExtractFrames, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._stub_until_the_folder(real_graph, monkeypatch)
        real_graph._has_outgoing_connections = lambda _param: False  # type: ignore[method-assign]

        with pytest.raises(ValueError, match="has no destination set"):
            next(real_graph.process())

        assert real_graph._execution_succeeded is False

    def test_a_settings_failure_routes_to_failed_when_wired(
        self, real_graph: ExtractFrames, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._stub_until_the_folder(real_graph, monkeypatch)
        real_graph._has_outgoing_connections = lambda _param: True  # type: ignore[method-assign]

        with pytest.raises(StopIteration):
            next(real_graph.process())

        assert real_graph._execution_succeeded is False
