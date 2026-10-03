"""Tests that file operation nodes resolve project macro paths like "{outputs}/image.png"."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from griptape_nodes.exe_types.core_types import ParameterList
from griptape_nodes.exe_types.node_types import BaseNode
from griptape_nodes.files.file import FileLoadError
from griptape_nodes.retained_mode.events.project_events import (
    GetPathForMacroRequest,
    GetPathForMacroResultFailure,
    GetPathForMacroResultSuccess,
    PathResolutionFailureReason,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

from griptape_nodes_library.files.copy_files import CopyFiles
from griptape_nodes_library.files.delete_file import DeleteFile
from griptape_nodes_library.files.file_exists import FileExists
from griptape_nodes_library.files.move_files import MoveFiles
from griptape_nodes_library.files.rename_file import RenameFile
from griptape_nodes_library.utils.macro_path_utils import resolve_macro_path


def _set_list(node: BaseNode, parameter_name: str, values: list[str]) -> None:
    parameter_list = node.get_parameter_by_name(parameter_name)
    assert isinstance(parameter_list, ParameterList)
    for value in values:
        child = parameter_list.add_child_parameter()
        node.set_parameter_value(child.name, value)


@pytest.fixture
def outputs_dir(tmp_path: Path) -> Path:
    path = tmp_path / "outputs"
    path.mkdir()
    return path


@pytest.fixture
def stub_project(monkeypatch: pytest.MonkeyPatch, outputs_dir: Path) -> None:
    """Resolve "{outputs}" to outputs_dir, passing every other request to the real engine."""
    real_handle_request = GriptapeNodes.handle_request

    def handle_request(request: Any) -> Any:
        if isinstance(request, GetPathForMacroRequest):
            resolved = Path(request.parsed_macro.template.replace("{outputs}", str(outputs_dir)))
            return GetPathForMacroResultSuccess(result_details="", resolved_path=resolved, absolute_path=resolved)
        return real_handle_request(request)

    monkeypatch.setattr(GriptapeNodes, "handle_request", staticmethod(handle_request))


@pytest.fixture
def stub_no_project(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail every macro resolution, as the engine does when no project is loaded."""
    real_handle_request = GriptapeNodes.handle_request

    def handle_request(request: Any) -> Any:
        if isinstance(request, GetPathForMacroRequest):
            return GetPathForMacroResultFailure(
                result_details="No project loaded",
                failure_reason=PathResolutionFailureReason.MACRO_RESOLUTION_ERROR,
            )
        return real_handle_request(request)

    monkeypatch.setattr(GriptapeNodes, "handle_request", staticmethod(handle_request))


class TestResolveMacroPath:
    @pytest.mark.usefixtures("stub_project")
    def test_resolves_macro_to_absolute_path(self, outputs_dir: Path) -> None:
        assert resolve_macro_path("{outputs}/images/foo.png") == str(outputs_dir / "images" / "foo.png")

    @pytest.mark.usefixtures("stub_project")
    def test_resolves_macro_glob_pattern(self, outputs_dir: Path) -> None:
        assert resolve_macro_path("{outputs}/*.png") == str(outputs_dir / "*.png")

    @pytest.mark.parametrize(
        "path",
        [
            "images/foo.png",
            "/abs/foo.png",
            "/abs/*.png",
            "http://localhost:8124/workspace/foo.png",
            "{unclosed/foo.png",
        ],
    )
    def test_returns_non_macro_path_unchanged(self, path: str) -> None:
        assert resolve_macro_path(path) == path

    @pytest.mark.usefixtures("stub_no_project")
    def test_raises_when_resolution_fails(self) -> None:
        with pytest.raises(FileLoadError, match="No project loaded"):
            resolve_macro_path("{outputs}/foo.png")


class TestFileExists:
    @pytest.fixture
    def node(self, griptape_nodes: GriptapeNodes) -> FileExists:  # noqa: ARG002
        return FileExists("file_exists")

    @pytest.mark.usefixtures("stub_project")
    def test_finds_file_at_macro_path(self, node: FileExists, outputs_dir: Path) -> None:
        (outputs_dir / "foo.png").write_bytes(b"png")
        node.parameter_values["path"] = "{outputs}/foo.png"

        node.process()

        assert node.parameter_output_values["exists"] is True
        assert node.parameter_output_values["is_directory"] is False

    @pytest.mark.usefixtures("stub_project")
    def test_finds_directory_at_macro_path(self, node: FileExists, outputs_dir: Path) -> None:
        (outputs_dir / "images").mkdir()
        node.parameter_values["path"] = "{outputs}/images"

        node.process()

        assert node.parameter_output_values["exists"] is True
        assert node.parameter_output_values["is_directory"] is True

    @pytest.mark.usefixtures("stub_no_project")
    def test_raises_when_macro_cannot_resolve(self, node: FileExists) -> None:
        node.parameter_values["path"] = "{outputs}/foo.png"

        with pytest.raises(ValueError, match="No project loaded"):
            node.process()


class TestCopyFiles:
    @pytest.fixture
    def node(self, griptape_nodes: GriptapeNodes) -> CopyFiles:  # noqa: ARG002
        return CopyFiles("copy_files")

    @pytest.mark.usefixtures("stub_project")
    def test_copies_from_macro_source_to_macro_destination(self, node: CopyFiles, outputs_dir: Path) -> None:
        (outputs_dir / "foo.png").write_bytes(b"png")
        (outputs_dir / "archive").mkdir()
        _set_list(node, "source_paths", ["{outputs}/foo.png"])
        node.parameter_values["destination_path"] = "{outputs}/archive"

        node.process()

        assert node.get_parameter_value("was_successful") is True
        assert (outputs_dir / "archive" / "foo.png").read_bytes() == b"png"
        assert (outputs_dir / "foo.png").exists()

    @pytest.mark.usefixtures("stub_project")
    def test_copies_macro_glob_matches(self, node: CopyFiles, outputs_dir: Path, tmp_path: Path) -> None:
        (outputs_dir / "a.png").write_bytes(b"a")
        (outputs_dir / "b.png").write_bytes(b"b")
        (outputs_dir / "c.txt").write_bytes(b"c")
        destination = tmp_path / "dest"
        destination.mkdir()
        _set_list(node, "source_paths", ["{outputs}/*.png"])
        node.parameter_values["destination_path"] = str(destination)

        node.process()

        assert node.get_parameter_value("was_successful") is True
        assert sorted(p.name for p in destination.iterdir()) == ["a.png", "b.png"]

    @pytest.mark.usefixtures("stub_no_project")
    def test_fails_when_source_macro_cannot_resolve(self, node: CopyFiles, tmp_path: Path) -> None:
        _set_list(node, "source_paths", ["{outputs}/foo.png"])
        node.parameter_values["destination_path"] = str(tmp_path)

        node.process()

        assert node.get_parameter_value("was_successful") is False
        assert "No project loaded" in node.get_parameter_value("result_details")

    @pytest.mark.usefixtures("stub_no_project")
    def test_fails_when_destination_macro_cannot_resolve(self, node: CopyFiles, tmp_path: Path) -> None:
        source = tmp_path / "foo.png"
        source.write_bytes(b"png")
        _set_list(node, "source_paths", [str(source)])
        node.parameter_values["destination_path"] = "{outputs}/archive"

        node.process()

        assert node.get_parameter_value("was_successful") is False
        assert "No project loaded" in node.get_parameter_value("result_details")
        assert not (tmp_path / "{outputs}").exists()


class TestMoveFiles:
    @pytest.fixture
    def node(self, griptape_nodes: GriptapeNodes) -> MoveFiles:  # noqa: ARG002
        return MoveFiles("move_files")

    @pytest.mark.usefixtures("stub_project")
    def test_moves_from_macro_source_to_macro_destination(self, node: MoveFiles, outputs_dir: Path) -> None:
        (outputs_dir / "foo.png").write_bytes(b"png")
        (outputs_dir / "archive").mkdir()
        _set_list(node, "source_paths", ["{outputs}/foo.png"])
        node.parameter_values["destination_path"] = "{outputs}/archive"

        node.process()

        assert node.get_parameter_value("was_successful") is True
        assert (outputs_dir / "archive" / "foo.png").read_bytes() == b"png"
        assert not (outputs_dir / "foo.png").exists()

    @pytest.mark.usefixtures("stub_project")
    def test_overwrite_replaces_file_at_macro_destination(self, node: MoveFiles, outputs_dir: Path) -> None:
        (outputs_dir / "foo.png").write_bytes(b"new")
        (outputs_dir / "archive").mkdir()
        (outputs_dir / "archive" / "foo.png").write_bytes(b"old")
        _set_list(node, "source_paths", ["{outputs}/foo.png"])
        node.parameter_values["destination_path"] = "{outputs}/archive"
        node.parameter_values["overwrite"] = True

        node.process()

        assert node.get_parameter_value("was_successful") is True
        assert (outputs_dir / "archive" / "foo.png").read_bytes() == b"new"


class TestRenameFile:
    @pytest.fixture
    def node(self, griptape_nodes: GriptapeNodes) -> RenameFile:  # noqa: ARG002
        return RenameFile("rename_file")

    @pytest.mark.usefixtures("stub_project")
    def test_renames_macro_path_to_new_filename(self, node: RenameFile, outputs_dir: Path) -> None:
        (outputs_dir / "foo.png").write_bytes(b"png")
        node.parameter_values["old_path"] = "{outputs}/foo.png"
        node.parameter_values["new_path"] = "bar.png"

        node.process()

        assert node.get_parameter_value("was_successful") is True
        assert (outputs_dir / "bar.png").read_bytes() == b"png"
        assert not (outputs_dir / "foo.png").exists()

    @pytest.mark.usefixtures("stub_project")
    def test_overwrite_replaces_file_at_macro_new_path(self, node: RenameFile, outputs_dir: Path) -> None:
        (outputs_dir / "foo.png").write_bytes(b"new")
        (outputs_dir / "bar.png").write_bytes(b"old")
        node.parameter_values["old_path"] = "{outputs}/foo.png"
        node.parameter_values["new_path"] = "{outputs}/bar.png"
        node.parameter_values["overwrite"] = True

        node.process()

        assert node.get_parameter_value("was_successful") is True
        assert (outputs_dir / "bar.png").read_bytes() == b"new"
        assert not (outputs_dir / "foo.png").exists()

    @pytest.mark.usefixtures("stub_no_project")
    def test_fails_when_macro_cannot_resolve(self, node: RenameFile) -> None:
        node.parameter_values["old_path"] = "{outputs}/foo.png"
        node.parameter_values["new_path"] = "bar.png"

        node.process()

        assert node.get_parameter_value("was_successful") is False
        assert "No project loaded" in node.get_parameter_value("result_details")


class TestDeleteFile:
    @pytest.fixture
    def node(self, griptape_nodes: GriptapeNodes) -> DeleteFile:  # noqa: ARG002
        return DeleteFile("delete_file")

    @pytest.mark.usefixtures("stub_project")
    def test_deletes_file_at_macro_path(self, node: DeleteFile, outputs_dir: Path) -> None:
        (outputs_dir / "foo.png").write_bytes(b"png")
        (outputs_dir / "keep.png").write_bytes(b"png")
        _set_list(node, "file_paths", ["{outputs}/foo.png"])

        node.process()

        assert node.get_parameter_value("was_successful") is True
        assert not (outputs_dir / "foo.png").exists()
        assert (outputs_dir / "keep.png").exists()

    @pytest.mark.usefixtures("stub_project")
    def test_deletes_macro_glob_matches(self, node: DeleteFile, outputs_dir: Path) -> None:
        (outputs_dir / "a.png").write_bytes(b"a")
        (outputs_dir / "b.png").write_bytes(b"b")
        (outputs_dir / "c.txt").write_bytes(b"c")
        _set_list(node, "file_paths", ["{outputs}/*.png"])

        node.process()

        assert node.get_parameter_value("was_successful") is True
        assert sorted(p.name for p in outputs_dir.iterdir()) == ["c.txt"]

    @pytest.mark.usefixtures("stub_no_project")
    def test_fails_without_deleting_when_macro_cannot_resolve(self, node: DeleteFile) -> None:
        _set_list(node, "file_paths", ["{outputs}/foo.png"])

        node.process()

        assert node.get_parameter_value("was_successful") is False
        assert "No project loaded" in node.get_parameter_value("result_details")
