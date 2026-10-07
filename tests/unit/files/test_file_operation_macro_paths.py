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
    GetStateForMacroRequest,
    GetStateForMacroResultSuccess,
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


def _macro_state(request: GetStateForMacroRequest, *, known: set[str]) -> GetStateForMacroResultSuccess:
    """Report a macro as resolvable only when every variable in it is in ``known``."""
    names = {variable.name for variable in request.parsed_macro.get_variables()}
    return GetStateForMacroResultSuccess(
        result_details="",
        all_variables=set(),
        satisfied_variables=names & known,
        missing_required_variables=names - known,
        conflicting_variables=set(),
        can_resolve=names <= known,
    )


@pytest.fixture
def stub_project(monkeypatch: pytest.MonkeyPatch, outputs_dir: Path) -> None:
    """Resolve "{outputs}" to outputs_dir, passing every other request to the real engine."""
    real_handle_request = GriptapeNodes.handle_request

    def handle_request(request: Any) -> Any:
        if isinstance(request, GetStateForMacroRequest):
            return _macro_state(request, known={"outputs"})
        if isinstance(request, GetPathForMacroRequest):
            resolved = Path(request.parsed_macro.template.replace("{outputs}", str(outputs_dir)))
            return GetPathForMacroResultSuccess(result_details="", resolved_path=resolved, absolute_path=resolved)
        return real_handle_request(request)

    monkeypatch.setattr(GriptapeNodes, "handle_request", staticmethod(handle_request))


@pytest.fixture
def stub_resolution_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Report "{outputs}" as resolvable, then fail the resolution itself."""
    real_handle_request = GriptapeNodes.handle_request

    def handle_request(request: Any) -> Any:
        if isinstance(request, GetStateForMacroRequest):
            return _macro_state(request, known={"outputs"})
        if isinstance(request, GetPathForMacroRequest):
            return GetPathForMacroResultFailure(
                result_details="Resolution failed",
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

    @pytest.mark.usefixtures("stub_resolution_failure")
    def test_raises_when_resolution_fails(self) -> None:
        with pytest.raises(FileLoadError, match="Resolution failed"):
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

    @pytest.mark.usefixtures("stub_resolution_failure")
    def test_raises_when_macro_cannot_resolve(self, node: FileExists) -> None:
        node.parameter_values["path"] = "{outputs}/foo.png"

        with pytest.raises(ValueError, match="Resolution failed"):
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

    @pytest.mark.usefixtures("stub_resolution_failure")
    def test_fails_when_source_macro_cannot_resolve(self, node: CopyFiles, tmp_path: Path) -> None:
        _set_list(node, "source_paths", ["{outputs}/foo.png"])
        node.parameter_values["destination_path"] = str(tmp_path)

        with pytest.raises(ValueError, match="Resolution failed"):
            node.process()

        assert node.get_parameter_value("was_successful") is False
        assert "Resolution failed" in node.get_parameter_value("result_details")

    @pytest.mark.usefixtures("stub_resolution_failure")
    def test_fails_when_destination_macro_cannot_resolve(self, node: CopyFiles, tmp_path: Path) -> None:
        source = tmp_path / "foo.png"
        source.write_bytes(b"png")
        _set_list(node, "source_paths", [str(source)])
        node.parameter_values["destination_path"] = "{outputs}/archive"

        with pytest.raises(ValueError, match="Resolution failed"):
            node.process()

        assert node.get_parameter_value("was_successful") is False
        assert "Resolution failed" in node.get_parameter_value("result_details")
        assert not (tmp_path / "{outputs}").exists()

    def test_failure_routes_to_failed_output_when_wired(self, node: CopyFiles, tmp_path: Path) -> None:
        """With Failed wired, a copy where nothing succeeded reports per-file details instead of raising."""
        _set_list(node, "source_paths", [str(tmp_path / "missing.png")])
        node.parameter_values["destination_path"] = str(tmp_path / "dest")
        node._has_outgoing_connections = lambda _param: True  # type: ignore[method-assign]

        node.process()

        assert node.get_parameter_value("was_successful") is False
        details = node.get_parameter_value("result_details")
        assert "All source paths were invalid" in details
        assert "missing.png" in details


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

    def test_raises_when_destination_is_empty(self, node: MoveFiles, tmp_path: Path) -> None:
        source = tmp_path / "foo.png"
        source.write_bytes(b"png")
        _set_list(node, "source_paths", [str(source)])
        node.parameter_values["destination_path"] = ""
        node._has_outgoing_connections = lambda _param: False  # type: ignore[method-assign]

        with pytest.raises(ValueError, match="'destination_path' is empty"):
            node.process()

        assert node.get_parameter_value("was_successful") is False
        assert source.exists()


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

    @pytest.mark.usefixtures("stub_resolution_failure")
    def test_fails_when_macro_cannot_resolve(self, node: RenameFile) -> None:
        node.parameter_values["old_path"] = "{outputs}/foo.png"
        node.parameter_values["new_path"] = "bar.png"

        with pytest.raises(ValueError, match="Resolution failed"):
            node.process()

        assert node.get_parameter_value("was_successful") is False
        assert "Resolution failed" in node.get_parameter_value("result_details")


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

    def test_raises_with_per_file_details_when_nothing_is_deleted(self, node: DeleteFile, tmp_path: Path) -> None:
        _set_list(node, "file_paths", [str(tmp_path / "missing.png")])
        node._has_outgoing_connections = lambda _param: False  # type: ignore[method-assign]

        with pytest.raises(ValueError, match="All paths were invalid") as exc_info:
            node.process()

        assert "missing.png" in str(exc_info.value)
        assert node.get_parameter_value("was_successful") is False

    @pytest.mark.usefixtures("stub_resolution_failure")
    def test_fails_without_deleting_when_macro_cannot_resolve(self, node: DeleteFile) -> None:
        _set_list(node, "file_paths", ["{outputs}/foo.png"])

        with pytest.raises(ValueError, match="Resolution failed"):
            node.process()

        assert node.get_parameter_value("was_successful") is False
        assert "Resolution failed" in node.get_parameter_value("result_details")


# ---------------------------------------------------------------------------
# Braces in real filenames
#
# These run against the real engine and its default project. Braces are legal in
# filenames, so only variables the project defines may be treated as macros.
# ---------------------------------------------------------------------------

ENV_VAR = "GTN_FILE_OPS_TEST_VAR"


@pytest.fixture
def env_var(monkeypatch: pytest.MonkeyPatch) -> str:
    """A shell environment variable the engine would substitute into a macro."""
    monkeypatch.setenv(ENV_VAR, "substituted")
    return ENV_VAR


class TestBracesInFilenames:
    def test_existing_brace_filename_is_returned_unchanged(self, tmp_path: Path) -> None:
        path = tmp_path / "notes {draft}.txt"
        path.write_text("draft")

        assert resolve_macro_path(str(path)) == str(path)

    @pytest.mark.parametrize("name", ["report {v2}.txt", "{ouputs}/foo.png"], ids=["new_file", "typo"])
    def test_unknown_variable_is_returned_unchanged(self, name: str) -> None:
        assert resolve_macro_path(name) == name

    def test_shell_environment_variable_is_not_substituted(self, tmp_path: Path, env_var: str) -> None:
        path = str(tmp_path / f"{{{env_var}}}.txt")

        assert resolve_macro_path(path) == path

    def test_project_variable_still_resolves(self) -> None:
        assert "{outputs}" not in resolve_macro_path("{outputs}/foo.png")

    def test_file_exists_finds_brace_filename(self, griptape_nodes: GriptapeNodes, tmp_path: Path) -> None:  # noqa: ARG002
        path = tmp_path / "notes {draft}.txt"
        path.write_text("draft")
        node = FileExists("file_exists")
        node.parameter_values["path"] = str(path)

        node.process()

        assert node.parameter_output_values["exists"] is True

    def test_delete_file_deletes_brace_filename(self, griptape_nodes: GriptapeNodes, tmp_path: Path) -> None:  # noqa: ARG002
        path = tmp_path / "notes {draft}.txt"
        path.write_text("draft")
        node = DeleteFile("delete_file")
        _set_list(node, "file_paths", [str(path)])

        node.process()

        assert node.get_parameter_value("was_successful") is True
        assert not path.exists()

    def test_delete_file_never_targets_the_env_substituted_file(
        self,
        griptape_nodes: GriptapeNodes,  # noqa: ARG002
        tmp_path: Path,
        env_var: str,
    ) -> None:
        """The engine would turn "{VAR}.txt" into "substituted.txt" and delete that instead."""
        other_file = tmp_path / "substituted.txt"
        other_file.write_text("keep me")
        node = DeleteFile("delete_file")
        _set_list(node, "file_paths", [str(tmp_path / f"{{{env_var}}}.txt")])

        with pytest.raises(ValueError, match="No files were deleted|All paths were invalid"):
            node.process()

        assert other_file.read_text() == "keep me"

    def test_rename_file_to_brace_filename(self, griptape_nodes: GriptapeNodes, tmp_path: Path) -> None:  # noqa: ARG002
        old_path = tmp_path / "report.txt"
        old_path.write_text("report")
        node = RenameFile("rename_file")
        node.parameter_values["old_path"] = str(old_path)
        node.parameter_values["new_path"] = "report {v2}.txt"

        node.process()

        assert node.get_parameter_value("was_successful") is True
        assert (tmp_path / "report {v2}.txt").read_text() == "report"
        assert not old_path.exists()

    def test_rename_brace_filename(self, griptape_nodes: GriptapeNodes, tmp_path: Path) -> None:  # noqa: ARG002
        old_path = tmp_path / "notes {draft}.txt"
        old_path.write_text("draft")
        node = RenameFile("rename_file")
        node.parameter_values["old_path"] = str(old_path)
        node.parameter_values["new_path"] = "notes.txt"

        node.process()

        assert node.get_parameter_value("was_successful") is True
        assert (tmp_path / "notes.txt").read_text() == "draft"


# ---------------------------------------------------------------------------
# Builtins the project recognizes but can't fill
#
# With no workflow open, the engine knows {workflow_name} but can't supply it. That's
# not a filename, so it must fail rather than be used as a literal path.
# ---------------------------------------------------------------------------


class TestUnavailableBuiltins:
    def test_resolve_raises_for_unavailable_builtin(self) -> None:
        with pytest.raises(FileLoadError, match="workflow_name"):
            resolve_macro_path("{outputs}/{workflow_name}")

    def test_copy_fails_instead_of_writing_to_literal_folder(
        self,
        griptape_nodes: GriptapeNodes,  # noqa: ARG002
        tmp_path: Path,
    ) -> None:
        source = tmp_path / "foo.png"
        source.write_bytes(b"png")
        node = CopyFiles("copy_files")
        _set_list(node, "source_paths", [str(source)])
        node.parameter_values["destination_path"] = "{outputs}/{workflow_name}"

        with pytest.raises(ValueError, match="workflow_name"):
            node.process()

        assert node.get_parameter_value("was_successful") is False
        assert "workflow_name" in node.get_parameter_value("result_details")
