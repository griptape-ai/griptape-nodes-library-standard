"""Tests that overwriting with Rename File and Move Files never loses the destination when the move fails."""

from __future__ import annotations

import contextlib
from pathlib import Path
from unittest.mock import patch

import pytest
from griptape_nodes.exe_types.core_types import ParameterList
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

from griptape_nodes_library.files.move_files import MoveFiles
from griptape_nodes_library.files.rename_file import RenameFile


def _deny(*_args: object) -> Path:
    msg = "Permission denied"
    raise PermissionError(msg)


def _fail_moves() -> contextlib.AbstractContextManager[dict[str, object]]:
    """Make every rename or replace on disk fail, as it does for a source in a read-only folder."""
    return patch.multiple(Path, rename=_deny, replace=_deny)


class TestRenameFileOverwrite:
    @pytest.fixture
    def node(self, griptape_nodes: GriptapeNodes) -> RenameFile:  # noqa: ARG002
        return RenameFile("rename_file")

    def test_failed_rename_keeps_destination(self, node: RenameFile, tmp_path: Path) -> None:
        (tmp_path / "src.txt").write_text("new")
        (tmp_path / "dest.txt").write_text("PRECIOUS")
        node.parameter_values["old_path"] = str(tmp_path / "src.txt")
        node.parameter_values["new_path"] = str(tmp_path / "dest.txt")
        node.parameter_values["overwrite"] = True

        with _fail_moves(), pytest.raises(OSError, match="permission_denied"):
            node.process()

        assert (tmp_path / "dest.txt").read_text() == "PRECIOUS"
        assert (tmp_path / "src.txt").read_text() == "new"


class TestMoveFilesOverwrite:
    @pytest.fixture
    def node(self, griptape_nodes: GriptapeNodes) -> MoveFiles:  # noqa: ARG002
        return MoveFiles("move_files")

    def test_failed_move_keeps_destination(self, node: MoveFiles, tmp_path: Path) -> None:
        (tmp_path / "s.txt").write_text("new")
        (tmp_path / "out").mkdir()
        (tmp_path / "out" / "s.txt").write_text("PRECIOUS")
        source_paths = node.get_parameter_by_name("source_paths")
        assert isinstance(source_paths, ParameterList)
        node.set_parameter_value(source_paths.add_child_parameter().name, str(tmp_path / "s.txt"))
        node.parameter_values["destination_path"] = str(tmp_path / "out")
        node.parameter_values["overwrite"] = True

        with _fail_moves(), pytest.raises(OSError, match="No files were moved"):
            node.process()

        assert (tmp_path / "out" / "s.txt").read_text() == "PRECIOUS"
        assert (tmp_path / "s.txt").read_text() == "new"

    def test_has_no_copied_paths_output(self, node: MoveFiles) -> None:
        assert node.get_parameter_by_name("copied_paths") is None
        assert node.get_parameter_by_name("moved_paths") is not None
