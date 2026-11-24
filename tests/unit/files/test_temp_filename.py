"""Tests for TempFilename node."""

from pathlib import Path

import pytest
from griptape_nodes.exe_types.core_types import NodeError
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

from griptape_nodes_library.files.temp_filename import TempFilename


class TestTempFilenameProcess:
    """Tests for TempFilename.process()."""

    @pytest.fixture
    def node(self, griptape_nodes: GriptapeNodes) -> TempFilename:  # noqa: ARG002
        return TempFilename(name="test_temp_filename")

    def test_generates_path_in_directory(self, node: TempFilename, tmp_path: Path) -> None:
        node.parameter_values["directory"] = str(tmp_path)
        node.parameter_values["suffix"] = "_backup"
        node.parameter_values["extension"] = "txt"

        node.process()

        result = Path(node.parameter_output_values["filename"])
        assert result.parent == tmp_path.resolve()
        assert result.name.endswith("_backup.txt")
        assert not result.exists()

    def test_missing_directory_fails(self, node: TempFilename, tmp_path: Path) -> None:
        node.parameter_values["directory"] = str(tmp_path / "nope")

        with pytest.raises(NodeError, match="Directory does not exist"):
            node.process()

        assert not node.parameter_output_values.get("filename")

    def test_failure_clears_previous_filename(self, node: TempFilename, tmp_path: Path) -> None:
        directory = tmp_path / "gone"
        directory.mkdir()
        node.parameter_values["directory"] = str(directory)
        node.process()
        assert node.parameter_output_values["filename"]

        directory.rmdir()
        with pytest.raises(NodeError, match="Directory does not exist"):
            node.process()

        assert node.parameter_output_values["filename"] == ""
        assert node.get_parameter_value("filename") == ""

    def test_file_as_directory_fails(self, node: TempFilename, tmp_path: Path) -> None:
        file_path = tmp_path / "file.txt"
        file_path.write_text("content")
        node.parameter_values["directory"] = str(file_path)

        with pytest.raises(NodeError, match="Path is not a directory"):
            node.process()


class TestTempFilenameAfterValueSet:
    """Tests for TempFilename.after_value_set()."""

    def test_missing_directory_clears_output(
        self,
        griptape_nodes: GriptapeNodes,  # noqa: ARG002
        tmp_path: Path,
    ) -> None:
        node = TempFilename(name="test_temp_filename")
        node.set_parameter_value("directory", str(tmp_path))
        assert node.parameter_output_values["filename"]

        node.set_parameter_value("directory", str(tmp_path / "nope"))

        assert node.parameter_output_values["filename"] == ""
