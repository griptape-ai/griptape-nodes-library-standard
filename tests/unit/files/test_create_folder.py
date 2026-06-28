"""Tests for CreateFolder node."""

from pathlib import Path

import pytest
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

from griptape_nodes_library.files.create_folder import CreateFolder


class TestCreateFolderProcess:
    """Tests for CreateFolder.process()."""

    @pytest.fixture
    def node(self, griptape_nodes: GriptapeNodes) -> CreateFolder:  # noqa: ARG002
        return CreateFolder(name="test_create_folder")

    def test_creates_new_directory(self, node: CreateFolder, tmp_path: Path) -> None:
        target_dir = tmp_path / "new_folder"
        node.parameter_values["folder_path"] = str(target_dir)
        node.parameter_values["create_parents"] = True
        node.parameter_values["fail_if_already_exists"] = False

        node.process()

        assert target_dir.exists()
        assert target_dir.is_dir()
        assert node.parameter_output_values["created_path"] == str(target_dir)
        assert node.parameter_output_values["already_existed"] is False

    def test_existing_directory_succeeds_when_fail_flag_false(self, node: CreateFolder, tmp_path: Path) -> None:
        target_dir = tmp_path / "existing_folder"
        target_dir.mkdir()
        node.parameter_values["folder_path"] = str(target_dir)
        node.parameter_values["fail_if_already_exists"] = False

        node.process()

        assert node.parameter_output_values["created_path"] == str(target_dir)
        assert node.parameter_output_values["already_existed"] is True

    def test_existing_directory_fails_when_fail_flag_true(self, node: CreateFolder, tmp_path: Path) -> None:
        target_dir = tmp_path / "existing_folder"
        target_dir.mkdir()
        node.parameter_values["folder_path"] = str(target_dir)
        node.parameter_values["fail_if_already_exists"] = True

        node.process()

        assert node.get_parameter_value("created_path") == ""
        assert node.get_parameter_value("already_existed") is False

    def test_fails_when_target_is_existing_file(self, node: CreateFolder, tmp_path: Path) -> None:
        target_file = tmp_path / "existing_file.txt"
        target_file.write_text("content")
        node.parameter_values["folder_path"] = str(target_file)

        node.process()

        assert target_file.exists()
        assert target_file.is_file()
        assert node.get_parameter_value("created_path") == ""
        assert node.get_parameter_value("already_existed") is False

    def test_fails_when_path_is_empty(self, node: CreateFolder) -> None:
        node.parameter_values["folder_path"] = ""

        node.process()

        assert node.get_parameter_value("created_path") == ""
        assert node.get_parameter_value("already_existed") is False

    def test_create_parents_false_fails_for_missing_parent(self, node: CreateFolder, tmp_path: Path) -> None:
        target_dir = tmp_path / "missing_parent" / "child_folder"
        node.parameter_values["folder_path"] = str(target_dir)
        node.parameter_values["create_parents"] = False
        node.parameter_values["fail_if_already_exists"] = False

        node.process()

        assert not target_dir.exists()
        assert node.get_parameter_value("created_path") == ""
        assert node.get_parameter_value("already_existed") is False
