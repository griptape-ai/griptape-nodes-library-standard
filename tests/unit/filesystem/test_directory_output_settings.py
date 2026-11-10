"""Tests for DirectoryOutputSettings.directory_destination across its value shapes."""

from pathlib import Path

import pytest
from griptape_nodes.files.directory import DirectoryDestination
from griptape_nodes.retained_mode.events.os_events import ExistingFilePolicy
from griptape_nodes.retained_mode.events.project_events import MacroPath
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

from griptape_nodes_library.filesystem.directory_output_settings import DirectoryOutputSettings

_DEFAULT_MACRO = "{outputs}/{sub_dirs?:/}{dir_name}_v{###}"


@pytest.fixture
def node(griptape_nodes: GriptapeNodes) -> DirectoryOutputSettings:  # noqa: ARG001
    return DirectoryOutputSettings(name="test_directory_output_settings")


def test_macro_and_policy_default_from_the_save_output_directory_situation(node: DirectoryOutputSettings) -> None:
    assert node.get_parameter_value("macro") == _DEFAULT_MACRO
    assert node.get_parameter_value("if_dir_exists") == "Increment Version"
    assert node.get_parameter_value("auto_create_path") is True


def test_empty_dirname_has_no_destination(node: DirectoryOutputSettings) -> None:
    assert node.directory_destination is None


class TestExplicitLocalPath:
    """An absolute path, a `~` path, or a `file://` URI names an exact location, not versioned."""

    def test_absolute_path_is_used_verbatim_and_overwrites(self, node: DirectoryOutputSettings, tmp_path: Path) -> None:
        node.set_parameter_value("dirname", str(tmp_path / "renders"))

        destination = node.directory_destination

        assert isinstance(destination, DirectoryDestination)
        assert destination.resolve() == str(tmp_path / "renders")
        assert destination._existing_dir_policy == ExistingFilePolicy.OVERWRITE

    def test_home_path_is_expanded_and_overwrites(
        self, node: DirectoryOutputSettings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        node.set_parameter_value("dirname", "~/renders")

        destination = node.directory_destination

        assert destination is not None
        assert destination.resolve() == str(tmp_path / "renders")
        assert destination._existing_dir_policy == ExistingFilePolicy.OVERWRITE

    def test_file_uri_is_used_as_a_local_path_and_overwrites(
        self, node: DirectoryOutputSettings, tmp_path: Path
    ) -> None:
        node.set_parameter_value("dirname", f"file://{tmp_path / 'renders'}")

        destination = node.directory_destination

        assert destination is not None
        assert destination.resolve() == str(tmp_path / "renders")
        assert destination._existing_dir_policy == ExistingFilePolicy.OVERWRITE

    def test_ignores_if_dir_exists(self, node: DirectoryOutputSettings, tmp_path: Path) -> None:
        """An explicit location bypasses the collision policy: the user chose this exact folder."""
        node.set_parameter_value("if_dir_exists", "Abort / Error")
        node.set_parameter_value("dirname", str(tmp_path / "renders"))

        destination = node.directory_destination
        assert destination is not None
        assert destination._existing_dir_policy == ExistingFilePolicy.OVERWRITE


class TestFullMacroTemplate:
    """A dirname containing a macro variable is used as the full template."""

    def test_used_verbatim_with_the_configured_policy(self, node: DirectoryOutputSettings) -> None:
        node.set_parameter_value("dirname", "{outputs}/frames_v{###}")

        destination = node.directory_destination

        assert destination is not None
        assert destination.location == "{outputs}/frames_v{###}"
        assert destination._existing_dir_policy == ExistingFilePolicy.CREATE_NEW

    def test_policy_follows_if_dir_exists(self, node: DirectoryOutputSettings) -> None:
        node.set_parameter_value("if_dir_exists", "Overwrite Existing")
        node.set_parameter_value("dirname", "{outputs}/frames_v{###}")

        destination = node.directory_destination

        assert destination is not None
        assert destination._existing_dir_policy == ExistingFilePolicy.OVERWRITE


class TestPlainRelativeName:
    """A plain folder name nests under the save_output_directory situation's macro."""

    def test_nests_under_the_situation_macro_with_sub_dirs(self, node: DirectoryOutputSettings) -> None:
        node.set_parameter_value("dirname", "renders/shot01")

        destination = node.directory_destination

        assert destination is not None
        dir_path = destination._directory._dir_path
        assert isinstance(dir_path, MacroPath)
        assert dir_path.parsed_macro.template == _DEFAULT_MACRO
        assert dir_path.variables == {"dir_name": "shot01", "sub_dirs": "renders"}
        assert destination._existing_dir_policy == ExistingFilePolicy.CREATE_NEW

    def test_without_a_parent_has_no_sub_dirs(self, node: DirectoryOutputSettings) -> None:
        node.set_parameter_value("dirname", "shot01")

        destination = node.directory_destination

        assert destination is not None
        dir_path = destination._directory._dir_path
        assert isinstance(dir_path, MacroPath)
        assert dir_path.variables == {"dir_name": "shot01"}

    def test_with_no_macro_available_has_no_destination(self, node: DirectoryOutputSettings) -> None:
        node.set_parameter_value("macro", "")
        node.set_parameter_value("dirname", "renders")

        assert node.directory_destination is None
