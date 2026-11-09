"""Tests for how the shared 'directory' parameter turns what the user typed into a folder."""

from pathlib import Path

import pytest
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

from griptape_nodes_library.files.extract_pdf_images import DEFAULT_DIRECTORY, ExtractPdfImages


@pytest.fixture
def node(griptape_nodes: GriptapeNodes) -> ExtractPdfImages:  # noqa: ARG001
    return ExtractPdfImages(name="test_directory")


def test_absolute_path_is_used_as_typed(node: ExtractPdfImages, tmp_path: Path) -> None:
    node.parameter_values["directory"] = str(tmp_path / "renders")

    assert node._resolve_output_dir() == tmp_path / "renders"


def test_absolute_path_with_version_token_takes_the_next_free_version(node: ExtractPdfImages, tmp_path: Path) -> None:
    (tmp_path / "scan_v001").mkdir()
    (tmp_path / "scan_v002").mkdir()
    node.parameter_values["directory"] = str(tmp_path / "scan_v{###}")

    assert node._resolve_output_dir() == tmp_path / "scan_v003"


def test_home_folder_is_expanded_instead_of_nested_under_outputs(
    node: ExtractPdfImages, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    node.parameter_values["directory"] = "~/renders_v{###}"

    assert node._resolve_output_dir() == tmp_path / "renders_v001"


def test_relative_name_goes_under_outputs(node: ExtractPdfImages) -> None:
    node.parameter_values["directory"] = "my_scans"

    resolved = node._resolve_output_dir()

    assert resolved.name == "my_scans"
    assert resolved.parent.name == "outputs"


def test_default_directory_is_versioned_under_outputs(node: ExtractPdfImages) -> None:
    node.parameter_values["directory"] = DEFAULT_DIRECTORY

    resolved = node._resolve_output_dir()

    assert resolved.parent.name == "outputs"
    assert resolved.name.startswith("extracted_v")
