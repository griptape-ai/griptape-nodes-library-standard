"""Tests for SaveToProject node."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from griptape_nodes.files.file import FileDestination
from griptape_nodes.retained_mode.events.project_events import (
    AttemptMapAbsolutePathToProjectRequest,
    AttemptMapAbsolutePathToProjectResultSuccess,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

from griptape_nodes_library.files.save_to_project import SaveToProject


@pytest.fixture
def node(griptape_nodes: GriptapeNodes) -> SaveToProject:  # noqa: ARG001
    return SaveToProject(name="test_save_to_project")


@pytest.fixture
def stub_project(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Map paths under an "outputs" dir to "{outputs}/...", passing every other request to the real engine."""
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    real_handle_request = GriptapeNodes.handle_request

    def handle_request(request: Any) -> Any:
        if isinstance(request, AttemptMapAbsolutePathToProjectRequest):
            path = Path(request.absolute_path)
            mapped = (
                f"{{outputs}}/{path.relative_to(outputs_dir).as_posix()}" if path.is_relative_to(outputs_dir) else None
            )
            return AttemptMapAbsolutePathToProjectResultSuccess(result_details="", mapped_path=mapped)
        return real_handle_request(request)

    monkeypatch.setattr(GriptapeNodes, "handle_request", staticmethod(handle_request))
    return outputs_dir


def _run(node: SaveToProject, monkeypatch: pytest.MonkeyPatch, source: Path, destination: Path) -> None:
    monkeypatch.setattr(node._file_param, "build_file", lambda: FileDestination(str(destination)))
    node.parameter_values["source"] = str(source)
    asyncio.run(node.aprocess())


class TestSaveToProjectResultDetails:
    def test_reports_macro_path_when_saved_inside_project(
        self, node: SaveToProject, stub_project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        source = tmp_path / "source.txt"
        source.write_text("content")
        destination = stub_project / "saved.txt"

        _run(node, monkeypatch, source, destination)

        assert node.parameter_output_values["saved_url"].value == "{outputs}/saved.txt"
        result_details = node.get_parameter_value("result_details")
        assert "{outputs}/saved.txt" in result_details
        assert str(tmp_path) not in result_details

    def test_reports_absolute_path_when_saved_outside_project(
        self,
        node: SaveToProject,
        stub_project: Path,  # noqa: ARG002
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        source = tmp_path / "source.txt"
        source.write_text("content")
        destination = tmp_path / "elsewhere" / "saved.txt"

        _run(node, monkeypatch, source, destination)

        assert node.parameter_output_values["saved_url"].value == str(destination)
        assert str(destination) in node.get_parameter_value("result_details")
