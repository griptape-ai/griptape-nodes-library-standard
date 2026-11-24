"""Unit tests for ResolveMacroPath and ResolveMacroPaths nodes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
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

from griptape_nodes_library.files.resolve_macro_path import ResolveMacroPath
from griptape_nodes_library.files.resolve_macro_path_base import BaseResolveMacroPath
from griptape_nodes_library.files.resolve_macro_paths import ResolveMacroPaths

_RELATIVE = "inputs/my_file.png"
_ABSOLUTE = "/workspace/inputs/my_file.png"

_UNCHANGED_PATHS = [
    "/abs/file.txt",
    "https://example.com/a.png",
    "notes {draft}.txt",
]


@pytest.fixture(autouse=True)
def _stub_extract(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pass string values through _extract_path_string unchanged, avoiding OSManager."""
    monkeypatch.setattr(
        BaseResolveMacroPath,
        "_extract_path_string",
        lambda self, v: str(v) if v else None,
    )


def _macro_state(request: GetStateForMacroRequest) -> GetStateForMacroResultSuccess:
    """Report a macro as resolvable only when every variable in it is "inputs"."""
    names = {variable.name for variable in request.parsed_macro.get_variables()}
    return GetStateForMacroResultSuccess(
        result_details="",
        all_variables=set(),
        satisfied_variables=names & {"inputs"},
        missing_required_variables=names - {"inputs"},
        conflicting_variables=set(),
        can_resolve=names <= {"inputs"},
    )


def _stub_project(monkeypatch: pytest.MonkeyPatch, path_result: Any) -> None:
    """Resolve "{inputs}" macros to ``path_result``, passing other requests to the real engine."""
    real_handle_request = GriptapeNodes.handle_request

    def handle_request(request: Any) -> Any:
        if isinstance(request, GetStateForMacroRequest):
            return _macro_state(request)
        if isinstance(request, GetPathForMacroRequest):
            return path_result
        return real_handle_request(request)

    monkeypatch.setattr(GriptapeNodes, "handle_request", staticmethod(handle_request))


def _stub_success(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_project(
        monkeypatch,
        GetPathForMacroResultSuccess(result_details="", resolved_path=Path(_RELATIVE), absolute_path=Path(_ABSOLUTE)),
    )


def _stub_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_project(
        monkeypatch,
        GetPathForMacroResultFailure(
            result_details="Resolution failed",
            failure_reason=PathResolutionFailureReason.MACRO_RESOLUTION_ERROR,
        ),
    )


# ── ResolveMacroPath ──────────────────────────────────────────────────────────


class TestResolveMacroPathDefaults:
    def test_path_default_is_empty(self, griptape_nodes: GriptapeNodes) -> None:  # noqa: ARG002
        node = ResolveMacroPath("test")
        assert node.get_parameter_value("path") == ""

    def test_resolved_path_default_is_empty(self, griptape_nodes: GriptapeNodes) -> None:  # noqa: ARG002
        node = ResolveMacroPath("test")
        assert node.get_parameter_value("resolved_path") == ""


class TestResolveMacroPathProcess:
    @pytest.fixture
    def node(self, griptape_nodes: GriptapeNodes) -> ResolveMacroPath:  # noqa: ARG002
        return ResolveMacroPath("test")

    def test_returns_absolute_path_not_relative(self, node: ResolveMacroPath, monkeypatch: pytest.MonkeyPatch) -> None:
        """Regression: output must come from absolute_path, not resolved_path."""
        _stub_success(monkeypatch)
        node.parameter_values["path"] = "{inputs}/my_file.png"
        node.process()
        assert node.parameter_output_values["resolved_path"] == _ABSOLUTE

    @pytest.mark.parametrize("path", _UNCHANGED_PATHS)
    def test_non_macro_path_comes_back_unchanged(
        self, node: ResolveMacroPath, monkeypatch: pytest.MonkeyPatch, path: str
    ) -> None:
        _stub_success(monkeypatch)
        node.parameter_values["path"] = path
        node.process()
        assert node.parameter_output_values["resolved_path"] == path
        assert node.parameter_output_values["was_successful"] is True

    def test_relative_path_is_anchored_to_workspace(
        self, node: ResolveMacroPath, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_success(monkeypatch)
        node.parameter_values["path"] = "relative/x.txt"
        node.process()
        workspace = Path(GriptapeNodes.ConfigManager().workspace_path).resolve()
        assert Path(node.parameter_output_values["resolved_path"]) == workspace / "relative" / "x.txt"

    def test_empty_path_yields_empty_string(self, node: ResolveMacroPath) -> None:
        node.parameter_values["path"] = ""
        node.process()
        assert node.parameter_output_values["resolved_path"] == ""

    def test_success_sets_was_successful(self, node: ResolveMacroPath, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_success(monkeypatch)
        node.parameter_values["path"] = "{inputs}/my_file.png"
        node.process()
        assert node.parameter_output_values["was_successful"] is True

    def test_failure_clears_resolved_path(self, node: ResolveMacroPath, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_failure(monkeypatch)
        node.parameter_values["path"] = "{inputs}/file.png"
        with pytest.raises(FileLoadError):
            node.process()
        assert node.parameter_output_values["resolved_path"] == ""

    def test_failure_sets_was_not_successful(self, node: ResolveMacroPath, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_failure(monkeypatch)
        node.parameter_values["path"] = "{inputs}/file.png"
        with pytest.raises(FileLoadError):
            node.process()
        assert node.parameter_output_values["was_successful"] is False


# ── ResolveMacroPaths ─────────────────────────────────────────────────────────


class TestResolveMacroPathsDefaults:
    def test_paths_default_is_empty_list(self, griptape_nodes: GriptapeNodes) -> None:  # noqa: ARG002
        node = ResolveMacroPaths("test")
        assert node.get_parameter_value("paths") == []

    def test_resolved_paths_default_is_empty_list(self, griptape_nodes: GriptapeNodes) -> None:  # noqa: ARG002
        node = ResolveMacroPaths("test")
        assert node.get_parameter_value("resolved_paths") == []


class TestResolveMacroPathsProcess:
    @pytest.fixture
    def node(self, griptape_nodes: GriptapeNodes, monkeypatch: pytest.MonkeyPatch) -> ResolveMacroPaths:  # noqa: ARG002
        n = ResolveMacroPaths("test")
        # ParameterList uses get_parameter_list_value; shim it to read parameter_values directly.
        monkeypatch.setattr(n, "get_parameter_list_value", lambda name: n.parameter_values.get(name, []))
        return n

    def test_returns_absolute_path_not_relative(self, node: ResolveMacroPaths, monkeypatch: pytest.MonkeyPatch) -> None:
        """Regression: output must come from absolute_path, not resolved_path."""
        _stub_success(monkeypatch)
        node.parameter_values["paths"] = ["{inputs}/my_file.png"]
        node.process()
        assert node.parameter_output_values["resolved_paths"] == [_ABSOLUTE]

    def test_non_macro_paths_come_back_unchanged(
        self, node: ResolveMacroPaths, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_success(monkeypatch)
        node.parameter_values["paths"] = ["{inputs}/my_file.png", *_UNCHANGED_PATHS]
        node.process()
        assert node.parameter_output_values["resolved_paths"] == [_ABSOLUTE, *_UNCHANGED_PATHS]
        assert node.parameter_output_values["was_successful"] is True

    def test_empty_list_yields_empty_output(self, node: ResolveMacroPaths) -> None:
        node.parameter_values["paths"] = []
        node.process()
        assert node.parameter_output_values["resolved_paths"] == []

    def test_resolves_multiple_paths(self, node: ResolveMacroPaths, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_success(monkeypatch)
        node.parameter_values["paths"] = ["{inputs}/a.png", "{inputs}/b.png"]
        node.process()
        assert node.parameter_output_values["resolved_paths"] == [_ABSOLUTE, _ABSOLUTE]

    def test_success_reports_count_in_details(self, node: ResolveMacroPaths, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_success(monkeypatch)
        node.parameter_values["paths"] = ["{inputs}/a.png", "{inputs}/b.png"]
        node.process()
        assert node.parameter_output_values["was_successful"] is True
        assert "2" in node.parameter_output_values["result_details"]

    def test_failure_clears_resolved_paths(self, node: ResolveMacroPaths, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_failure(monkeypatch)
        node.parameter_values["paths"] = ["{inputs}/file.png"]
        with pytest.raises(FileLoadError):
            node.process()
        assert node.parameter_output_values["resolved_paths"] == []
