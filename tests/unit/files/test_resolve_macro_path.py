"""Unit tests for ResolveMacroPath and ResolveMacroPaths nodes."""

from __future__ import annotations

from pathlib import Path

import pytest
from griptape_nodes.retained_mode.events.project_events import (
    GetPathForMacroResultFailure,
    GetPathForMacroResultSuccess,
    PathResolutionFailureReason,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

import griptape_nodes_library.files.resolve_macro_path_base as base_module
from griptape_nodes_library.files.resolve_macro_path import ResolveMacroPath
from griptape_nodes_library.files.resolve_macro_path_base import BaseResolveMacroPath
from griptape_nodes_library.files.resolve_macro_paths import ResolveMacroPaths

_RELATIVE = "inputs/my_file.png"
_ABSOLUTE = "/workspace/inputs/my_file.png"


@pytest.fixture(autouse=True)
def _stub_extract(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pass string values through _extract_path_string unchanged, avoiding OSManager."""
    monkeypatch.setattr(
        BaseResolveMacroPath,
        "_extract_path_string",
        lambda self, v: str(v) if v else None,
    )


def _stub_success(
    monkeypatch: pytest.MonkeyPatch,
    absolute: str = _ABSOLUTE,
    relative: str = _RELATIVE,
) -> None:
    result = GetPathForMacroResultSuccess(
        result_details="",
        resolved_path=Path(relative),
        absolute_path=Path(absolute),
    )
    monkeypatch.setattr(
        base_module.GriptapeNodes,
        "handle_request",
        staticmethod(lambda _req: result),
    )


def _stub_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    result = GetPathForMacroResultFailure(
        result_details="",
        failure_reason=PathResolutionFailureReason.MACRO_RESOLUTION_ERROR,
    )
    monkeypatch.setattr(
        base_module.GriptapeNodes,
        "handle_request",
        staticmethod(lambda _req: result),
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
        _stub_success(monkeypatch, absolute=_ABSOLUTE, relative=_RELATIVE)
        node.parameter_values["path"] = "{inputs}/my_file.png"
        node.process()
        assert node.parameter_output_values["resolved_path"] == _ABSOLUTE
        assert node.parameter_output_values["resolved_path"] != _RELATIVE

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
        node.parameter_values["path"] = "{bad}/file.png"
        with pytest.raises(Exception):
            node.process()
        assert node.parameter_output_values["resolved_path"] == ""

    def test_failure_sets_was_not_successful(self, node: ResolveMacroPath, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_failure(monkeypatch)
        node.parameter_values["path"] = "{bad}/file.png"
        with pytest.raises(Exception):
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
        _stub_success(monkeypatch, absolute=_ABSOLUTE, relative=_RELATIVE)
        node.parameter_values["paths"] = ["{inputs}/my_file.png"]
        node.process()
        assert node.parameter_output_values["resolved_paths"] == [_ABSOLUTE]

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
        node.parameter_values["paths"] = ["{bad}/file.png"]
        with pytest.raises(Exception):
            node.process()
        assert node.parameter_output_values["resolved_paths"] == []
