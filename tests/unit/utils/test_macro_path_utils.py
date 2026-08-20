"""Unit tests for resolve_to_macro_path."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from griptape_nodes.retained_mode.events.project_events import (
    AttemptMapAbsolutePathToProjectRequest,
    AttemptMapAbsolutePathToProjectResultSuccess,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

import griptape_nodes_library.utils.macro_path_utils as macro_path_utils_module
from griptape_nodes_library.utils.macro_path_utils import resolve_to_macro_path


class TestResolveToMacroPath:
    def test_macro_with_variables_returns_unchanged_without_filesystem_access(
        self, griptape_nodes: GriptapeNodes, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _fail(_request: object) -> object:
            msg = "handle_request should not be called for a macro-with-variables input"
            raise AssertionError(msg)

        monkeypatch.setattr(macro_path_utils_module.GriptapeNodes, "handle_request", staticmethod(_fail))

        result = resolve_to_macro_path("{inputs}/image.png")

        assert result.resolved_path == "{inputs}/image.png"
        assert result.is_external is False

    def test_data_uri_returns_unchanged_and_is_external(
        self, griptape_nodes: GriptapeNodes, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        data_uri = "data:image/png;base64,AAAA"

        def _fail_handle_request(_request: object) -> object:
            msg = "handle_request should not be called for a data URI input"
            raise AssertionError(msg)

        def _fail_file_construction(*_args: object, **_kwargs: object) -> None:
            msg = "File() should not be constructed for a data URI input"
            raise AssertionError(msg)

        monkeypatch.setattr(macro_path_utils_module.GriptapeNodes, "handle_request", staticmethod(_fail_handle_request))
        monkeypatch.setattr(macro_path_utils_module, "File", _fail_file_construction)

        result = resolve_to_macro_path(data_uri)

        assert result.resolved_path == data_uri
        assert result.is_external is True

    def test_bare_relative_path_resolves_against_workspace_not_cwd(
        self, griptape_nodes: GriptapeNodes, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        workspace_dir = tmp_path / "workspace"
        cwd_dir = tmp_path / "cwd"
        workspace_dir.mkdir()
        cwd_dir.mkdir()

        relative_file = workspace_dir / "relative" / "foo.png"
        relative_file.parent.mkdir(parents=True)
        relative_file.write_bytes(b"")

        monkeypatch.chdir(cwd_dir)
        config_manager = GriptapeNodes.ConfigManager()
        monkeypatch.setattr(config_manager, "workspace_path", workspace_dir)

        captured_requests: list[AttemptMapAbsolutePathToProjectRequest] = []

        def fake_handle_request(request: AttemptMapAbsolutePathToProjectRequest) -> object:
            captured_requests.append(request)
            return AttemptMapAbsolutePathToProjectResultSuccess(mapped_path="{inputs}/foo.png", result_details="mapped")

        monkeypatch.setattr(macro_path_utils_module.GriptapeNodes, "handle_request", staticmethod(fake_handle_request))

        result = resolve_to_macro_path("relative/foo.png")

        assert len(captured_requests) == 1
        assert captured_requests[0].absolute_path == relative_file.resolve()
        assert result.resolved_path == "{inputs}/foo.png"
        assert result.is_external is False

    @pytest.mark.skipif(
        sys.platform.startswith("win"), reason="symlink creation requires elevated privileges on Windows"
    )
    def test_absolute_path_through_symlinked_workspace_resolves_to_real_path(
        self,
        monkeypatch,
        griptape_nodes: GriptapeNodes,
        tmp_path: Path,
    ) -> None:
        """Regression: a symlinked workspace component must not make an in-project file external.

        End-to-end against the real engine (no stubbed `handle_request`): the engine
        compares against a symlink-resolved workspace path, so an unresolved symlink-spelled
        path maps to nothing and the file wrongly reports as external.
        """
        real_workspace_dir = tmp_path / "real_workspace"
        real_workspace_dir.mkdir()
        symlinked_workspace_dir = tmp_path / "workspace_link"
        symlinked_workspace_dir.symlink_to(real_workspace_dir, target_is_directory=True)

        project_file = real_workspace_dir / "inputs" / "foo.png"
        project_file.parent.mkdir(parents=True)
        project_file.write_bytes(b"")

        config_manager = GriptapeNodes.ConfigManager()
        monkeypatch.setattr(config_manager, "workspace_path", symlinked_workspace_dir)

        absolute_path_via_symlink = str(symlinked_workspace_dir / "inputs" / "foo.png")

        result = resolve_to_macro_path(absolute_path_via_symlink)

        assert result.resolved_path == "{inputs}/foo.png"
        assert result.is_external is False

    def test_nonexistent_path_returns_external(self, griptape_nodes: GriptapeNodes) -> None:
        result = resolve_to_macro_path("/definitely/does/not/exist/on/disk.png")

        assert result.is_external is True

    @pytest.mark.skipif(
        sys.platform.startswith("win"),
        reason="creates a path component containing a colon, which Windows disallows in filenames",
    )
    def test_url_that_survives_file_resolve_is_not_anchored_to_cwd(
        self, griptape_nodes: GriptapeNodes, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression: a URL that `File.resolve()` returns unchanged (as engine versions with
        static-server URL mapping do for a non-static-server URL, e.g. a remote `https://`
        address) must never be anchored to the process cwd by `Path(...).resolve()`.
        """
        url = "https://example.com/a.mp4"

        class _FakeFile:
            def __init__(self, path: str) -> None:
                self._path = path

            def resolve(self) -> str:
                return self._path

        monkeypatch.setattr(macro_path_utils_module, "File", _FakeFile)
        monkeypatch.chdir(tmp_path)

        # If the URL were (incorrectly) anchored to the process cwd by `Path(...).resolve()`,
        # it would resolve to this on-disk path. Create it so the buggy code path would find
        # "the file exists" and proceed to call `handle_request` -- which must never happen
        # for a URL.
        cwd_anchored_path = Path(url).resolve()
        cwd_anchored_path.parent.mkdir(parents=True, exist_ok=True)
        cwd_anchored_path.write_bytes(b"")

        def _fail_handle_request(_request: object) -> object:
            msg = "handle_request should not be called for a URL that survives File.resolve()"
            raise AssertionError(msg)

        monkeypatch.setattr(macro_path_utils_module.GriptapeNodes, "handle_request", staticmethod(_fail_handle_request))

        result = resolve_to_macro_path(url)

        assert result.resolved_path == url
        assert result.is_external is True
