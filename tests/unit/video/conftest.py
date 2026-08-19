"""Shared fixture for exercising the real public-URL upload path.

``PublicArtifactUrlParameter.get_public_url_for_parameter`` opens with its own "is this already
public?" test, and in engine 0.96.0 that test is a weak substring check on ``"localhost"``. Mocking
the method out therefore asserts a node's *intent* while hiding what actually reaches the provider
payload -- which is how tests once passed against a library-side fix that changed nothing. This
fixture stubs only the credential/bucket resolution and the storage driver beneath it, leaving the
method itself to run.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
from griptape_nodes.exe_types.param_components.artifact_url.public_artifact_url_parameter import (
    PublicArtifactUrlParameter,
)

from griptape_nodes_library.video import seedance_common

if TYPE_CHECKING:
    from pathlib import Path


class UploadEnv:
    """Fake Griptape Cloud storage, recording what was really uploaded.

    Doubles as the assertion surface: ``uploaded_keys`` is empty unless bytes actually moved, which
    ``_pending_asset_uploads`` cannot tell you (it is appended to before the upload is attempted).
    """

    signed_url = "https://cloud.griptape.ai/api/buckets/b/signed/style.png"
    inline_data_uri = "data:image/png;base64,SU5MSU5FRA=="

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self.uploaded_keys: list[str] = []

    def upload_file(self, path: Any, file_content: Any) -> str:  # noqa: ARG002
        self.uploaded_keys.append(str(path))
        return self.signed_url

    def delete_file(self, path: Any) -> None:
        return None


@pytest.fixture
def upload_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> UploadEnv:
    """Real ``get_public_url_for_parameter``, fake bucket, workspace pointed at ``tmp_path``.

    Also places a file where a ``/workspace/static_files/style.png`` URL resolves to, so the
    static-server cases upload for real instead of being asserted through a mock.
    """
    env = UploadEnv(tmp_path)

    def fake_init(
        self: PublicArtifactUrlParameter,
        node: Any,
        artifact_url_parameter: Any,
        disclaimer_message: str | None = None,
        request_timeout: float | None = None,
    ) -> None:
        self._node = node
        self._parameter = artifact_url_parameter
        self._disclaimer_message = disclaimer_message
        self._request_timeout = request_timeout
        self._storage_driver = env  # type: ignore[assignment] - fake stands in for GriptapeCloudStorageDriver

    monkeypatch.setattr(PublicArtifactUrlParameter, "__init__", fake_init)
    monkeypatch.setattr(
        seedance_common.GriptapeNodes, "ConfigManager", lambda: SimpleNamespace(workspace_path=tmp_path)
    )

    static_file = tmp_path / "static_files" / "style.png"
    static_file.parent.mkdir(parents=True, exist_ok=True)
    static_file.write_bytes(b"\x89PNG style bytes")
    return env
