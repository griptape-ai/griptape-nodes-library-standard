from __future__ import annotations

import asyncio
import base64
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import httpx
import pytest

from griptape_nodes_library.media import public_urls
from griptape_nodes_library.media.public_urls import (
    adelete_uploaded_artifacts,
    gather_limited,
    local_path_for_static_server_url,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestGatherLimited:
    @pytest.mark.asyncio
    async def test_results_keep_input_order(self) -> None:
        async def after(delay: float, value: int) -> int:
            await asyncio.sleep(delay)
            return value

        assert await gather_limited([after(0.03, 1), after(0.0, 2), after(0.01, 3)]) == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_runs_at_most_limit_at_once(self) -> None:
        running = 0
        peak = 0

        async def work() -> None:
            nonlocal running, peak
            running += 1
            peak = max(peak, running)
            await asyncio.sleep(0.01)
            running -= 1

        limit = 2
        await gather_limited([work() for _ in range(6)], limit=limit)

        assert peak == limit

    @pytest.mark.asyncio
    async def test_first_failure_is_raised_after_the_rest_stop(self) -> None:
        stopped: list[str] = []

        async def fail() -> None:
            msg = "upload failed"
            raise ValueError(msg)

        async def slow() -> None:
            try:
                await asyncio.sleep(5)
            finally:
                stopped.append("slow")

        with pytest.raises(ValueError, match="upload failed"):
            await gather_limited([slow(), fail()])

        assert stopped == ["slow"]


class TestGatherLimitedQueuedCoroutines:
    @pytest.mark.asyncio
    async def test_queued_coroutines_are_closed_after_a_failure(self) -> None:
        started: list[int] = []

        async def work(index: int) -> None:
            started.append(index)
            if index == 0:
                msg = "upload failed"
                raise ValueError(msg)
            await asyncio.sleep(5)

        coros = [work(index) for index in range(6)]
        with pytest.raises(ValueError, match="upload failed"):
            await gather_limited(coros, limit=2)

        # Some were still queued when the failure cancelled the rest, and every one was closed
        # rather than left unawaited.
        assert len(started) < len(coros)
        assert all(coro.cr_frame is None for coro in coros)


class TestDeleteAll:
    @pytest.mark.asyncio
    async def test_one_failed_delete_does_not_skip_the_rest(self, caplog: pytest.LogCaptureFixture) -> None:
        deleted: list[str] = []

        async def fail() -> None:
            msg = "delete failed"
            raise RuntimeError(msg)

        async def delete_second() -> None:
            deleted.append("second")

        helpers: list[Any] = [
            SimpleNamespace(adelete_uploaded_artifact=fail),
            SimpleNamespace(adelete_uploaded_artifact=delete_second),
        ]

        await adelete_uploaded_artifacts(helpers, node_name="Node")

        assert deleted == ["second"]
        assert "delete failed" in caplog.text


class TestLocalPathForStaticServerUrl:
    @pytest.fixture
    def workspace(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
        workspace = tmp_path / "workspace-root"
        (workspace / "static_files").mkdir(parents=True)
        monkeypatch.setattr(
            public_urls.GriptapeNodes, "ConfigManager", lambda: SimpleNamespace(workspace_path=workspace)
        )
        return workspace

    def test_maps_any_host_spelling_to_the_served_file(self, workspace: Path) -> None:
        served = workspace / "static_files" / "my clip.mp4"
        served.write_bytes(b"clip")

        url = "http://192.168.1.20:8124/workspace/static_files/my%20clip.mp4?t=1"

        assert local_path_for_static_server_url(url) == str(served.resolve())

    def test_rejects_a_path_escaping_the_workspace(self, workspace: Path) -> None:
        (workspace.parent / "secret.txt").write_text("secret")

        assert local_path_for_static_server_url("http://localhost:8124/workspace/../secret.txt") is None

    def test_returns_none_for_a_file_not_on_this_machine(self, workspace: Path) -> None:  # noqa: ARG002
        assert local_path_for_static_server_url("http://other-host:8124/workspace/static_files/x.mp4") is None


class TestDownloadAsDataUri:
    @staticmethod
    def _serve(monkeypatch: pytest.MonkeyPatch, content_type: str | None) -> None:
        headers = {"content-type": content_type} if content_type else {}
        transport = httpx.MockTransport(lambda _request: httpx.Response(200, content=b"bytes", headers=headers))
        real_client = httpx.AsyncClient
        monkeypatch.setattr(public_urls.httpx, "AsyncClient", lambda: real_client(transport=transport))

    @pytest.mark.asyncio
    async def test_uses_the_response_media_type(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._serve(monkeypatch, "video/quicktime")

        data_uri = await public_urls.adownload_as_data_uri("http://lan-host/clip", fallback_mime="video/mp4")

        assert data_uri == "data:video/quicktime;base64," + base64.b64encode(b"bytes").decode()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("url", "content_type", "expected"),
        [
            # A generic response type falls back to the path's extension...
            ("http://lan-host/clip.webm?sig=1", "application/octet-stream", "video/webm"),
            # ...and to the caller's fallback when the path has none, never to text/plain.
            ("http://lan-host/videos/abc", None, "video/mp4"),
            # A non-media extension guess is no better than none.
            ("http://lan-host/out.ts", "application/octet-stream", "video/mp4"),
        ],
    )
    async def test_falls_back_when_the_response_is_untyped(
        self, monkeypatch: pytest.MonkeyPatch, url: str, content_type: str | None, expected: str
    ) -> None:
        self._serve(monkeypatch, content_type)

        data_uri = await public_urls.adownload_as_data_uri(url, fallback_mime="video/mp4")

        assert data_uri.startswith(f"data:{expected};base64,")
