from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from griptape_nodes_library.media.public_urls import (
    adelete_uploaded_artifact,
    adelete_uploaded_artifacts,
    aget_public_url,
    gather_limited,
)


class TestEngineCompatibility:
    @pytest.mark.asyncio
    async def test_prefers_the_async_upload(self) -> None:
        async def aget() -> str:
            return "https://async"

        helper: Any = SimpleNamespace(aget_public_url_for_parameter=aget, get_public_url_for_parameter=lambda: "sync")

        assert await aget_public_url(helper) == "https://async"

    @pytest.mark.asyncio
    async def test_falls_back_to_the_blocking_upload(self) -> None:
        helper: Any = SimpleNamespace(get_public_url_for_parameter=lambda: "https://sync")

        assert await aget_public_url(helper) == "https://sync"

    @pytest.mark.asyncio
    async def test_delete_falls_back_to_the_blocking_delete(self) -> None:
        deleted: list[bool] = []
        helper: Any = SimpleNamespace(delete_uploaded_artifact=lambda: deleted.append(True))

        await adelete_uploaded_artifact(helper)

        assert deleted == [True]


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

        def fail() -> None:
            msg = "delete failed"
            raise RuntimeError(msg)

        helpers: list[Any] = [
            SimpleNamespace(delete_uploaded_artifact=fail),
            SimpleNamespace(delete_uploaded_artifact=lambda: deleted.append("second")),
        ]

        await adelete_uploaded_artifacts(helpers, node_name="Node")

        assert deleted == ["second"]
        assert "delete failed" in caplog.text
