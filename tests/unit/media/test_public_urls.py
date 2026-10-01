from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from griptape_nodes_library.media.public_urls import adelete_uploaded_artifact, aget_public_url, gather_limited


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
