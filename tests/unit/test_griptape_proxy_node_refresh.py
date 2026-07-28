from __future__ import annotations

from typing import Any

import pytest
from griptape_nodes.exe_types.core_types import ParameterMode
from griptape_nodes.exe_types.param_types.parameter_button import ParameterButton
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString

from griptape_nodes_library.image.flux_2_image_generation import Flux2ImageGeneration
from griptape_nodes_library.proxy.griptape_proxy_node import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_RUNNING,
    STATUS_TIMED_OUT,
)


def _find_status_group_children(node: Any) -> dict[str, Any]:
    status_group = node.status_component.get_parameter_group()
    return {child.name: child for child in status_group.children}


def test_status_group_contains_generation_id_status_and_refresh_button() -> None:
    node = Flux2ImageGeneration(name="Flux2")
    children = _find_status_group_children(node)

    assert "generation_id" in children
    assert isinstance(children["generation_id"], ParameterString)
    assert ParameterMode.OUTPUT in children["generation_id"].get_mode()

    assert "generation_status" in children
    assert isinstance(children["generation_status"], ParameterString)
    assert ParameterMode.OUTPUT in children["generation_status"].get_mode()

    assert "generation_refresh" in children
    assert isinstance(children["generation_refresh"], ParameterButton)


@pytest.mark.asyncio
async def test_polling_timeout_preserves_generation_id_and_sets_timed_out_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Timeout must not wipe generation_id — that's how the Refresh button recovers the result."""

    class RunningResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"status": STATUS_RUNNING}

    class FakeAsyncClient:
        async def __aenter__(self) -> FakeAsyncClient:
            return self

        async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            return None

        async def get(self, url: str, headers: dict[str, str], timeout: int) -> RunningResponse:
            return RunningResponse()

    monkeypatch.setattr("griptape_nodes_library.proxy.griptape_proxy_node.httpx.AsyncClient", FakeAsyncClient)

    async def noop_sleep(_: float) -> None:
        pass

    monkeypatch.setattr("griptape_nodes_library.proxy.griptape_proxy_node.asyncio.sleep", noop_sleep)

    node = Flux2ImageGeneration(name="Flux2")
    node.set_parameter_value("timeout", 5)
    # Simulate that submission already wrote the generation_id (as _submit_and_poll does).
    node.parameter_output_values["generation_id"] = "gen-preserved"
    node._set_safe_defaults = lambda: None  # type: ignore[method-assign]

    result = await node._poll_generation_status("gen-preserved", {"Authorization": "Bearer key"})

    assert result is None
    assert node.parameter_output_values["generation_id"] == "gen-preserved"
    assert node.parameter_output_values["generation_status"] == STATUS_TIMED_OUT


def test_handle_terminal_status_preserves_generation_id_on_failure() -> None:
    node = Flux2ImageGeneration(name="Flux2")
    node.parameter_output_values["generation_id"] = "gen-failed"
    node._set_safe_defaults = lambda: None  # type: ignore[method-assign]

    is_terminal, terminal_result = node._handle_terminal_status(
        STATUS_FAILED,
        {"status": STATUS_FAILED, "status_detail": {"error": "boom", "details": "bad"}},
    )

    assert is_terminal is True
    assert terminal_result is None
    assert node.parameter_output_values["generation_id"] == "gen-failed"
    assert node.parameter_output_values["generation_status"] == STATUS_FAILED


@pytest.mark.asyncio
async def test_refresh_async_without_generation_id_reports_unavailable() -> None:
    node = Flux2ImageGeneration(name="Flux2")
    # Explicitly empty
    node.parameter_output_values["generation_id"] = ""

    captured: list[dict[str, Any]] = []
    node._set_status_results = lambda **kwargs: captured.append(kwargs)  # type: ignore[method-assign]

    await node._refresh_async()

    assert len(captured) == 1
    assert captured[0]["was_successful"] is False
    assert "No generation ID" in captured[0]["result_details"]


@pytest.mark.asyncio
async def test_refresh_async_running_updates_status_without_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    class RunningResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"status": STATUS_RUNNING}

    class FakeAsyncClient:
        async def __aenter__(self) -> FakeAsyncClient:
            return self

        async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            return None

        async def get(self, url: str, headers: dict[str, str], timeout: int) -> RunningResponse:
            return RunningResponse()

    monkeypatch.setattr("griptape_nodes_library.proxy.griptape_proxy_node.httpx.AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(
        "griptape_nodes_library.proxy.provider_asset_access.GriptapeNodes.SecretsManager",
        lambda: type("S", (), {"get_secret": lambda self, _name: "fake-key"})(),
    )

    node = Flux2ImageGeneration(name="Flux2")
    node.parameter_output_values["generation_id"] = "gen-running"

    parse_called = False

    async def _parse_result(_result_json: dict[str, Any], _generation_id: str) -> None:
        nonlocal parse_called
        parse_called = True

    node._parse_result = _parse_result  # type: ignore[method-assign]

    captured: list[dict[str, Any]] = []
    node._set_status_results = lambda **kwargs: captured.append(kwargs)  # type: ignore[method-assign]

    await node._refresh_async()

    assert node.parameter_output_values["generation_status"] == STATUS_RUNNING
    assert parse_called is False
    # _refresh_render_status emits one set_status_results call for non-completed states.
    assert any("still in progress" in c.get("result_details", "") for c in captured)


@pytest.mark.asyncio
async def test_refresh_async_completed_invokes_parse_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the cloud reports COMPLETED, refresh must fetch the result and call _parse_result."""

    class StatusResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"status": STATUS_COMPLETED}

    class ResultResponse:
        headers = {"content-type": "application/json"}

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"images": [{"url": "https://example.com/x.png"}]}

    call_log: list[str] = []

    class FakeAsyncClient:
        async def __aenter__(self) -> FakeAsyncClient:
            return self

        async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            return None

        async def get(self, url: str, headers: dict[str, str], timeout: int) -> Any:
            call_log.append(url)
            if url.endswith("/result"):
                return ResultResponse()
            return StatusResponse()

    monkeypatch.setattr("griptape_nodes_library.proxy.griptape_proxy_node.httpx.AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(
        "griptape_nodes_library.proxy.provider_asset_access.GriptapeNodes.SecretsManager",
        lambda: type("S", (), {"get_secret": lambda self, _name: "fake-key"})(),
    )

    node = Flux2ImageGeneration(name="Flux2")
    node.parameter_output_values["generation_id"] = "gen-done"

    parsed: dict[str, Any] = {}

    async def _parse_result(result_json: dict[str, Any], generation_id: str) -> None:
        parsed["result_json"] = result_json
        parsed["generation_id"] = generation_id

    node._parse_result = _parse_result  # type: ignore[method-assign]
    node._set_status_results = lambda **_kwargs: None  # type: ignore[method-assign]

    await node._refresh_async()

    assert node.parameter_output_values["generation_status"] == STATUS_COMPLETED
    assert parsed["generation_id"] == "gen-done"
    assert parsed["result_json"] == {"images": [{"url": "https://example.com/x.png"}]}
    # Both endpoints were hit: status, then /result
    assert any(u.endswith("/generations/gen-done") for u in call_log)
    assert any(u.endswith("/generations/gen-done/result") for u in call_log)
