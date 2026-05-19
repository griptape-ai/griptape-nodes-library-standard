from __future__ import annotations

import ast
import asyncio
import importlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from griptape_nodes.exe_types.param_components.artifact_url.public_artifact_url_parameter import (
    PublicArtifactUrlParameter,
)

from griptape_nodes_library.image.flux_2_image_generation import Flux2ImageGeneration


def _iter_proxy_node_classes() -> Iterator[tuple[str, str]]:
    library_root = Path(__file__).parents[2] / "griptape_nodes_library"
    for path in sorted(library_root.rglob("*.py")):
        module_rel_path = path.relative_to(library_root.parent)
        module_name = ".".join(module_rel_path.with_suffix("").parts)
        tree = ast.parse(path.read_text())
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            if not any(isinstance(base, ast.Name) and base.id == "GriptapeProxyNode" for base in node.bases):
                continue
            yield node.name, module_name


PROXY_NODE_CLASSES = tuple(_iter_proxy_node_classes())


@pytest.fixture(autouse=True)
def stub_public_artifact_bucket_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        PublicArtifactUrlParameter, "_get_bucket_id", staticmethod(lambda *_args, **_kwargs: "test-bucket")
    )


@pytest.mark.parametrize(("class_name", "module_name"), PROXY_NODE_CLASSES)
def test_every_proxy_node_has_timeout_parameter(class_name: str, module_name: str) -> None:
    module = importlib.import_module(module_name)
    node_class = getattr(module, class_name)
    node = node_class(name=class_name)

    timeout_param = next((p for p in node.parameters if p.name == "timeout"), None)
    assert timeout_param is not None, f"{class_name} is missing a 'timeout' parameter"

    cls = type(node)
    expected_default = cls.DEFAULT_MAX_ATTEMPTS * cls.DEFAULT_POLL_INTERVAL
    assert timeout_param.default_value == expected_default, (
        f"{class_name}: expected timeout default {expected_default}, got {timeout_param.default_value}"
    )


@pytest.mark.asyncio
async def test_timeout_parameter_limits_poll_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    poll_count = 0

    class AlwaysRunningResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"status": "RUNNING"}

    class FakeAsyncClient:
        async def __aenter__(self) -> FakeAsyncClient:
            return self

        async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            return None

        async def get(self, url: str, headers: dict[str, str], timeout: int) -> AlwaysRunningResponse:
            nonlocal poll_count
            poll_count += 1
            return AlwaysRunningResponse()

    monkeypatch.setattr("griptape_nodes_library.proxy.griptape_proxy_node.httpx.AsyncClient", FakeAsyncClient)

    async def noop_sleep(_: float) -> None:
        pass

    monkeypatch.setattr("griptape_nodes_library.proxy.griptape_proxy_node.asyncio.sleep", noop_sleep)

    node = Flux2ImageGeneration(name="Flux2")
    node.set_parameter_value("timeout", 10)

    status_calls: list[dict[str, Any]] = []

    def capture_status_results(**kwargs: Any) -> None:
        status_calls.append(kwargs)

    node._set_status_results = capture_status_results  # type: ignore[method-assign]
    node._set_safe_defaults = lambda: None  # type: ignore[method-assign]

    headers = {"Authorization": "Bearer key"}
    result = await node._poll_generation_status("gen-abc", headers)

    assert result is None
    # poll_interval=5, timeout=10 → max_attempts=2
    assert poll_count == 2
    assert len(status_calls) == 1
    assert status_calls[0]["was_successful"] is False
    assert "10 seconds" in status_calls[0]["result_details"]


@pytest.mark.asyncio
async def test_zero_timeout_polls_until_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    poll_count = 0
    complete_after = 15

    class EventuallyCompletedResponse:
        def __init__(self, count: int) -> None:
            self._count = count

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            if self._count >= complete_after:
                return {"status": "COMPLETED"}
            return {"status": "RUNNING"}

    class FakeAsyncClient:
        async def __aenter__(self) -> FakeAsyncClient:
            return self

        async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            return None

        async def get(self, url: str, headers: dict[str, str], timeout: int) -> EventuallyCompletedResponse:
            nonlocal poll_count
            poll_count += 1
            return EventuallyCompletedResponse(poll_count)

    monkeypatch.setattr("griptape_nodes_library.proxy.griptape_proxy_node.httpx.AsyncClient", FakeAsyncClient)

    async def noop_sleep(_: float) -> None:
        pass

    monkeypatch.setattr("griptape_nodes_library.proxy.griptape_proxy_node.asyncio.sleep", noop_sleep)

    node = Flux2ImageGeneration(name="Flux2")
    node.set_parameter_value("timeout", 0)

    timeout_fired = False

    def capture_status_results(**kwargs: Any) -> None:
        nonlocal timeout_fired
        if not kwargs.get("was_successful") and "timed out" in kwargs.get("result_details", ""):
            timeout_fired = True

    node._set_status_results = capture_status_results  # type: ignore[method-assign]

    headers = {"Authorization": "Bearer key"}
    # Wrap in asyncio.wait_for to guard against infinite-loop regression
    result = await asyncio.wait_for(node._poll_generation_status("gen-xyz", headers), timeout=5.0)

    assert result is not None
    assert result["status"] == "COMPLETED"
    assert poll_count == complete_after
    assert not timeout_fired


def test_timeout_parameter_does_not_mutate_class_attribute() -> None:
    original_max_attempts = Flux2ImageGeneration.DEFAULT_MAX_ATTEMPTS

    node = Flux2ImageGeneration(name="Flux2")
    node.set_parameter_value("timeout", 30)

    # _resolve_timeout_seconds reads the parameter
    resolved = node._resolve_timeout_seconds()
    assert resolved == 30

    # Class attribute must be untouched
    assert Flux2ImageGeneration.DEFAULT_MAX_ATTEMPTS == original_max_attempts
    assert node.DEFAULT_MAX_ATTEMPTS == original_max_attempts
