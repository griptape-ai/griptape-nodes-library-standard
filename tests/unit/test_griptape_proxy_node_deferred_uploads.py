"""Issue #875: a polling timeout must not delete an upload the provider still needs.

When polling times out, the provider may not have fetched the uploaded file yet. The node's
own cleanup used to run anyway (finally), so the queued job then failed with no useful reason.
These tests pin the contract:

- a run that times out holds the URL uploads back (deferred, not deleted);
- a run that does not time out deletes them as before;
- a Refresh that sees the generation finished (or failed) deletes the held uploads;
- a Refresh that sees the generation still running leaves them alone;
- the node's next run deletes a leftover held upload so it is not leaked forever.
"""

from __future__ import annotations

from typing import Any

import pytest
from griptape_nodes.exe_types.param_components.artifact_url.public_artifact_url_parameter import (
    PublicArtifactUrlParameter,
)

from griptape_nodes_library.image.flux_2_image_generation import Flux2ImageGeneration
from griptape_nodes_library.proxy.griptape_proxy_node import (
    STATUS_FAILED,
    STATUS_RUNNING,
    STATUS_TIMED_OUT,
    GenerationIncompleteError,
)


def _fake_helper(node: Any) -> PublicArtifactUrlParameter:
    """A URL-upload stand-in: the base only calls adelete_uploaded_artifact on it."""
    helper = PublicArtifactUrlParameter.__new__(PublicArtifactUrlParameter)
    helper.adelete_uploaded_artifact = _deleted  # type: ignore[method-assign]
    node._deleted_helpers.append(helper)  # type: ignore[attr-defined]
    return helper


def _deleted() -> Any:
    raise AssertionError("deleted inline")


@pytest.fixture()
def node() -> Any:
    return Flux2ImageGeneration(name="Flux2")


def _arm_defer(node: Any) -> None:
    node._deleted_helpers = []
    node._deferred_uploads = []
    node._poll_timed_out = True


# --- the hold decision ---------------------------------------------------------


def test_no_timeout_defers_nothing(node: Any) -> None:
    helper_a = object()
    helper_b = object()
    out = node._hold_deferred_uploads_if_timed_out([helper_a, helper_b])
    assert out == [helper_a, helper_b]
    assert node._deferred_uploads == []


def test_timeout_defers_everything(node: Any) -> None:
    _arm_defer(node)
    helper_a = _fake_helper(node)
    helper_b = _fake_helper(node)
    out = node._hold_deferred_uploads_if_timed_out([helper_a, helper_b])
    assert out == []
    assert node._deferred_uploads == [helper_a, helper_b]


def test_timeout_skips_missing_uploads(node: Any) -> None:
    _arm_defer(node)
    helper = _fake_helper(node)
    out = node._hold_deferred_uploads_if_timed_out([helper, None, None])
    assert out == []
    assert node._deferred_uploads == [helper]


# --- refresh deletes a held upload once the generation is finished -------------


@pytest.mark.asyncio
async def test_refresh_completed_deletes_deferred_uploads(node: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    _arm_defer(node)
    held = _fake_helper(node)
    node._deferred_uploads = [held]
    captured: list[Any] = []
    monkeypatch.setattr(
        "griptape_nodes_library.proxy.griptape_proxy_node.adelete_uploaded_artifacts",
        lambda helpers, *, node_name: _capture(captured, helpers, node_name),
    )
    monkeypatch.setattr(node, "_fetch_generation_result", lambda generation_id: _none_coro(None))
    node._set_status_results = lambda **kwargs: None  # type: ignore[method-assign]

    await node._refresh_completed("gen-1")

    assert captured == [([held], "Flux2")]
    assert node._deferred_uploads == []


@pytest.mark.asyncio
async def test_refresh_failed_deletes_deferred_uploads(node: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    _arm_defer(node)
    held = _fake_helper(node)
    node._deferred_uploads = [held]
    captured: list[Any] = []
    monkeypatch.setattr(
        "griptape_nodes_library.proxy.griptape_proxy_node.adelete_uploaded_artifacts",
        lambda helpers, *, node_name: _capture(captured, helpers, node_name),
    )
    captured_status: list[Any] = []
    node._set_status_results = lambda **kwargs: captured_status.append(kwargs)  # type: ignore[method-assign]

    await node._refresh_render_status("gen-1", STATUS_FAILED, {"status": STATUS_FAILED, "status_detail": {}})

    assert captured == [([held], "Flux2")]
    assert node._deferred_uploads == []
    assert captured_status[0]["result_details"].startswith("Generation `gen-1` ended with status FAILED")


@pytest.mark.asyncio
async def test_refresh_still_running_keeps_deferred_uploads(node: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    _arm_defer(node)
    held = _fake_helper(node)
    node._deferred_uploads = [held]
    captured: list[Any] = []
    monkeypatch.setattr(
        "griptape_nodes_library.proxy.griptape_proxy_node.adelete_uploaded_artifacts",
        lambda helpers, *, node_name: _capture(captured, helpers, node_name),
    )
    node._set_status_results = lambda **kwargs: None  # type: ignore[method-assign]

    await node._refresh_render_status("gen-1", STATUS_RUNNING, {"status": STATUS_RUNNING})

    assert captured == []
    assert node._deferred_uploads == [held]


# --- the next run takes over a leftover held upload ----------------------------


@pytest.mark.asyncio
async def test_next_run_deletes_leftover_deferred_uploads(node: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    _arm_defer(node)
    held = _fake_helper(node)
    node._deferred_uploads = [held]
    captured: list[Any] = []
    monkeypatch.setattr(
        "griptape_nodes_library.proxy.griptape_proxy_node.adelete_uploaded_artifacts",
        lambda helpers, *, node_name: _capture(captured, helpers, node_name),
    )

    await node._delete_deferred_uploads()

    assert captured == [([held], "Flux2")]
    assert node._deferred_uploads == []
    # A second pass with nothing held deletes nothing.
    await node._delete_deferred_uploads()
    assert captured == [([held], "Flux2")]


# --- the timeout path itself sets the flag the node cleanup consults -----------


@pytest.mark.asyncio
async def test_poll_timeout_sets_flag_before_error_unwinds(node: Any, monkeypatch: pytest.MonkeyPatch) -> None:
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

        async def get(self, url: str, headers: dict[str, Any], timeout: int) -> RunningResponse:
            return RunningResponse()

    monkeypatch.setattr("griptape_nodes_library.proxy.griptape_proxy_node.httpx.AsyncClient", FakeAsyncClient)

    async def noop_sleep(_: float) -> None:
        pass

    monkeypatch.setattr("griptape_nodes_library.proxy.griptape_proxy_node.asyncio.sleep", noop_sleep)

    node.set_parameter_value("timeout", 5)
    node.parameter_output_values["generation_id"] = "gen-timed-out"
    node._set_safe_defaults = lambda: None  # type: ignore[method-assign]
    node._poll_timed_out = False

    with pytest.raises(GenerationIncompleteError):
        await node._poll_generation_status("gen-timed-out", {"Authorization": "Bearer key"})

    assert node._poll_timed_out is True
    assert node.parameter_output_values["generation_status"] == STATUS_TIMED_OUT


# --- helpers -------------------------------------------------------------------


async def _capture(captured: list[Any], helpers: Any, node_name: str) -> None:
    captured.append((list(helpers), node_name))


def _none_coro(value: Any) -> Any:
    async def coro() -> Any:
        return value

    return coro()
