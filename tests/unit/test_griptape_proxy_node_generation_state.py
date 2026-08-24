"""Tests for live publication of generation state and adoption of an existing generation.

The incident these cover: a generation polled for ~58 minutes holding a FREE org's only
concurrency slot, 14 more queued behind it, and the session died before any node resolved.
Because generation IDs only ever reached `parameter_output_values` — which the engine
flushes at node-resolve — nothing outside the dead session could name the work that had
been paid for. These tests pin the two properties that make that unrecoverable state
impossible: the ID is published the moment it exists, and COMPLETED is published only once
the result is genuinely on the node.
"""

from __future__ import annotations

from typing import Any

import pytest
from griptape_nodes.exe_types.core_types import ParameterMode

from griptape_nodes_library.image.flux_2_image_generation import Flux2ImageGeneration
from griptape_nodes_library.proxy import griptape_proxy_node as proxy_module
from griptape_nodes_library.proxy.griptape_proxy_node import (
    MAX_TIMEOUT_SECONDS,
    STATUS_COMPLETED,
    STATUS_ERRORED,
    STATUS_QUEUED,
    STATUS_RUNNING,
    STATUS_TIMED_OUT,
)


class _Published:
    """Records what the node pushed to the UI, in order."""

    def __init__(self, node: Any) -> None:
        self.calls: list[tuple[str, Any]] = []
        node.publish_update_to_parameter = self._record  # type: ignore[method-assign]

    def _record(self, parameter_name: str, value: Any) -> None:
        self.calls.append((parameter_name, value))

    def values_for(self, parameter_name: str) -> list[Any]:
        return [value for name, value in self.calls if name == parameter_name]

    def statuses(self) -> list[Any]:
        return self.values_for("generation_status")

    def ids(self) -> list[Any]:
        return self.values_for("generation_id")


class _Cleared:
    """Stand-in for the ResultPayload from declare_model_invocation."""

    def failed(self) -> bool:
        return False


def _make_node(monkeypatch: pytest.MonkeyPatch) -> Flux2ImageGeneration:
    node = Flux2ImageGeneration(name="Flux2")
    node._set_safe_defaults = lambda: None  # type: ignore[method-assign]
    node._set_status_results = lambda **_kwargs: None  # type: ignore[method-assign]

    async def _cleared(_node: Any, _model_id: str) -> _Cleared:
        return _Cleared()

    monkeypatch.setattr(proxy_module, "declare_model_invocation", _cleared)
    return node


def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    async def noop_sleep(_: float) -> None:
        pass

    monkeypatch.setattr("griptape_nodes_library.proxy.griptape_proxy_node.asyncio.sleep", noop_sleep)


def _fake_client(monkeypatch: pytest.MonkeyPatch, payloads: list[dict[str, Any]]) -> None:
    """Serve `payloads` in order from the status endpoint, repeating the last one."""

    class Response:
        def __init__(self, body: dict[str, Any]) -> None:
            self._body = body

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return self._body

    remaining = list(payloads)

    class FakeAsyncClient:
        async def __aenter__(self) -> FakeAsyncClient:
            return self

        async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            return None

        async def get(self, url: str, headers: dict[str, str], timeout: int) -> Response:  # noqa: ARG002
            body = remaining.pop(0) if len(remaining) > 1 else remaining[0]
            return Response(body)

    monkeypatch.setattr("griptape_nodes_library.proxy.griptape_proxy_node.httpx.AsyncClient", FakeAsyncClient)


# --- TASK 1: publish generation state live ---------------------------------------------


@pytest.mark.asyncio
async def test_generation_id_is_published_before_polling_begins(monkeypatch: pytest.MonkeyPatch) -> None:
    """The load-bearing guarantee: the ID reaches the UI before the long wait, not after it.

    A node that never resolves never flushes parameter_output_values, so if the ID is only
    written there, a session that dies mid-poll strands the generation. Publishing before
    _poll_generation_status is what makes the ID outlive the session.
    """
    node = _make_node(monkeypatch)
    published = _Published(node)

    async def _build_payload() -> dict[str, Any]:
        return {"prompt": "x"}

    async def _submit_generation(_payload: Any, _headers: Any, _model: Any) -> str:
        return "gen-published-early"

    ids_seen_when_polling_started: list[Any] = []

    async def _poll(_generation_id: str, _headers: dict[str, str]) -> None:
        ids_seen_when_polling_started.extend(published.ids())
        return None

    node._build_payload = _build_payload  # type: ignore[method-assign]
    node._get_api_model_id = lambda: "flux-2"  # type: ignore[method-assign]
    node._submit_generation = _submit_generation  # type: ignore[method-assign]
    node._poll_generation_status = _poll  # type: ignore[method-assign]

    await node._submit_and_poll({"Authorization": "Bearer k"})

    # Published, and published *before* polling started.
    assert "gen-published-early" in published.ids()
    assert ids_seen_when_polling_started == ["gen-published-early"]
    # QUEUED goes out too, which is the state in which the cloud still allows a free cancel.
    assert published.statuses() == [STATUS_QUEUED]
    assert node.parameter_output_values["generation_id"] == "gen-published-early"


@pytest.mark.asyncio
async def test_poll_loop_publishes_intermediate_status(monkeypatch: pytest.MonkeyPatch) -> None:
    node = _make_node(monkeypatch)
    published = _Published(node)
    _no_sleep(monkeypatch)
    _fake_client(monkeypatch, [{"status": STATUS_QUEUED}, {"status": STATUS_RUNNING}])

    node.set_parameter_value("timeout", 10)
    await node._poll_generation_status("gen-1", {"Authorization": "Bearer k"})

    assert STATUS_QUEUED in published.statuses()
    assert STATUS_RUNNING in published.statuses()


@pytest.mark.asyncio
async def test_poll_loop_withholds_completed_until_result_is_on_the_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """COMPLETED must not be published by the poll loop.

    The editor reads a published COMPLETED as "the node has the result, stop offering
    recovery". The poll loop sees COMPLETED one fetch and one parse before that is true, so
    publishing it there would hide the recovery affordance for a result the node does not
    hold — the exact failure this feature exists to prevent.
    """
    node = _make_node(monkeypatch)
    published = _Published(node)
    _no_sleep(monkeypatch)
    _fake_client(monkeypatch, [{"status": STATUS_COMPLETED}])

    result = await node._poll_generation_status("gen-done", {"Authorization": "Bearer k"})

    # The loop reported the terminal result upward...
    assert result == {"status": STATUS_COMPLETED}
    # ...and recorded it locally, but did not announce it to the editor.
    assert node.parameter_output_values["generation_status"] == STATUS_COMPLETED
    assert STATUS_COMPLETED not in published.statuses()


def test_publish_generation_completed_is_the_only_path_that_announces_completed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node = _make_node(monkeypatch)
    published = _Published(node)

    node._publish_generation_state(status=STATUS_COMPLETED)
    assert STATUS_COMPLETED not in published.statuses()

    node._publish_generation_completed()
    assert published.statuses() == [STATUS_COMPLETED]


@pytest.mark.asyncio
async def test_polling_error_exhaustion_preserves_generation_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """Giving up because polling kept erroring must still leave the ID recoverable.

    The two error branches called _set_safe_defaults() and returned without restoring
    generation_id, so a run that died to repeated transport errors lost the pointer to
    billable work in exactly the way the timeout path was fixed not to.
    """
    node = _make_node(monkeypatch)
    published = _Published(node)
    _no_sleep(monkeypatch)

    class ExplodingClient:
        async def __aenter__(self) -> ExplodingClient:
            return self

        async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            return None

        async def get(self, url: str, headers: dict[str, str], timeout: int) -> Any:  # noqa: ARG002
            msg = "connection reset"
            raise RuntimeError(msg)

    monkeypatch.setattr("griptape_nodes_library.proxy.griptape_proxy_node.httpx.AsyncClient", ExplodingClient)

    node.set_parameter_value("timeout", 5)
    result = await node._poll_generation_status("gen-errored", {"Authorization": "Bearer k"})

    assert result is None
    assert node.parameter_output_values["generation_id"] == "gen-errored"
    assert "gen-errored" in published.ids()


@pytest.mark.asyncio
async def test_parse_failure_publishes_id_before_the_failure_can_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    """A parse failure that crashes the flow must still leave the ID recoverable.

    `_handle_result_parsing_error` ends in `_handle_failure_exception`, which re-raises when
    the Failed output has no outgoing connection. Publishing after that call would be
    skipped in precisely the case that matters: the generation ran and was billed, and only
    our parsing of it failed.
    """
    node = _make_node(monkeypatch)
    published = _Published(node)

    async def _submit_and_poll(_headers: dict[str, str]) -> tuple[str, dict[str, Any]]:
        return "gen-unparseable", {"status": STATUS_COMPLETED}

    async def _fetch(_generation_id: str) -> dict[str, Any]:
        return {"images": []}

    async def _parse_result(_result_json: dict[str, Any], _generation_id: str) -> None:
        msg = "unexpected result shape"
        raise ValueError(msg)

    def _explode(exception: Exception) -> None:
        raise exception

    node._submit_and_poll = _submit_and_poll  # type: ignore[method-assign]
    node._fetch_generation_result = _fetch  # type: ignore[method-assign]
    node._parse_result = _parse_result  # type: ignore[method-assign]
    node._validate_api_key = lambda: "fake-key"  # type: ignore[method-assign]
    # Stand in for an unconnected Failed output, which re-raises.
    node._handle_failure_exception = _explode  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="unexpected result shape"):
        await node._process_generation()

    assert "gen-unparseable" in published.ids()
    # Short of COMPLETED, so the editor keeps offering recovery.
    assert STATUS_COMPLETED not in published.statuses()
    # And not relabelled ERRORED: in the cloud's vocabulary that means "internal failure,
    # nothing is billed", which is the opposite of a generation that ran and was charged.
    assert STATUS_ERRORED not in published.statuses()


# --- TASK 2: adopt an existing generation ID -------------------------------------------


def test_generation_id_parameter_is_settable_and_visible_for_adoption() -> None:
    """Pasting an ID from the editor's tray is the only way a raw-bytes result reaches a user."""
    node = Flux2ImageGeneration(name="Flux2")
    status_group = node.status_component.get_parameter_group()
    generation_id = next(child for child in status_group.children if child.name == "generation_id")

    assert ParameterMode.PROPERTY in generation_id.get_mode()
    # Still reported as an output by the node that submitted it.
    assert ParameterMode.OUTPUT in generation_id.get_mode()
    assert generation_id.settable is True


@pytest.mark.asyncio
async def test_refresh_prefers_pasted_id_over_the_nodes_own_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """A pasted ID lands in parameter_values; reading only output values would ignore it."""
    node = _make_node(monkeypatch)
    _fake_client(monkeypatch, [{"status": STATUS_RUNNING}])
    monkeypatch.setattr(
        "griptape_nodes_library.proxy.provider_asset_access.GriptapeNodes.SecretsManager",
        lambda: type("S", (), {"get_secret": lambda self, _name: "fake-key"})(),
    )

    node.parameter_output_values["generation_id"] = "gen-from-this-node"
    node.set_parameter_value("generation_id", "gen-pasted-by-user")

    await node._refresh_async()

    assert node._resolve_refresh_generation_id() == "gen-pasted-by-user"
    assert node.parameter_output_values["generation_id"] == "gen-pasted-by-user"


def test_refresh_refuses_a_generation_from_a_different_model(monkeypatch: pytest.MonkeyPatch) -> None:
    node = _make_node(monkeypatch)
    node._get_api_model_id = lambda: "flux-2"  # type: ignore[method-assign]

    mismatch = node._refresh_model_mismatch({"status": STATUS_COMPLETED, "model": "sora-2"})

    assert mismatch is not None
    assert "sora-2" in mismatch
    assert "flux-2" in mismatch


def test_refresh_accepts_a_matching_model_ignoring_operation_suffix(monkeypatch: pytest.MonkeyPatch) -> None:
    """`_get_api_model_id` may carry an operation suffix the cloud does not echo back."""
    node = _make_node(monkeypatch)
    node._get_api_model_id = lambda: "grok-imagine-video:generate"  # type: ignore[method-assign]
    node._get_catalog_model_id = lambda: "grok-imagine-video"  # type: ignore[method-assign]

    assert node._refresh_model_mismatch({"model": "grok-imagine-video"}) is None
    assert node._refresh_model_mismatch({"model_id": "GROK-IMAGINE-VIDEO"}) is None


def test_refresh_fails_open_when_the_response_names_no_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never block the only binary-recovery route on a field the API may not send."""
    node = _make_node(monkeypatch)
    node._get_api_model_id = lambda: "flux-2"  # type: ignore[method-assign]

    assert node._refresh_model_mismatch({"status": STATUS_COMPLETED}) is None


def test_status_constants_match_the_proxy_spec_vocabulary() -> None:
    """`generation_status` is reconciled against the cloud, so the values must be its own.

    These six are `ProxyGenerationStatus` in the proxy Smithy spec. TIMED_OUT is
    deliberately not among them: it is this library's local "we stopped watching" marker,
    which the editor renders as a neutral badge.
    """
    assert (STATUS_QUEUED, STATUS_RUNNING, STATUS_COMPLETED) == ("QUEUED", "RUNNING", "COMPLETED")
    assert (proxy_module.STATUS_FAILED, STATUS_ERRORED, proxy_module.STATUS_CANCELLED) == (
        "FAILED",
        "ERRORED",
        "CANCELLED",
    )
    assert STATUS_TIMED_OUT == "TIMED_OUT"


# --- TASK 3: bounded polling ------------------------------------------------------------


def test_timeout_zero_is_clamped_rather_than_unbounded() -> None:
    """`timeout=0` used to poll forever, which is how one generation held the only slot."""
    node = Flux2ImageGeneration(name="Flux2")
    node.set_parameter_value("timeout", 0)

    assert node._resolve_timeout_seconds() == MAX_TIMEOUT_SECONDS


def test_timeout_is_capped_at_the_documented_maximum() -> None:
    node = Flux2ImageGeneration(name="Flux2")
    node.set_parameter_value("timeout", MAX_TIMEOUT_SECONDS * 10)

    assert node._resolve_timeout_seconds() == MAX_TIMEOUT_SECONDS


@pytest.mark.asyncio
async def test_polling_stops_on_wall_clock_deadline_even_with_attempts_left(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Attempt-counting alone does not bound elapsed time.

    Each iteration costs the HTTP request (up to 60s on a hanging poll) plus the poll
    interval, so an attempt cap derived from `timeout / poll_interval` let a 600s timeout
    keep a node polling for over two hours. The deadline is the bound that actually
    reflects what the user asked for.
    """
    node = _make_node(monkeypatch)
    _fake_client(monkeypatch, [{"status": STATUS_RUNNING}])

    # Every "sleep" advances the loop clock far more than the poll interval, standing in for
    # slow responses. Attempts remain available; only the deadline can end this.
    class FastClock:
        def __init__(self) -> None:
            self._t = 0.0

        def time(self) -> float:
            return self._t

        def advance(self, seconds: float) -> None:
            self._t += seconds

    clock = FastClock()
    monkeypatch.setattr("griptape_nodes_library.proxy.griptape_proxy_node.asyncio.get_running_loop", lambda: clock)

    async def slow_sleep(_: float) -> None:
        clock.advance(60.0)

    monkeypatch.setattr("griptape_nodes_library.proxy.griptape_proxy_node.asyncio.sleep", slow_sleep)

    node.set_parameter_value("timeout", 600)
    result = await node._poll_generation_status("gen-slow", {"Authorization": "Bearer k"})

    assert result is None
    assert node.parameter_output_values["generation_status"] == STATUS_TIMED_OUT
    # 600s of budget at ~60s per iteration: nowhere near the 120-attempt cap.
    assert clock.time() <= 600 + 60


@pytest.mark.asyncio
async def test_timeout_does_not_cancel_the_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cancel only works while QUEUED, and a RUNNING generation bills either way.

    Cancelling on timeout would throw away a result the user has already paid for, so the
    timeout path must never issue a cancel.
    """
    node = _make_node(monkeypatch)
    _no_sleep(monkeypatch)
    _fake_client(monkeypatch, [{"status": STATUS_RUNNING}])

    cancelled: list[str] = []
    if hasattr(node, "_request_generation_cancel"):  # composes with PR #550 once merged
        node._request_generation_cancel = lambda gid, _headers: cancelled.append(gid)  # type: ignore[method-assign]

    node.set_parameter_value("timeout", 5)
    await node._poll_generation_status("gen-running", {"Authorization": "Bearer k"})

    assert cancelled == []
    assert node.parameter_output_values["generation_status"] == STATUS_TIMED_OUT
    assert node.parameter_output_values["generation_id"] == "gen-running"
