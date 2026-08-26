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

import ast
import inspect
from pathlib import Path
from typing import Any

import httpx
import pytest
from griptape_nodes.exe_types.core_types import ParameterMode
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString

import griptape_nodes_library
from griptape_nodes_library.image.flux_2_image_generation import MODEL_OPTIONS, Flux2ImageGeneration
from griptape_nodes_library.image.flux_image_generation import FluxImageGeneration
from griptape_nodes_library.image.topaz_image_enhance import TopazImageEnhance
from griptape_nodes_library.proxy import griptape_proxy_node as proxy_module
from griptape_nodes_library.proxy.griptape_proxy_node import (
    MAX_TIMEOUT_SECONDS,
    STATUS_COMPLETED,
    STATUS_ERRORED,
    STATUS_QUEUED,
    STATUS_RUNNING,
    STATUS_TIMED_OUT,
    GriptapeProxyNode,
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


def _make_node(monkeypatch: pytest.MonkeyPatch, *, real_safe_defaults: bool = False) -> Flux2ImageGeneration:
    """Build a node with the network and status plumbing stubbed out.

    `_set_safe_defaults` is stubbed by default to keep unrelated tests focused, but it blanks
    `generation_id` in every subclass, so any test asserting that an ID survives a failure
    path must pass `real_safe_defaults=True` or it proves nothing.
    """
    node = Flux2ImageGeneration(name="Flux2")
    if not real_safe_defaults:
        node._set_safe_defaults = lambda: None  # type: ignore[method-assign]
    node._set_status_results = lambda **_kwargs: None  # type: ignore[method-assign]

    async def _cleared(_node: Any, _model_id: str) -> _Cleared:
        return _Cleared()

    monkeypatch.setattr(proxy_module, "declare_model_invocation", _cleared)
    return node


def _returns(value: Any) -> Any:
    """An async stand-in for a coroutine method that just yields `value`."""

    async def _coroutine(*_args: Any, **_kwargs: Any) -> Any:
        return value

    return _coroutine


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
async def test_poll_loop_does_not_publish_a_fabricated_status(monkeypatch: pytest.MonkeyPatch) -> None:
    """`generation_status` is reconciled against the cloud, so only report what it said.

    A single malformed response would otherwise replace a good RUNNING badge with our own
    "unknown" sentinel — a value the cloud has no notion of.
    """
    node = _make_node(monkeypatch)
    published = _Published(node)
    _no_sleep(monkeypatch)
    _fake_client(monkeypatch, [{"status": STATUS_RUNNING}, {"detail": "no status field"}])

    node.set_parameter_value("timeout", 10)
    await node._poll_generation_status("gen-1", {"Authorization": "Bearer k"})

    # The malformed second response contributes nothing: RUNNING then our own timeout marker,
    # with no sentinel wedged between them.
    assert published.statuses() == [STATUS_RUNNING, STATUS_TIMED_OUT]
    assert node.parameter_output_values["generation_status"] == STATUS_TIMED_OUT


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
    # ...but announced COMPLETED on neither channel. `parameter_output_values` is not inert:
    # TrackedParameterOutputValues.__setitem__ emits an AlterElementEvent on every write, so
    # recording COMPLETED here would put it in front of the user before it was true.
    assert node.parameter_output_values.get("generation_status") != STATUS_COMPLETED
    assert STATUS_COMPLETED not in published.statuses()


def test_publish_generation_completed_is_the_only_path_that_announces_completed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node = _make_node(monkeypatch)
    published = _Published(node)

    node._publish_generation_state(status=STATUS_COMPLETED)
    assert STATUS_COMPLETED not in published.statuses()
    assert node.parameter_output_values.get("generation_status") != STATUS_COMPLETED

    node._publish_generation_completed()
    assert published.statuses() == [STATUS_COMPLETED]
    assert node.parameter_output_values["generation_status"] == STATUS_COMPLETED


def test_publish_generation_completed_reasserts_the_cloud_generation_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_parse_result` runs before COMPLETED is announced, and some subclasses clobber the ID.

    Four subclasses (wan_*, qwen_*) overwrite `generation_id` with a provider-side task id in
    `_parse_result`. That value 404s when the editor reconciles it against the cloud, and the
    field is now visible and copyable, so the base class restores the truth last.
    """
    node = _make_node(monkeypatch)
    published = _Published(node)

    # Stand in for a subclass's _parse_result writing a provider task id.
    node.parameter_output_values["generation_id"] = "provider-task-9999"

    node._publish_generation_completed("gen-real")

    assert node.parameter_output_values["generation_id"] == "gen-real"
    assert published.ids() == ["gen-real"]


@pytest.mark.asyncio
async def test_the_download_helper_reports_failure_without_raising() -> None:
    """The premise of the two tests below, pinned separately so they cannot pass vacuously.

    `_download_and_save` is how ~20 subclasses put a result on the node, and a download that
    dies part-way is the likeliest failure *after* the generation has been billed. It handles
    that itself: output cleared, `_set_status_results(was_successful=False)`, no exception. So a
    caller that treats "`_parse_result` returned" as "the result is on the node" is wrong on
    exactly the path where being wrong costs the user their recovery affordance.
    """
    node = Flux2ImageGeneration(name="Flux2")

    async def _explode(_url: str) -> bytes:
        msg = "connection reset mid-download"
        raise RuntimeError(msg)

    node._download_bytes_from_url = _explode  # type: ignore[method-assign]

    await node._download_and_save(
        "https://provider.example/asset.png",
        "image",
        lambda value, name: {"value": value, "name": name},
        media_kind="image",
    )

    assert node._execution_succeeded is False
    assert node.parameter_output_values["image"] is None


@pytest.mark.asyncio
async def test_completed_is_withheld_when_the_result_did_not_land(monkeypatch: pytest.MonkeyPatch) -> None:
    """A reported (not raised) parse failure must not announce COMPLETED.

    The editor stops offering recovery the moment it sees COMPLETED, so publishing it for a node
    holding no result hides the affordance in the one case it is needed: the generation is billed,
    the provider URL is still live, and the user has to be able to try again. The ID has to stay
    published for the same reason.
    """
    node = Flux2ImageGeneration(name="Flux2")
    published = _Published(node)

    async def _parse_result(_result_json: dict[str, Any], _generation_id: str) -> None:
        # What `_download_and_save` does on a failed download, minus the network.
        node.parameter_output_values["image"] = None
        node._set_status_results(was_successful=False, result_details="could not be retrieved")

    node._parse_result = _parse_result  # type: ignore[method-assign]
    node._fetch_generation_result = _returns({"images": [{"url": "https://provider.example/x.png"}]})  # type: ignore[method-assign]
    node._submit_and_poll = _returns(("gen-billed", {"status": STATUS_COMPLETED}))  # type: ignore[method-assign]
    node._validate_api_key = lambda: "fake-key"  # type: ignore[method-assign]
    node._prepare_user_auth_info = lambda: None  # type: ignore[method-assign]

    await node._process_generation()

    assert STATUS_COMPLETED not in published.statuses()
    assert node.parameter_output_values.get("generation_status") != STATUS_COMPLETED
    assert node.parameter_output_values["generation_id"] == "gen-billed"
    assert published.ids()[-1] == "gen-billed"


@pytest.mark.asyncio
async def test_refresh_does_not_report_success_when_the_result_did_not_land() -> None:
    """Same failure via Refresh, where the wrong answer was also *stated* to the user.

    `_refresh_completed` overwrote the subclass's verdict with "completed and result was
    retrieved", so a failed download reported success and lost the provider URL the subclass had
    just put in `result_details` — the one thing the user could still act on.
    """
    node = Flux2ImageGeneration(name="Flux2")
    published = _Published(node)

    async def _parse_result(_result_json: dict[str, Any], _generation_id: str) -> None:
        node._set_status_results(
            was_successful=False,
            result_details="generation completed upstream but the image could not be retrieved. Provider URL: https://p/x",
        )

    node._parse_result = _parse_result  # type: ignore[method-assign]
    node._fetch_generation_result = _returns({"images": [{"url": "https://p/x"}]})  # type: ignore[method-assign]

    await node._refresh_completed("gen-billed")

    assert node._execution_succeeded is False
    assert "could not be retrieved" in (node.parameter_output_values.get("result_details") or "")
    assert STATUS_COMPLETED not in published.statuses()
    assert published.ids()[-1] == "gen-billed"


@pytest.mark.asyncio
async def test_a_previous_runs_verdict_does_not_withhold_completed() -> None:
    """The gate has to describe *this* parse, or Refresh after any failure never says COMPLETED.

    `_process_generation` clears the verdict at the top, but the Refresh path does not and sets it
    on nearly every branch — so a node that failed, then recovered via Refresh, would be stuck
    with the old FALSE and never announce the result it now holds.
    """
    node = Flux2ImageGeneration(name="Flux2")
    published = _Published(node)
    node._set_status_results(was_successful=False, result_details="an earlier run failed")

    async def _parse_result(_result_json: dict[str, Any], _generation_id: str) -> None:
        node.parameter_output_values["image"] = {"value": "saved"}

    node._parse_result = _parse_result  # type: ignore[method-assign]

    landed = await node._parse_result_onto_node({"images": []}, "gen-recovered")

    assert landed is True
    node._publish_generation_completed("gen-recovered")
    assert published.statuses() == [STATUS_COMPLETED]


@pytest.mark.parametrize("failure", ["transport", "http_status"])
@pytest.mark.asyncio
async def test_polling_error_exhaustion_preserves_generation_id(monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    """Giving up because polling kept erroring must still leave the ID recoverable.

    The two error branches called _set_safe_defaults() and returned without restoring
    generation_id, so a run that died to repeated transport errors lost the pointer to
    billable work in exactly the way the timeout path was fixed not to. Run against the real
    _set_safe_defaults, since that is what does the blanking.

    Parametrised because the branches are separate code with identical restore logic: the
    `httpx.HTTPStatusError` handler is the one a 502 from the proxy takes, and testing only the
    generic one would leave it free to regress.
    """
    node = _make_node(monkeypatch, real_safe_defaults=True)
    published = _Published(node)
    _no_sleep(monkeypatch)

    class ExplodingClient:
        async def __aenter__(self) -> ExplodingClient:
            return self

        async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            return None

        async def get(self, url: str, headers: dict[str, str], timeout: int) -> Any:  # noqa: ARG002
            if failure == "http_status":
                request = httpx.Request("GET", url)
                response = httpx.Response(502, text="bad gateway", request=request)
                raise httpx.HTTPStatusError("502", request=request, response=response)
            msg = "connection reset"
            raise RuntimeError(msg)

    monkeypatch.setattr("griptape_nodes_library.proxy.griptape_proxy_node.httpx.AsyncClient", ExplodingClient)

    node.set_parameter_value("timeout", 5)
    result = await node._poll_generation_status("gen-errored", {"Authorization": "Bearer k"})

    assert result is None
    assert node.parameter_output_values["generation_id"] == "gen-errored"
    assert "gen-errored" in published.ids()


@pytest.mark.asyncio
async def test_parse_failure_leaves_the_id_recoverable_on_both_channels(monkeypatch: pytest.MonkeyPatch) -> None:
    """A parse failure that crashes the flow must still leave the ID recoverable.

    Two orderings have to be right at once, and they pull in opposite directions:

    * `_handle_result_parsing_error` ends in `_handle_failure_exception`, which re-raises
      when the Failed output has no outgoing connection — so the publish must happen
      *before* that, or it is skipped in exactly the case that matters.
    * the same handler calls `_set_safe_defaults()`, which blanks `generation_id` in every
      subclass — so the publish must happen *after* that, or the output value is wiped a
      moment later and the node's own Refresh button reports nothing to recover.

    Hence `real_safe_defaults=True`: with the stub in place this test passes even when the
    ID has been blanked, which is precisely how the bug survived the first review.
    """
    node = _make_node(monkeypatch, real_safe_defaults=True)
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

    # Announced to the tray...
    assert published.ids()[-1] == "gen-unparseable"
    # ...and still on the node afterwards, so the in-node Refresh button can act on it.
    assert node.parameter_output_values["generation_id"] == "gen-unparseable"
    # Short of COMPLETED, so the editor keeps offering recovery.
    assert STATUS_COMPLETED not in published.statuses()
    # And not relabelled ERRORED: in the cloud's vocabulary that means "internal failure,
    # nothing is billed", which is the opposite of a generation that ran and was charged.
    assert STATUS_ERRORED not in published.statuses()


@pytest.mark.asyncio
async def test_result_fetch_failure_keeps_the_id_and_does_not_claim_completed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The likeliest recovery path of all: the download fails after the generation is billed.

    `_fetch_generation_result` calls `_set_safe_defaults()`, which blanks `generation_id`.
    Leaving that unpatched stranded paid work: nothing on the node could name the generation
    to retry, on the one route by which a raw-bytes result can be recovered at all.

    The COMPLETED assertions pin the other half. The poll loop has seen COMPLETED from the
    cloud by the time this runs and the obvious implementation publishes it there; withholding
    it until `_parse_result` succeeds is what keeps the editor offering recovery for a result
    the node does not hold. Note the badge cannot merely be *restored* to something else here —
    `_publish_generation_state` drops COMPLETED on both channels, so the only way it reaches the
    node is `_publish_generation_completed`, which this path must never reach.
    """
    node = _make_node(monkeypatch, real_safe_defaults=True)
    published = _Published(node)
    node._validate_api_key = lambda: "fake-key"  # type: ignore[method-assign]

    async def _submit_and_poll(_headers: dict[str, str]) -> tuple[str, dict[str, Any]]:
        # What the real poll loop leaves behind on seeing COMPLETED: the last non-terminal
        # status on the badge, and COMPLETED reported only in the returned payload.
        node._publish_generation_state(status=STATUS_RUNNING)
        return "gen-undownloadable", {"status": STATUS_COMPLETED}

    class ExplodingClient:
        async def __aenter__(self) -> ExplodingClient:
            return self

        async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            return None

        async def get(self, url: str, headers: dict[str, str], timeout: int) -> Any:  # noqa: ARG002
            msg = "connection reset mid-download"
            raise RuntimeError(msg)

    node._submit_and_poll = _submit_and_poll  # type: ignore[method-assign]
    monkeypatch.setattr("griptape_nodes_library.proxy.griptape_proxy_node.httpx.AsyncClient", ExplodingClient)

    await node._process_generation()

    assert node.parameter_output_values["generation_id"] == "gen-undownloadable"
    assert published.ids()[-1] == "gen-undownloadable"
    # Never claimed the node has a result it failed to download; the badge is left at the last
    # status the cloud actually reported, which is short of COMPLETED and so keeps recovery on
    # offer.
    assert node.parameter_output_values["generation_status"] == STATUS_RUNNING
    assert STATUS_COMPLETED not in published.statuses()


@pytest.mark.asyncio
@pytest.mark.parametrize("node_class", [Flux2ImageGeneration, FluxImageGeneration])
async def test_result_fetch_keeps_the_id_when_the_api_key_disappears(
    monkeypatch: pytest.MonkeyPatch, node_class: type[Any]
) -> None:
    """The last member of the defect class the other fetch branches were fixed for.

    `_fetch_generation_result` re-validates the API key, and the generation is COMPLETED and
    billed by the time it does. If the key is rotated or cleared between the final status poll
    and the result fetch, `_handle_api_key_validation_error` calls `_set_safe_defaults()` and the
    pointer to paid work is gone.

    The restore is a `finally` around the handler call, not a parameter on the handler, and both
    halves of that matter:

    * ordering — `finally` runs after the handler's `_set_safe_defaults()` has blanked the ID, and
      still runs when `_handle_failure_exception` re-raises because the Failed output is
      unconnected. A publish before the call is wiped; one after it is skipped.
    * reach — twelve subclasses override `_handle_api_key_validation_error` with the
      two-argument signature and never call `super()`, so threading a keyword through raised
      `TypeError` on this exact path for those nodes, which is worse than the bug being fixed.
      Hence the parametrisation: `Flux2ImageGeneration` inherits the base handler,
      `FluxImageGeneration` is one of the twelve overriders. Testing only the former is how that
      regression got in — `reportIncompatibleMethodOverride` is off, so pyright does not flag it.
    """
    node = node_class(name="Node")
    node._set_status_results = lambda **_kwargs: None  # type: ignore[method-assign]
    published = _Published(node)

    async def _submit_and_poll(_headers: dict[str, str]) -> tuple[str, dict[str, Any]]:
        node._publish_generation_state(status=STATUS_RUNNING)
        return "gen-keyless", {"status": STATUS_COMPLETED}

    node._submit_and_poll = _submit_and_poll  # type: ignore[method-assign]

    calls = {"n": 0}

    def _validate_api_key() -> str:
        # Succeeds for submission, then the key vanishes before the result fetch.
        calls["n"] += 1
        if calls["n"] > 1:
            msg = "Node is missing GRIPTAPE_API_KEY."
            raise ValueError(msg)
        return "fake-key"

    node._validate_api_key = _validate_api_key  # type: ignore[method-assign]

    # A ValueError and nothing else: a TypeError here means the handler signature drifted.
    with pytest.raises(ValueError, match="missing GRIPTAPE_API_KEY"):
        await node._process_generation()

    assert node.parameter_output_values["generation_id"] == "gen-keyless"
    assert published.ids()[-1] == "gen-keyless"
    assert STATUS_COMPLETED not in published.statuses()


def test_the_api_key_handler_signature_all_subclasses_share() -> None:
    """Pin the contract that the `finally` above exists to respect.

    Twelve subclasses override `_handle_api_key_validation_error` without calling `super()`, so
    the base class cannot add a parameter to it — and with
    `reportIncompatibleMethodOverride = false` in pyproject.toml, nothing but this test would
    notice. If a future change needs to pass state into the handler, those overrides have to move
    first.
    """
    base = inspect.signature(GriptapeProxyNode._handle_api_key_validation_error)
    assert list(base.parameters) == ["self", "e"]
    assert not inspect.iscoroutinefunction(GriptapeProxyNode._handle_api_key_validation_error)

    # Parsed rather than imported: importing all 46 subclasses to read one signature is far more
    # machinery than reading the source, and this also catches a module the test never imports.
    library_root = Path(griptape_nodes_library.__file__).parent
    base_module = library_root / "proxy" / "griptape_proxy_node.py"
    overriders: list[str] = []
    for path in sorted(library_root.rglob("*.py")):
        if path == base_module:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.ClassDef):
                continue
            for item in node.body:
                # `AsyncFunctionDef` too: the base calls this handler synchronously, so an
                # override defined with `async def` returns a coroutine nobody awaits and the
                # failure is never handled at all — the likeliest way this contract breaks next,
                # and invisible to a check that only walks `FunctionDef`.
                if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if item.name != "_handle_api_key_validation_error":
                    continue
                where = f"{path.name}:{node.name}"
                assert isinstance(item, ast.FunctionDef), f"{where} is async; the base calls it synchronously"
                # Arity and kinds, not names: every call site passes `e` positionally, so a
                # subclass is free to call it something else. Adding or removing a parameter is
                # what breaks, and `reportIncompatibleMethodOverride = false` in pyproject.toml
                # means nothing else in the toolchain would notice.
                assert len(item.args.args) == len(base.parameters), f"{where} takes {len(item.args.args)} positional"
                assert not item.args.kwonlyargs, f"{where} adds keyword-only args the base does not have"
                assert item.args.vararg is None and item.args.kwarg is None, f"{where} adds *args/**kwargs"
                overriders.append(where)

    assert overriders, "expected subclass overrides; if they are all gone, this test can go too"


# --- TASK 2: adopt an existing generation ID -------------------------------------------


def test_generation_id_parameter_is_settable_and_visible_for_adoption() -> None:
    """Pasting an ID from the editor's tray is the only way a raw-bytes result reaches a user."""
    node = Flux2ImageGeneration(name="Flux2")
    status_group = node.status_component.get_parameter_group()
    generation_id = next(child for child in status_group.children if child.name == "generation_id")

    # `children` is typed as BaseNodeElement; narrow before reading parameter attributes.
    assert isinstance(generation_id, ParameterString)
    assert ParameterMode.PROPERTY in generation_id.get_mode()
    # Still reported as an output by the node that submitted it.
    assert ParameterMode.OUTPUT in generation_id.get_mode()
    assert generation_id.settable is True
    # Visible, too: the declaration previously carried hide/hide_property, and re-adding
    # either would leave the field settable in principle but unreachable in the editor.
    ui_options = generation_id.ui_options or {}
    assert ui_options.get("hide") is not True
    assert ui_options.get("hide_property") is not True


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


@pytest.mark.asyncio
async def test_submitting_retires_a_previously_pasted_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """Adoption is one-shot, or it recreates the incident it exists to fix.

    A pasted ID lands in `parameter_values`, which nothing else in the class writes, so
    without retiring it the *first* ID ever typed outranks the node's own `generation_id`
    forever — and the next Refresh overwrites the real ID with the stale one. The sequence
    below is the one that loses paid work: adopt `gen-OLD`, run the node so `gen-NEW` is
    billed, then Refresh and watch `gen-NEW` disappear.
    """
    node = _make_node(monkeypatch)

    async def _build_payload() -> dict[str, Any]:
        return {"prompt": "x"}

    async def _submit_generation(_payload: Any, _headers: Any, _model: Any) -> str:
        return "gen-NEW"

    async def _poll(_generation_id: str, _headers: dict[str, str]) -> None:
        return None

    node._build_payload = _build_payload  # type: ignore[method-assign]
    node._get_api_model_id = lambda: "flux-2-pro"  # type: ignore[method-assign]
    node._submit_generation = _submit_generation  # type: ignore[method-assign]
    node._poll_generation_status = _poll  # type: ignore[method-assign]
    node._validate_api_key = lambda: "fake-key"  # type: ignore[method-assign]
    node._prepare_user_auth_info = lambda: None  # type: ignore[method-assign]

    # The user adopted a generation earlier.
    node.set_parameter_value("generation_id", "gen-OLD")
    assert node._resolve_refresh_generation_id() == "gen-OLD"

    # Then ran the node, which submits and bills a new generation.
    await node._process_generation()

    # The paste no longer shadows the node's own work.
    assert node._resolve_refresh_generation_id() == "gen-NEW"
    assert node.parameter_output_values["generation_id"] == "gen-NEW"


@pytest.mark.asyncio
async def test_a_run_that_fails_before_submitting_keeps_the_pasted_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retiring the paste is only safe once there is a new ID to retire it *for*.

    Retiring at the top of the run destroyed it ahead of key validation, payload build and
    submission, so a user who pasted an ID to recover paid work and then accidentally ran the
    node — or ran it with a rotated key — was left with neither ID: the paste was popped and the
    output value was blanked by the same startup publish.
    """
    node = _make_node(monkeypatch)
    node.set_parameter_value("generation_id", "gen-PASTED")

    def _no_key() -> str:
        msg = "missing GT_CLOUD_API_KEY"
        raise ValueError(msg)

    node._validate_api_key = _no_key  # type: ignore[method-assign]
    node._prepare_user_auth_info = lambda: None  # type: ignore[method-assign]
    node._handle_api_key_validation_error = lambda _e: None  # type: ignore[method-assign]

    await node._process_generation()

    assert node._resolve_refresh_generation_id() == "gen-PASTED"


@pytest.mark.asyncio
async def test_a_refused_paste_names_the_generation_it_is_shadowing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every refusal path returns without clearing the field, and the field outranks the output.

    So the user who pastes a bad ID onto a node that has already timed out cannot reach their own
    generation by clicking Refresh again — each click re-reports the refusal, and nothing says the
    field is what is in the way.
    """
    node = _make_node(monkeypatch)
    captured: list[dict[str, Any]] = []
    node._set_status_results = lambda **kwargs: captured.append(kwargs)  # type: ignore[method-assign]

    node.parameter_output_values["generation_id"] = "gen-OWN"
    node.set_parameter_value("generation_id", "https://cloud.griptape.ai/generations/gen-abc")

    await node._refresh_async()

    assert len(captured) == 1
    assert captured[0]["was_successful"] is False
    assert "gen-OWN" in captured[0]["result_details"]
    assert "Clear the `generation_id` field" in captured[0]["result_details"]


@pytest.mark.asyncio
async def test_a_refusal_says_nothing_about_shadowing_when_there_is_nothing_to_shadow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The note is only true when the node has an ID of its own that the paste is hiding."""
    node = _make_node(monkeypatch)
    captured: list[dict[str, Any]] = []
    node._set_status_results = lambda **kwargs: captured.append(kwargs)  # type: ignore[method-assign]

    node.set_parameter_value("generation_id", "gen abc")

    await node._refresh_async()

    assert "Clear the `generation_id` field" not in captured[0]["result_details"]


@pytest.mark.asyncio
async def test_a_publish_failure_does_not_replace_the_api_key_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ID restore runs in a `finally`, so anything it raises would displace the real error.

    `_handle_api_key_validation_error` ends in `_handle_failure_exception`, which re-raises when
    the Failed output is unconnected. An exception escaping the `finally` would replace that
    ValueError with an unrelated one — the user would be told the wrong thing, and the flow would
    route on the wrong failure.
    """
    node = _make_node(monkeypatch, real_safe_defaults=True)

    def _no_key() -> str:
        msg = "missing GRIPTAPE_API_KEY"
        raise ValueError(msg)

    def _explode(**_kwargs: Any) -> None:
        msg = "editor channel is gone"
        raise RuntimeError(msg)

    node._validate_api_key = _no_key  # type: ignore[method-assign]
    node._publish_generation_state = _explode  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="missing GRIPTAPE_API_KEY"):
        await node._fetch_generation_result("gen-keyless")


def test_refresh_accepts_another_variant_of_the_same_node_family(monkeypatch: pytest.MonkeyPatch) -> None:
    """The guard must key off the node class, not the dropdown's current position.

    Every model the node offers shares one `_parse_result`, so refusing `flux-2-flex` because
    the dropdown currently says `flux-2-pro` would block recovery with no other node type to
    send the user to. The candidate set comes from the model-access component's
    `model_choices`, which stores provider model ids and is the node's own list of what it
    can run.
    """
    node = _make_node(monkeypatch)
    node._get_api_model_id = lambda: "flux-2-pro"  # type: ignore[method-assign]

    assert node._refresh_model_mismatch({"model_id": "flux-2-flex"}) is None
    assert node._refresh_model_mismatch({"model_id": "flux-2-klein-9b"}) is None
    # A genuinely foreign model is still refused.
    assert node._refresh_model_mismatch({"model_id": "sora-2"}) is not None


def test_refresh_does_not_false_block_a_path_shaped_model_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """Matching on the last path segment alone would equate unrelated models.

    `kling/v2-1/master` must not be read as a model called `master`.
    """
    node = _make_node(monkeypatch)
    node._get_api_model_id = lambda: "kling/v2-1/master"  # type: ignore[method-assign]
    node._get_catalog_model_id = lambda: "kling/v2-1/master"  # type: ignore[method-assign]
    node._supported_model_ids = lambda: {"kling/v2-1/master"}  # type: ignore[method-assign]

    assert node._refresh_model_mismatch({"model_id": "kling/v2-1/master"}) is None
    assert node._refresh_model_mismatch({"model_id": "wan/v2-2/master"}) is not None


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


def test_the_spec_defined_model_field_outranks_the_friendly_label(monkeypatch: pytest.MonkeyPatch) -> None:
    """`model_id` is the authoritative field; `model` is where a display name shows up.

    Subclasses put a human label under `model` in the request payload (`"Starlight Precise 2.6"`),
    so if the cloud echoes request-side fields back and `model` were consulted first, a display
    name would reach the comparison and manufacture a mismatch against the user's own generation.
    """
    node = _make_node(monkeypatch)
    node._get_api_model_id = lambda: "topaz-video-slp-2.6"  # type: ignore[method-assign]
    node._get_catalog_model_id = lambda: "topaz-video-slp-2.6"  # type: ignore[method-assign]
    node._supported_model_ids = lambda: {"topaz-video-slp-2.6"}  # type: ignore[method-assign]

    status_json = {"model_id": "topaz-video-slp-2.6", "model": "Starlight Precise 2.6"}

    assert node._extract_generation_model_id(status_json) == "topaz-video-slp-2.6"
    assert node._refresh_model_mismatch(status_json) is None


def test_refresh_fails_open_when_the_response_names_no_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never block the only binary-recovery route on a field the API may not send."""
    node = _make_node(monkeypatch)
    node._get_api_model_id = lambda: "flux-2"  # type: ignore[method-assign]

    assert node._refresh_model_mismatch({"status": STATUS_COMPLETED}) is None


def test_refresh_survives_a_node_that_cannot_report_its_model() -> None:
    """An empty candidate set must fail open, not refuse everything.

    `_supported_model_ids` is best-effort by design — it exists to *widen* what Refresh accepts.
    A subclass whose `_get_api_model_id` raises would otherwise turn the widening into a total
    block on the one route by which a raw-bytes result can be recovered.
    """
    node = Flux2ImageGeneration(name="Flux2")

    def _explode() -> str:
        msg = "no model configured"
        raise RuntimeError(msg)

    node._get_api_model_id = _explode  # type: ignore[method-assign]
    node._get_catalog_model_id = _explode  # type: ignore[method-assign]
    node._model_access = None

    assert node._supported_model_ids() == set()
    assert node._refresh_model_mismatch({"model_id": "anything-at-all"}) is None


def test_refresh_accepts_a_generation_after_a_plain_dropdown_moved() -> None:
    """The false-block the family comparison exists to prevent, on a real node.

    `TopazImageEnhance` has no `ModelAccessComponent`, so its candidate set collapses to
    `topaz-{operation}` for whatever the dropdown says *right now* — exactly the basis the guard
    is documented not to use. Submit with `enhance`, time out, flip the dropdown to `denoise`,
    then click Refresh on your own billed generation: comparing exact ids refused it and named a
    `topaz-enhance` node type that does not exist, while `_parse_result` here is
    operation-agnostic and would have succeeded.
    """
    node = TopazImageEnhance(name="Topaz")
    assert node._model_access is None

    node.set_parameter_value("operation", "denoise")
    assert node._get_api_model_id() == "topaz-denoise"

    # Every operation the dropdown offers, not just the convenient ones: three of the nine are
    # themselves hyphenated, and a rule that dropped only the final id segment left those in a
    # different family from the single-word ones — refusing 42 of the 81 pairs while passing a
    # test that happened to pick `enhance`/`denoise`.
    for operation in ("enhance", "enhance-generative", "sharpen-generative", "restore-generative", "matting"):
        assert node._refresh_model_mismatch({"model_id": f"topaz-{operation}"}) is None, operation

    # Cross-provider — the actual footgun — is still refused.
    assert node._refresh_model_mismatch({"model_id": "flux-2-pro"}) is not None
    assert node._refresh_model_mismatch({"model_id": "kling/v2-1/master"}) is not None


def test_refresh_does_not_loosen_the_comparison_for_nodes_that_declare_their_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The leading-segment fallback is only for nodes that cannot enumerate their family.

    A node with a `ModelAccessComponent` already knows every model it can run, so comparing on
    the leading segment there would buy nothing and start accepting foreign models that happen to
    share a prefix.
    """
    node = _make_node(monkeypatch)
    node._supported_model_ids = lambda: {"sora-2"}  # type: ignore[method-assign]

    assert node._model_access is not None
    assert node._refresh_model_mismatch({"model_id": "sora-9"}) is not None
    # The same node with the component removed takes the widened path, so the branch — not some
    # incidental property of these ids — is what makes the assertion above hold.
    node._model_access = None
    assert node._refresh_model_mismatch({"model_id": "sora-9"}) is None


def test_the_refusal_names_models_the_user_could_actually_select(monkeypatch: pytest.MonkeyPatch) -> None:
    """The refusal is the whole value of the guard, so what it lists has to be actionable.

    Model ids carry an operation suffix (`kling:motion-control`) that the comparison strips but
    the message must not: `kling` is not something a user can paste, select, or search for.
    """
    node = _make_node(monkeypatch)
    node._supported_model_ids = lambda: {"kling:motion-control", "kling:standard"}  # type: ignore[method-assign]

    mismatch = node._refresh_model_mismatch({"model_id": "sora-2"})

    assert mismatch is not None
    assert "kling:motion-control" in mismatch
    assert "kling:standard" in mismatch


def test_supported_model_ids_is_every_declared_model_not_the_permitted_subset() -> None:
    """A licence downgrade must not strand a generation the org already paid for.

    That property rests entirely on `model_choices` being the node's *declared* models: an
    OFFER_MODEL denial decorates a dropdown row and gates execution, it does not remove the
    choice. So the load-bearing assertion is the identity against the subclass's own
    `MODEL_OPTIONS` — if a future engine ever made `model_choices` return only the permitted
    subset, that is what breaks, loudly, instead of adoption of a now-denied model quietly
    starting to be refused. (This test cannot itself install a denial: no engine is running, so
    nothing is denied here either way, and asserting over a set nothing has filtered would only
    restate itself.)
    """
    node = Flux2ImageGeneration(name="Flux2")
    assert node._model_access is not None

    assert node._model_access.model_choices == MODEL_OPTIONS
    assert len(MODEL_OPTIONS) > 1, "test needs a multi-model node to be meaningful"
    assert set(MODEL_OPTIONS) <= node._supported_model_ids()

    # Every declared model is adoptable regardless of what the current selection is.
    node.set_parameter_value("model", MODEL_OPTIONS[0])
    for choice in MODEL_OPTIONS:
        assert node._refresh_model_mismatch({"model_id": choice}) is None


@pytest.mark.parametrize(
    "pasted",
    [
        "https://cloud.griptape.ai/generations/gen-abc",
        "../../v1/organizations",
        "gen-abc?expand=all",
        "gen-abc#fragment",
        "gen abc",
        "..",
        ".",
        "%2e%2e%2fv1%2forganizations",
    ],
)
@pytest.mark.asyncio
async def test_refresh_refuses_a_pasted_value_that_is_not_a_generation_id(
    monkeypatch: pytest.MonkeyPatch, pasted: str
) -> None:
    """Making `generation_id` settable means this value is now user-typed.

    Every ID is interpolated into `generations/{id}` and resolved with `urljoin`, which honours
    dot segments, `?` and `#` — so a whole pasted URL or a stray query string silently addresses
    a different endpoint with the user's key attached instead of saying anything useful. Nothing
    should leave the process.
    """
    node = _make_node(monkeypatch, real_safe_defaults=True)

    captured: list[dict[str, Any]] = []
    node._set_status_results = lambda **kwargs: captured.append(kwargs)  # type: ignore[method-assign]

    class NoRequestsClient:
        async def __aenter__(self) -> NoRequestsClient:
            return self

        async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            return None

        async def get(self, url: str, **_kwargs: Any) -> Any:
            msg = f"no request should have been made, got {url}"
            raise AssertionError(msg)

    monkeypatch.setattr("griptape_nodes_library.proxy.griptape_proxy_node.httpx.AsyncClient", NoRequestsClient)
    node.set_parameter_value("generation_id", pasted)

    await node._refresh_async()

    assert len(captured) == 1
    assert captured[0]["was_successful"] is False
    assert "does not look like a generation ID" in captured[0]["result_details"]


def test_plain_generation_ids_are_not_rejected() -> None:
    """The shape check is a denylist for a reason: a false refusal blocks the only route.

    Anything that could address a generation has to pass, including forms this library has not
    seen — so only the characters that redirect the request are rejected.
    """
    node = Flux2ImageGeneration(name="Flux2")

    for candidate in (
        "gen-1",
        "1a2b3c4d-5e6f-7890-abcd-ef1234567890",
        "generation_01JQ8Z9ABCDEF",
        "abc.def",
        "kling:motion-control-42",
    ):
        assert node._unusable_generation_id_reason(candidate) is None


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
    """The cap in `_resolve_timeout_seconds` is the one that has to hold, not the UI's.

    `max_val` installs a Clamp trait that clamps on assignment, so going through
    `set_parameter_value` here would store 86400 before the code under test ever ran and the
    assertion would pass with `min(seconds, MAX_TIMEOUT_SECONDS)` deleted. Written straight
    into `parameter_values` to bypass the trait — which is also the realistic path for a value
    restored from a workflow file saved before the cap existed.
    """
    node = Flux2ImageGeneration(name="Flux2")
    node.parameter_values["timeout"] = MAX_TIMEOUT_SECONDS * 10

    assert node.parameter_values["timeout"] == MAX_TIMEOUT_SECONDS * 10
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
    # Patch the production module's own `_loop_time` seam rather than
    # `asyncio.get_running_loop`, which would swap a real event loop for this stub across the
    # whole stdlib module for the duration of the test.
    monkeypatch.setattr("griptape_nodes_library.proxy.griptape_proxy_node._loop_time", clock.time)

    async def slow_sleep(_: float) -> None:
        clock.advance(60.0)

    monkeypatch.setattr("griptape_nodes_library.proxy.griptape_proxy_node.asyncio.sleep", slow_sleep)

    node.set_parameter_value("timeout", 600)
    result = await node._poll_generation_status("gen-slow", {"Authorization": "Bearer k"})

    assert result is None
    assert node.parameter_output_values["generation_status"] == STATUS_TIMED_OUT
    # 600s of budget at ~60s per iteration: nowhere near the 120-attempt cap. The deadline is
    # also checked at the top of the loop, so the overshoot is bounded by one sleep.
    assert clock.time() <= 600 + 60


@pytest.mark.asyncio
async def test_polling_does_not_overshoot_the_deadline_by_a_whole_extra_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The deadline is checked at the top of the loop as well as after each attempt.

    The post-attempt check runs *before* the sleep, so it cannot see that the sleep pushed
    the clock past the deadline. Without a check on re-entry the loop issues one more request
    — a whole request plus interval beyond what the user asked for. With `timeout=90` and 60s
    per pseudo-sleep, that is the difference between 2 requests and 3.
    """
    node = _make_node(monkeypatch)

    clock = {"t": 0.0}
    monkeypatch.setattr("griptape_nodes_library.proxy.griptape_proxy_node._loop_time", lambda: clock["t"])

    async def slow_sleep(_: float) -> None:
        clock["t"] += 60.0

    monkeypatch.setattr("griptape_nodes_library.proxy.griptape_proxy_node.asyncio.sleep", slow_sleep)

    requests: list[float] = []

    class CountingClient:
        async def __aenter__(self) -> CountingClient:
            return self

        async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            return None

        async def get(self, url: str, headers: dict[str, str], timeout: int) -> Any:  # noqa: ARG002
            requests.append(clock["t"])

            class Response:
                def raise_for_status(self) -> None:
                    return None

                def json(self) -> dict[str, Any]:
                    return {"status": STATUS_RUNNING}

            return Response()

    monkeypatch.setattr("griptape_nodes_library.proxy.griptape_proxy_node.httpx.AsyncClient", CountingClient)

    node.set_parameter_value("timeout", 90)
    await node._poll_generation_status("gen-overshoot", {"Authorization": "Bearer k"})

    assert requests == [0.0, 60.0]


@pytest.mark.asyncio
async def test_timeout_does_not_cancel_the_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cancel only works while QUEUED, and a RUNNING generation bills either way.

    Cancelling on timeout would throw away a result the user has already paid for, so the
    timeout path must never issue a cancel.
    """
    node = _make_node(monkeypatch, real_safe_defaults=True)
    _no_sleep(monkeypatch)

    # A client that records every verb, not just GET. Recording is what makes the assertion
    # real: a client that only implements `get` would turn a cancel into an AttributeError
    # inside the poll loop's blanket `except Exception` and swallow it, leaving every assertion
    # below passing. Cancel is POST /generations/{id}/cancel, so `post` has to exist *and* be
    # observed — that holds now and when PR #550 adds `_request_generation_cancel`.
    requests: list[tuple[str, str]] = []

    class RecordingClient:
        async def __aenter__(self) -> RecordingClient:
            return self

        async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            return None

        async def get(self, url: str, headers: dict[str, str], timeout: int) -> Any:  # noqa: ARG002
            requests.append(("GET", url))

            class Response:
                def raise_for_status(self) -> None:
                    return None

                def json(self) -> dict[str, Any]:
                    return {"status": STATUS_RUNNING}

            return Response()

        async def post(self, url: str, **_kwargs: Any) -> Any:
            requests.append(("POST", url))
            msg = f"unexpected POST to {url}"
            raise AssertionError(msg)

        async def delete(self, url: str, **_kwargs: Any) -> Any:
            requests.append(("DELETE", url))
            msg = f"unexpected DELETE to {url}"
            raise AssertionError(msg)

    monkeypatch.setattr("griptape_nodes_library.proxy.griptape_proxy_node.httpx.AsyncClient", RecordingClient)

    node.set_parameter_value("timeout", 15)
    await node._poll_generation_status("gen-running", {"Authorization": "Bearer k"})

    # Polled more than once, so the loop really ran rather than falling straight out.
    assert len(requests) > 1
    assert {verb for verb, _url in requests} == {"GET"}
    assert not any(url.endswith("/cancel") for _verb, url in requests)
    assert node.parameter_output_values["generation_status"] == STATUS_TIMED_OUT
    assert node.parameter_output_values["generation_id"] == "gen-running"
