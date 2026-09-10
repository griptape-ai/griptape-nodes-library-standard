from __future__ import annotations

import ast
import asyncio
import inspect
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import griptape_nodes_library.utils.griptape_cloud_headers as headers_module
from griptape_nodes_library.utils.griptape_cloud_headers import (
    build_griptape_cloud_headers,
    build_griptape_cloud_headers_async,
)

LIBRARY_ROOT = Path(__file__).parents[3] / "griptape_nodes_library"

# Every Griptape Cloud header build in the library, and whether the call it belongs to incurs
# spend. Keyed by enclosing function so a failure names the offender; one flag per call, in source
# order, so a second call in a listed function lengthens the tuple instead of overwriting the
# first one's answer.
#
# The `False` entries consume no credits, so there is nothing to attribute: two model/bucket
# listings, an asset-access probe, the proxy's two re-reads of a generation already paid for at
# submit, and the poll/cancel dict `_process_generation` builds alongside its billable one.
# Flipping any `True` here to `False` is how spend silently stops being attributed, which is why
# the map is asserted whole rather than as an allowlist. The reverse matters too: `(True, False)`
# for `_process_generation` collapsing back to `(True,)` means the poll loop went back to reusing
# the submit's attributed dict, which is invisible on the wire and free of any test failure but
# this one.
# The `utils/` entries hand the dict to a `griptape` driver rather than to `requests`, by three
# different routes: `cloud_driver_auth` spreads it into a constructor, `build_tool_from_config`
# assigns it after construction, and `_restored_cloud_credentials` writes it into a serialized
# driver dict for `from_dict` to pick up. `test_cloud_driver_auth.py` polices the first two --
# it reads construction sites, so it is blind to the third.
CLOUD_HEADER_CALLS = {
    ("config/prompt/griptape_cloud_prompt.py", "_list_models"): (False,),
    ("proxy/griptape_proxy_node.py", "_fetch_generation_result"): (False,),
    ("proxy/griptape_proxy_node.py", "_process_generation"): (True, False),
    ("proxy/griptape_proxy_node.py", "_refresh_async"): (False,),
    ("proxy/provider_asset_access.py", "check_provider_asset_access"): (False,),
    ("tools/file_manager_tool.py", "get_bucket_list"): (False,),
    ("utils/agent_utils.py", "_restored_cloud_credentials"): (True,),
    ("utils/agent_utils.py", "build_tool_from_config"): (True,),
    ("utils/cloud_driver_auth.py", "cloud_driver_auth"): (True,),
    ("video/omnihuman_video_generation.py", "_auto_detect_masks"): (True,),
    ("video/seedance_common.py", "_append_private_asset"): (True, False),
}

# The two spellings of the factory. Which one a call site must use is a structural rule,
# checked by `test_the_spelling_matches_the_caller` rather than recorded per site above:
# an `async def` takes the async one, everything else the sync one, no exceptions.
SYNC_FACTORY = "build_griptape_cloud_headers"
ASYNC_FACTORY = "build_griptape_cloud_headers_async"

# Everything that parks its thread on the engine, and what a coroutine should do instead.
# `cloud_driver_auth` is here because it wraps the sync factory: it is not a header build, so the
# map above never sees it, and a coroutine calling it stalls the loop with nothing in this file
# pointing at the reason.
BLOCKING_IN_A_COROUTINE = {
    SYNC_FACTORY: f"call {ASYNC_FACTORY}",
    "cloud_driver_auth": f"give it an async sibling over {ASYNC_FACTORY} first",
}

_BASE_HEADERS = {"Authorization": "Bearer tok", "Content-Type": "application/json"}

# Every behavioural claim below is asserted against both spellings. They are one function
# with two calling conventions, and the async one is the copy that gets forgotten -- a
# divergence would show up as an attribution header the coroutine call sites, which are the
# billable ones, quietly stop sending.
_FACTORIES = pytest.mark.parametrize(
    "factory", [build_griptape_cloud_headers, build_griptape_cloud_headers_async], ids=["sync", "async"]
)


def _headers(factory: Callable[..., Any], *args: Any, **kwargs: Any) -> dict[str, str]:
    """Call whichever spelling `factory` is, and return the dict."""
    built = factory(*args, **kwargs)
    return asyncio.run(built) if inspect.iscoroutine(built) else built


def _stub_attribution(monkeypatch: pytest.MonkeyPatch, answer: dict[str, str] | None) -> None:
    """Replace both lookups. `None` fails the test instead of answering."""

    def _sync() -> dict[str, str]:
        if answer is None:
            pytest.fail("asked the engine to attribute a call that spends nothing")
        return answer

    async def _async() -> dict[str, str]:
        return _sync()

    monkeypatch.setattr(headers_module, "attribution_header", _sync)
    monkeypatch.setattr(headers_module, "attribution_header_async", _async)


@_FACTORIES
def test_a_call_that_does_not_spend_sends_only_bearer_and_json(
    factory: Callable[..., Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`False` must not even ask -- the engine round trip is skipped, not just its result.

    There is nothing to attribute on a control-plane call, and the ~2s bound the helper carries
    is not worth paying to be told so. The stub fails rather than returns, so a merge moved
    outside the `if` is caught here instead of showing up as latency on a model listing.
    """
    _stub_attribution(monkeypatch, None)

    assert _headers(factory, "tok", attribution=False) == _BASE_HEADERS


@_FACTORIES
def test_a_call_that_spends_carries_what_the_engine_answered(
    factory: Callable[..., Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole point of #601, in one assertion: the header rides along with the credential."""
    _stub_attribution(monkeypatch, {"X-Griptape-Attribution": "an-envelope"})

    assert _headers(factory, "tok", attribution=True) == {
        **_BASE_HEADERS,
        "X-Griptape-Attribution": "an-envelope",
    }


@_FACTORIES
def test_an_unattributable_call_still_sends_everything_else(
    factory: Callable[..., Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`{}` from the helper is the fallback, so it has to be a no-op rather than a hole.

    This is the shape of every failure the helper absorbs -- no engine, a timeout, a declined
    answer -- and each one has to leave a working Cloud request behind. Cloud reports nothing
    for a missing attribution header, so the cost is a reporting field; dropping `Authorization`
    alongside it would cost the user the call.
    """
    _stub_attribution(monkeypatch, {})

    assert _headers(factory, "tok", attribution=True) == _BASE_HEADERS


@_FACTORIES
def test_attribution_must_be_stated(factory: Callable[..., Any]) -> None:
    """No default, so a new call site cannot inherit one by omission.

    Defaulting to `False` would make an unattributed billable call the quiet outcome, and the
    platform emits no metric for a missing header -- the failure would be invisible on both
    ends. Defaulting to `True` only trades that for over-reporting, which is recoverable but
    still guesses. Thirteen call sites make stating it free.
    """
    with pytest.raises(TypeError):
        _headers(factory, "tok")


@_FACTORIES
def test_each_call_returns_a_fresh_dict(factory: Callable[..., Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """Call sites mutate what they get back (`_submit_generation` adds the BYOK header)."""
    _stub_attribution(monkeypatch, {})

    first = _headers(factory, "tok", attribution=True)
    first["X-Mutated"] = "yes"
    assert "X-Mutated" not in _headers(factory, "tok", attribution=True)


def _builds_an_authorization_header(node: ast.AST) -> bool:
    """Whether `node` names an Authorization header key in a dict literal or a subscript store.

    Reading the AST rather than grepping skips the JSON example string in
    `json/json_schema_from_example.py` and `headers.get("Authorization", "")` reads. A
    plain-constant value does not count either: that is the debug-log redaction in the two
    video nodes (`{**headers, "Authorization": "Bearer ***"}`), which carries no credential.
    Every real build interpolates the token.
    """
    if isinstance(node, ast.Dict):
        # `keys` holds None for `**spread` entries; None is not an ast.Constant, so it drops out.
        return any(
            isinstance(key, ast.Constant) and key.value == "Authorization" and not isinstance(value, ast.Constant)
            for key, value in zip(node.keys, node.values, strict=True)
        )
    return (
        isinstance(node, ast.Subscript)
        and isinstance(node.ctx, ast.Store)
        and isinstance(node.slice, ast.Constant)
        and node.slice.value == "Authorization"
    )


def test_only_the_factory_builds_an_authorization_header() -> None:
    """An inline copy silently opts its call site out of every future shared header."""
    builders = {
        path.relative_to(LIBRARY_ROOT).as_posix()
        for path in sorted(LIBRARY_ROOT.rglob("*.py"))
        if any(_builds_an_authorization_header(node) for node in ast.walk(ast.parse(path.read_text())))
    }
    assert builders == {"utils/griptape_cloud_headers.py"}


def _calls_named(tree: ast.AST, *names: str) -> list[ast.Call]:
    """Every call to any of `names` in `tree`, in source order.

    Sorted, because `ast.walk` is breadth-first and its order is therefore not reading order:
    a call nested inside an `if` is yielded *before* a shallower call on a later line. The
    flag tuples below are positional, so collecting them in walk order would have the map
    disagree with the file it describes -- and disagree only for a function with more than one
    call, which is the single case the tuple exists to handle. `col_offset` orders a line that
    holds two calls.
    """
    return sorted(
        (node for node in ast.walk(tree) if isinstance(node, ast.Call) and getattr(node.func, "id", None) in names),
        key=lambda node: (node.lineno, node.col_offset),
    )


def test_calls_are_collected_in_source_order() -> None:
    """Pins the sort in `_calls_named`, which is invisible until a function grows a second call.

    Shaped like the one function most likely to grow one: a billable build inside a branch,
    then a free build after it. Unsorted, this records `(False, True)` -- the exact inversion
    that would have `CLOUD_HEADER_CALLS` mis-describe which of the two calls spends.
    """
    source = "def f():\n    if cond:\n        a = b(1)\n    c = b(2)\n"

    assert [ast.unparse(call) for call in _calls_named(ast.parse(source), "b")] == ["b(1)", "b(2)"]


def _cloud_header_calls() -> dict[tuple[str, str], tuple[bool | None, ...]]:
    """Every header build in the library: `{(file, function): (flag, per, call)}`.

    Both spellings are collected into the one map, because whether a call spends is the same
    question either way -- and because collecting only the sync name would have every site
    converted to the async factory silently drop out of the map that exists to hold them.

    Accumulated rather than assigned, so two calls in one function stay two entries -- assigning
    would let the second inherit the first's recorded answer, and an unattributed billable call
    is invisible from the server.
    """
    found: dict[tuple[str, str], tuple[bool | None, ...]] = {}
    for path in sorted(LIBRARY_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text())
        scopes = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        for node in _calls_named(tree, SYNC_FACTORY, ASYNC_FACTORY):
            enclosing = [f for f in scopes if f.lineno <= node.lineno <= (f.end_lineno or f.lineno)]
            # Innermost wins, so a nested def is not reported under the function it sits in.
            name = min(enclosing, key=lambda f: (f.end_lineno or f.lineno) - f.lineno).name if enclosing else "<module>"
            # A non-literal (a variable, a ternary) reads as None and fails the map, which is
            # the right answer: whether a site spends has to be legible without running it.
            flag = next(
                (
                    kw.value.value
                    for kw in node.keywords
                    if kw.arg == "attribution"
                    and isinstance(kw.value, ast.Constant)
                    and isinstance(kw.value.value, bool)
                ),
                None,
            )
            key = (path.relative_to(LIBRARY_ROOT).as_posix(), name)
            found[key] = (*found.get(key, ()), flag)
    return found


def test_every_cloud_call_declares_whether_it_spends() -> None:
    """A new call site, or a flipped flag, has to be argued for here rather than merged quietly."""
    assert _cloud_header_calls() == CLOUD_HEADER_CALLS


def test_the_spelling_matches_the_caller() -> None:
    """An `async def` takes the async factory; everything else takes the sync one.

    The sync spelling inside a coroutine is the bug this split exists to prevent: the
    attribution lookup blocks its thread, so on an event loop it stops every other task
    scheduled there for the length of an engine round trip -- and for the whole ~2s bound when
    the engine is wedged. Nothing about the call site looks wrong, and nothing fails; the node
    just goes quiet, which is why this is checked structurally instead of left to review.

    Enforced in both directions. `await` outside a coroutine is a syntax error, so the reverse
    case cannot ship as written -- but a sync helper that grew the async spelling and an
    `asyncio.run` around it would be a thread-blocking call wearing the non-blocking name, and
    this names it.

    Exceptionless on purpose, `attribution=False` sites included. That flag describes today's
    endpoint, and flipping one is a one-word edit; a coroutine left on the sync spelling because
    it happens not to spend today is a loop stall waiting for an unrelated change to arm it.

    Lexical `async def` is the whole rule, and it is narrower than "runs on the event loop".
    A node's `process()` runs on the loop too -- `BaseNode.aprocess` calls it directly, and for
    a generator `process()` only the *yielded* callable reaches `asyncio.to_thread`; the body up
    to the first yield does not. So a sync `cloud_driver_auth()` in a plain `process()` parks the
    loop exactly as the case above does, and this test reports it correct. Closing that gap means
    an async sibling for `cloud_driver_auth` and an `async def process()` at each site, which is
    a change to the nodes rather than to the check -- until then, read a pass here as "no
    coroutine blocks", not as "nothing blocks".
    """
    mismatched = set()
    for path in sorted(LIBRARY_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text())
        scopes = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        for node in _calls_named(tree, ASYNC_FACTORY, *BLOCKING_IN_A_COROUTINE):
            enclosing = [f for f in scopes if f.lineno <= node.lineno <= (f.end_lineno or f.lineno)]
            # Innermost wins: a plain `def` nested in an `async def` blocks only its own thread.
            scope = min(enclosing, key=lambda f: (f.end_lineno or f.lineno) - f.lineno) if enclosing else None
            called = node.func.id  # type: ignore[union-attr]  # pyright: ignore[reportAttributeAccessIssue]
            if isinstance(scope, ast.AsyncFunctionDef):
                complaint = BLOCKING_IN_A_COROUTINE.get(called, "")
            else:
                complaint = f"only a coroutine should call {ASYNC_FACTORY}" if called == ASYNC_FACTORY else ""
            if complaint:
                where = f"{path.relative_to(LIBRARY_ROOT).as_posix()}:{node.lineno}"
                mismatched.add(f"{where} ({scope.name if scope else '<module>'}) calls {called} -- {complaint}")

    assert mismatched == set()
