from __future__ import annotations

import ast
import asyncio
import collections
import inspect
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

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

    `Attribute` funcs count as well as `Name` ones. `headers_module.build_griptape_cloud_headers(...)`
    is the same call site as the bare name, and reading only `.id` would let a site written that
    way miss `CLOUD_HEADER_CALLS` entirely -- the map is asserted whole, but a call it never
    collected is not an absence it can see. Both test files here already import the module under
    an alias, so the spelling is house style rather than a hypothetical.
    """

    def called(node: ast.Call) -> str | None:
        func = node.func
        if isinstance(func, ast.Name):
            return func.id
        return func.attr if isinstance(func, ast.Attribute) else None

    return sorted(
        (node for node in ast.walk(tree) if isinstance(node, ast.Call) and called(node) in names),
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


def test_a_module_qualified_call_is_collected_too() -> None:
    """The guardrails are whole-map assertions, so a call they never collect is a silent pass.

    A site spelled `headers_module.build_griptape_cloud_headers(...)` would be absent from
    `CLOUD_HEADER_CALLS` rather than wrong in it, and absent from the spelling check rather
    than mismatched -- both tests would stay green while the call shipped unattributed or
    blocking. Two test files in this repo already use the alias import, so this is the
    spelling a new site is most likely to reach for.
    """
    source = "def f():\n    m.b(1)\n    b(2)\n    other.c(3)\n"

    assert [ast.unparse(call) for call in _calls_named(ast.parse(source), "b")] == ["m.b(1)", "b(2)"]


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

    Direct calls only, and lexical `async def` only. Two gaps follow, and neither is this
    test's to close:

    A coroutine that reaches the sync builder through a sync helper passes here, because the
    name in the body is the helper's. `test_no_coroutine_reaches_a_sync_build_indirectly` below
    is the companion that walks those hops, and it holds the two sites that already do.

    "Runs on the event loop" is wider than `async def`. A node's `process()` runs there too, as
    do its `__init__` and its value and connection hooks, so a sync `cloud_driver_auth()` in any
    of them parks the loop exactly as the case above does.
    `test_no_sync_entry_point_reaches_a_build_unrecorded` is the companion that counts those;
    closing them means an async sibling for `cloud_driver_auth` and an async entry point at each
    site, which is a change to the nodes rather than to the check.
    """
    mismatched = set()
    for path in sorted(LIBRARY_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text())
        scopes = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        for node in _calls_named(tree, ASYNC_FACTORY, *BLOCKING_IN_A_COROUTINE):
            enclosing = [f for f in scopes if f.lineno <= node.lineno <= (f.end_lineno or f.lineno)]
            # Innermost wins: a plain `def` nested in an `async def` blocks only its own thread.
            scope = min(enclosing, key=lambda f: (f.end_lineno or f.lineno) - f.lineno) if enclosing else None
            func = node.func
            called = func.id if isinstance(func, ast.Name) else func.attr  # type: ignore[union-attr]  # pyright: ignore[reportAttributeAccessIssue]
            if isinstance(scope, ast.AsyncFunctionDef):
                complaint = BLOCKING_IN_A_COROUTINE.get(called, "")
            else:
                complaint = f"only a coroutine should call {ASYNC_FACTORY}" if called == ASYNC_FACTORY else ""
            if complaint:
                where = f"{path.relative_to(LIBRARY_ROOT).as_posix()}:{node.lineno}"
                mismatched.add(f"{where} ({scope.name if scope else '<module>'}) calls {called} -- {complaint}")

    assert mismatched == set()


# Coroutines that reach a sync Cloud header build through a sync helper, and why each is still
# here. Every one parks the engine event loop for the length of an engine round trip -- on a
# worker, a forwarded request to the orchestrator, and the full `_TIMEOUT_SECONDS` when the
# orchestrator is wedged. Recorded rather than fixed because the fix is not in this layer: each
# needs an async sibling for the helper it calls, and `cloud_driver_auth` has none yet.
COROUTINES_THAT_BLOCK_TRANSITIVELY = {
    "audio/transcribe_audio.py:313 (_parse_result)": "unwrap_agent -> _restored_cloud_credentials, once per Cloud driver dict in the agent",
    "video/split_video.py:539 (aprocess)": "_parse_timecodes -> _parse_timecodes_with_agent -> cloud_driver_auth",
}


class _Function(NamedTuple):
    """One function definition: its name, whether it is a coroutine, and what it reaches."""

    name: str
    is_async: bool
    calls: frozenset[str]
    # Whether the body itself calls the *sync* factory with `attribution=True`. That is the
    # only build that dispatches: `build_griptape_cloud_headers` guards `attribution_header()`
    # behind `if attribution`, so an `attribution=False` site returns a plain dict and touches
    # the engine not at all, and the async factory awaits rather than parking the loop.
    attributes: bool


class _Reach(NamedTuple):
    """Function names that reach an attributing sync build, split by how each one resolves.

    Two buckets because this is a bare-name call graph with no receiver types. A name defined
    once in the whole library resolves anywhere (`everywhere`); a name defined once within a
    single file resolves only for calls made from that file (`in_file`), which is what keeps
    `agents/memory/`'s two `_get_agent` definitions from being confused for each other. A name
    defined twice inside one file resolves nowhere and is dropped -- there are none today, and
    the rule is here so that adding one fails loudly rather than resolving arbitrarily.
    """

    everywhere: frozenset[str]
    in_file: frozenset[tuple[str, str]]

    def reached_from(self, path: str, calls: frozenset[str]) -> bool:
        return bool(calls & self.everywhere) or any((path, call) in self.in_file for call in calls)


def _sync_helpers_that_attribute() -> _Reach:
    """Sync functions that reach an attributing build, directly or through another one.

    Seeded on the *sites that attribute* rather than on the factory's name, which is the
    difference between this and a call graph rooted at `build_griptape_cloud_headers`. Rooting
    it at the name counts `_list_models`, `get_bucket_list` and `check_provider_asset_access`,
    all three of which build with `attribution=False` and therefore dispatch nothing. That
    over-count is invisible while the consumer only asks about `process` bodies -- none of the
    three is reachable from one -- and produces four bogus entries the moment the consumer asks
    about lifecycle hooks, where all three are reached from an `__init__` or an
    `after_value_set`. The seed is the honest root: a stall starts where a dispatch does.
    """
    functions = _library_functions()
    by_name = collections.Counter(function.name for function in functions.values())
    unique = {name for name, count in by_name.items() if count == 1}
    per_file = collections.Counter((path, function.name) for (path, _), function in functions.items())

    def classify(reach: _Reach, path: str, name: str) -> _Reach:
        if name in unique:
            return reach._replace(everywhere=reach.everywhere | {name})
        if per_file[(path, name)] == 1:
            return reach._replace(in_file=reach.in_file | {(path, name)})
        return reach

    reach = _Reach(frozenset(), frozenset())
    for (path, _), function in functions.items():
        if function.attributes and not function.is_async:
            reach = classify(reach, path, function.name)

    while True:
        grown = reach
        for (path, _), function in functions.items():
            if not function.is_async and reach.reached_from(path, function.calls):
                grown = classify(grown, path, function.name)
        if grown == reach:
            return reach
        reach = grown


def _library_functions() -> dict[tuple[str, int], _Function]:
    """`{(file, lineno): _Function}`, one entry per function definition.

    Calls made inside a nested `def` are attributed to that def, not to the function it sits in
    -- a sync closure defined in a coroutine blocks only whichever thread runs it.
    """
    functions: dict[tuple[str, int], _Function] = {}
    for path in sorted(LIBRARY_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for function in ast.walk(tree):
            if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            nested = [
                f
                for f in ast.walk(function)
                if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)) and f is not function
            ]
            inside_nested = {line for f in nested for line in range(f.lineno, (f.end_lineno or f.lineno) + 1)}
            own = [
                node for node in ast.walk(function) if isinstance(node, ast.Call) and node.lineno not in inside_nested
            ]
            calls = {
                name for node in own if (name := getattr(node.func, "id", None) or getattr(node.func, "attr", None))
            }
            attributes = any(
                (getattr(node.func, "id", None) or getattr(node.func, "attr", None)) == SYNC_FACTORY
                and any(
                    kw.arg == "attribution" and isinstance(kw.value, ast.Constant) and kw.value.value is True
                    for kw in node.keywords
                )
                for node in own
            )
            key = (path.relative_to(LIBRARY_ROOT).as_posix(), function.lineno)
            functions[key] = _Function(
                function.name, isinstance(function, ast.AsyncFunctionDef), frozenset(calls), attributes
            )
    return functions


def test_no_coroutine_reaches_a_sync_build_indirectly() -> None:
    """A sync helper between the coroutine and the builder hides the stall from the direct check.

    `test_the_spelling_matches_the_caller` reads the names in the body, so
    `async def _parse_result` calling `unwrap_agent` looks clean -- the blocking build is two
    hops down. The stall is the same one either way: the loop stops until the engine answers,
    and on a worker that is a round trip to the orchestrator.

    Asserted whole rather than as a floor, so a coroutine that stops blocking has to be removed
    from the map here. A stale entry would leave the next added site looking accounted for.
    """
    reach = _sync_helpers_that_attribute()
    blocked = {
        f"{path}:{lineno} ({function.name})"
        for (path, lineno), function in _library_functions().items()
        if function.is_async and reach.reached_from(path, function.calls)
    }

    assert blocked == set(COROUTINES_THAT_BLOCK_TRANSITIVELY)


# The node methods the engine calls on its own event loop, and the request that gets each one
# there. `process` is the familiar one; the rest are why this map is keyed on "entry point"
# rather than on `process`, since a stall costs the same wherever the loop is parked.
#
#   process                      `BaseNode.aprocess` calls it directly (node_types.py:1165), and
#                                for a generator `process` only the *yielded* callable reaches
#                                `to_thread` -- the body between yields is resumed by
#                                `result.send()`, back on the loop.
#   __init__                     construction, via the `CreateNodeRequest` handler, which is a
#                                plain `def` and so runs wherever it was dispatched from
#                                (`event_manager.py:1301` is a bare `return callback(request)`).
#   before/after_value_set       `set_parameter_value` (node_types.py:1034), reached bare -- no
#                                `to_thread` -- from `async def _hydrate_and_run_node_inner`
#                                (`node_manager.py:3379`) on every hydrated parameter.
#   the connection hooks         the connection request handlers, dispatched as above.
#   validate_before_*_run        the pre-run validation pass, likewise.
#
# Listed rather than derived: `BaseNode` defines these, a node overrides the ones it needs, and
# nothing in this library marks them as engine-called. A hook that never appears costs nothing.
LOOP_ENTRY_POINTS = frozenset(
    {
        "__init__",
        "process",
        "before_value_set",
        "after_value_set",
        "before_incoming_connection",
        "after_incoming_connection",
        "before_outgoing_connection",
        "after_outgoing_connection",
        "before_incoming_connection_removed",
        "after_incoming_connection_removed",
        "before_outgoing_connection_removed",
        "after_outgoing_connection_removed",
        "after_settings_changed",
        "validate_before_node_run",
        "validate_before_workflow_run",
    }
)

# Every sync entry point that reaches an attributing build, and the shortest route it takes.
# These park the engine's event loop exactly as the coroutines above do, and for the same
# duration, so `async def` is the wrong test for the stall: this map is the other half of
# `COROUTINES_THAT_BLOCK_TRANSITIVELY` rather than a softer version of it.
#
# Recorded rather than fixed for the same reason as that map: the fix is an async sibling for
# the helper each one calls, and `cloud_driver_auth` has none yet. What the count buys in the
# meantime is visibility -- a build site is a call to a helper's helper, and nothing at the
# entry point names it. Asserted whole so a twenty-third arrives as a failing test.
#
# A route through `unwrap_agent` fires once per Griptape Cloud driver dict in the agent, and
# fires even on the paths passing `require_credential=False`: that flag governs whether a
# missing credential raises, not whether `_restored_cloud_credentials` runs. The four memory
# nodes are the sharp end of that -- they read or rewrite the agent's wire dict and send no
# request at all, so they park the loop for attribution with nothing to attribute.
#
# `random_text.py:178` is the one that is not paid per run. Constructing a `RandomText` builds
# its agent eagerly, so the round trip lands on every construction -- deserializing a saved
# workflow included, where a node the user never runs still waits on the engine.
SYNC_ENTRY_POINTS_THAT_BLOCK = {
    "agents/agent.py:730 (process)": "cloud_driver_auth; build_tools; unwrap_agent -- three routes, each its own round trip",
    "agents/memory/clear_agent_memory.py:25 (process)": "unwrap_agent -> _restored_cloud_credentials; rewrites memory, sends nothing",
    "agents/memory/display_agent_memory.py:83 (process)": "_get_memory_dict -> unwrap_agent; reads memory, sends nothing",
    "agents/memory/replace_item_in_agent_memory.py:170 (after_incoming_connection)": "_update_memory_choices -> _get_agent -> unwrap_agent; on every connection made",
    "agents/memory/replace_item_in_agent_memory.py:196 (after_value_set)": "_update_memory_choices -> _get_agent -> unwrap_agent; on every agent value set",
    "agents/memory/replace_item_in_agent_memory.py:233 (process)": "unwrap_agent -> _restored_cloud_credentials; rewrites memory, sends nothing",
    "agents/memory/summarize_agent_memory.py:62 (process)": "_get_agent -> unwrap_agent",
    "config/image/griptape_cloud_image_driver.py:65 (process)": "cloud_driver_auth",
    "config/prompt/griptape_cloud_prompt.py:111 (process)": "cloud_driver_auth",
    "image/create_image.py:198 (process)": "cloud_driver_auth; unwrap_agent -> _restored_cloud_credentials",
    "image/describe_image.py:342 (process)": "cloud_driver_auth; build_tools; unwrap_agent -- three routes",
    "number/askulator.py:92 (process)": "create_driver -> cloud_driver_auth",
    "tasks/mcp_task.py:342 (process)": "_setup_agent -> _create_driver -> cloud_driver_auth",
    "text/date_and_time.py:80 (process)": "create_driver -> cloud_driver_auth",
    "text/evaluate_text_result.py:163 (process)": "create_driver -> cloud_driver_auth",
    "text/random_text.py:178 (__init__)": "_initialize_agent -> cloud_driver_auth; once per construction, run or not",
    "text/random_text.py:309 (after_value_set)": "_get_random_selection -> _generate_with_agent -> _initialize_agent -> cloud_driver_auth",
    "text/random_text.py:337 (process)": "_get_random_selection -> _generate_with_agent -> _initialize_agent -> cloud_driver_auth",
    "text/scrape_web.py:40 (process)": "create_driver -> cloud_driver_auth",
    "text/search_web.py:132 (process)": "create_driver -> cloud_driver_auth",
    "text/summarize_text_task.py:46 (process)": "create_driver -> cloud_driver_auth",
    "tools/extraction_tool.py:18 (process)": "cloud_driver_auth",
}


def test_no_sync_entry_point_reaches_a_build_unrecorded() -> None:
    """The sync half of the loop-stall census, asserted whole.

    `test_no_coroutine_reaches_a_sync_build_indirectly` catches the `async def` spellings and
    is blind to these, because a sync entry point reads as ordinary blocking code: that the
    engine runs it on the loop is a fact about the caller, not about anything visible here.

    Filtered on `LOOP_ENTRY_POINTS` rather than resolved through the reach closure's own
    resolution rule, which would drop most of these: `process` is defined once per node, and
    the hooks likewise, so neither is ever library-unique. Safe in this direction, because the
    only thing being asked of the name is that the engine is the one calling it.
    """
    reach = _sync_helpers_that_attribute()
    blocking = {
        f"{path}:{lineno} ({function.name})"
        for (path, lineno), function in _library_functions().items()
        if not function.is_async and function.name in LOOP_ENTRY_POINTS and reach.reached_from(path, function.calls)
    }

    assert blocking == set(SYNC_ENTRY_POINTS_THAT_BLOCK)


def test_a_free_build_is_not_a_stall() -> None:
    """Pins the flag-awareness of the seed, which is otherwise invisible in the maps above.

    A census rooted at the factory's name rather than at the attributing call sites reports
    four entry points that dispatch nothing -- two `__init__`s reaching `_list_models` and
    `get_bucket_list`, and the `seedance_human_reference_asset` probe pair reaching
    `check_provider_asset_access`. All four build with `attribution=False`. They are the
    difference between a map that says where the loop stalls and one that says where a dict
    gets built, and nothing else in this file would notice the seed drifting back.
    """
    reach = _sync_helpers_that_attribute()
    free_builders = {"_list_models", "get_bucket_list", "check_provider_asset_access"}

    assert not (free_builders & reach.everywhere)
    assert not {name for _, name in reach.in_file} & free_builders
