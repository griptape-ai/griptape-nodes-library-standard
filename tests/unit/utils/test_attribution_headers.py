from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

from griptape_nodes_library.utils.attribution_headers import build_attribution_headers

LIBRARY_ROOT = Path(__file__).parents[3] / "griptape_nodes_library"

# Files allowed to spell an Authorization header, and why each one is here.
#
# `utils/attribution_headers.py` is the factory. Every Cloud call that incurs spend builds its
# headers there, so the attribution header the platform needs for budgeting is one line
# rather than one edit per call site.
#
# The rest are deliberate exemptions, not debt. Each builds headers for a request that
# consumes no credits, so there is no usage to attribute and nothing for the factory to
# add: `griptape_cloud_prompt.py` lists models, `file_manager_tool.py` lists buckets,
# `provider_asset_access.py` probes asset access, and `griptape_proxy_node.py` re-reads a
# generation already paid for in `_fetch_generation_result` and `_refresh_async`.
#
# `griptape_proxy_node.py` is the one entry this test cannot fully police, because it both
# calls the factory (`_process_generation`) and keeps two intended inline dicts. Membership
# here is therefore satisfied by the inline dicts alone: reverting `_process_generation` to
# an inline copy was tried, and the entire suite stayed green. The other four entries have no
# such gap -- none of them touches the factory, so removing their call would drop them from
# the set and fail. Reverting `omnihuman_video_generation.py` to inline was tried too, and
# this test did fail, which is what confines the gap to this single file.
#
# `test_proxy_node_keeps_exactly_its_two_inline_header_dicts` below closes it, by counting how
# many inline header dicts each function in that file builds rather than just checking the file.
#
# What this cannot see: a node that hands an `api_key` to a framework driver
# (`GriptapeCloudPromptDriver`, `GriptapeCloudImageGenerationDriver`,
# `GriptapeCloudFileManagerDriver`) never spells the header itself -- the driver builds it
# inside `griptape`. Those sites are billable and are tracked separately.
AUTHORIZATION_HEADER_OWNERS = {
    LIBRARY_ROOT / "utils" / "attribution_headers.py",
    LIBRARY_ROOT / "config" / "prompt" / "griptape_cloud_prompt.py",
    LIBRARY_ROOT / "proxy" / "provider_asset_access.py",
    LIBRARY_ROOT / "proxy" / "griptape_proxy_node.py",
    LIBRARY_ROOT / "tools" / "file_manager_tool.py",
}


def test_default_headers_are_bearer_and_json() -> None:
    """Pins the exact dict every Cloud call sends, so a new shared header shows up in a diff."""
    assert build_attribution_headers("tok") == {"Authorization": "Bearer tok", "Content-Type": "application/json"}


def test_each_call_returns_a_fresh_dict() -> None:
    """Call sites mutate what they get back (`_submit_generation` adds the BYOK header)."""
    first = build_attribution_headers("tok")
    first["X-Mutated"] = "yes"
    assert "X-Mutated" not in build_attribution_headers("tok")


def _builds_an_authorization_header(node: ast.AST) -> bool:
    """Whether `node` names an Authorization header key in a dict literal or a subscript store.

    A dict literal is how every inline copy was written before centralization; a subscript
    store (``headers["Authorization"] = ...``) is the natural way someone would re-add one
    once the dict form is gone. Reading the AST rather than grepping keeps the JSON example
    string in ``json/json_schema_from_example.py`` -- documentation, not a request -- from
    counting, and likewise ignores ``headers.get("Authorization", "")`` reads.

    A dict whose ``Authorization`` value is a plain constant does not count: with nothing
    interpolated in, it cannot carry a credential. That is what the debug-log redaction
    copies in ``video/veo3_video_generation.py`` and ``video/minimax_hailuo_video_generation.py``
    are (``{**headers, "Authorization": "Bearer ***"}``), and flagging them would drag two
    unrelated video nodes into every change this test governs. Every real header build
    interpolates the token, so it is an ``ast.JoinedStr`` and still counts. The gap this
    leaves -- a token hardcoded as a literal -- is a credential-in-source problem rather
    than an attribution one, and is not what this test is looking for.
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


def _files_building_an_authorization_header() -> set[Path]:
    """Library files that build an Authorization header, by any of the shapes above."""
    return {
        path
        for path in sorted(LIBRARY_ROOT.rglob("*.py"))
        if any(_builds_an_authorization_header(node) for node in ast.walk(ast.parse(path.read_text())))
    }


def test_only_the_header_factory_builds_an_authorization_header() -> None:
    """An inline copy silently opts its call site out of every future shared header."""
    assert _files_building_an_authorization_header() == AUTHORIZATION_HEADER_OWNERS


# The two inline dicts `griptape_proxy_node.py` is allowed to keep, counted per function so
# that a failure both names the offender and catches a second dict added inside a function
# already on the list. Names alone would inherit the presence-not-count blind spot this test
# exists to fix, one level down. Both re-read a generation that was already paid for at
# submit, so there is no fresh usage to attribute:
# `_fetch_generation_result` retrieves a finished result, `_refresh_async` backs the Refresh
# button. Anything else in this file that needs an Authorization header is making a billable
# call and belongs in the factory.
INLINE_HEADER_FUNCTIONS = {"_fetch_generation_result": 1, "_refresh_async": 1}


def _authorization_headers_by_function(path: Path) -> Counter[str]:
    """How many Authorization headers each function in `path` builds inline."""
    tree = ast.parse(path.read_text())
    scopes = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    counts: Counter[str] = Counter()
    for node in ast.walk(tree):
        # Narrowing to the two shapes the predicate matches is also what gives `node` a `lineno`.
        if not isinstance(node, (ast.Dict, ast.Subscript)) or not _builds_an_authorization_header(node):
            continue
        enclosing = [f for f in scopes if f.lineno <= node.lineno <= (f.end_lineno or f.lineno)]
        # Innermost wins, so a nested def is not reported under the function it sits in.
        counts[
            min(enclosing, key=lambda f: (f.end_lineno or f.lineno) - f.lineno).name if enclosing else "<module>"
        ] += 1
    return counts


def test_proxy_node_keeps_exactly_its_two_inline_header_dicts() -> None:
    """The file-level test cannot see a third dict added here, because this file is an owner.

    `griptape_proxy_node.py` earns its place in `AUTHORIZATION_HEADER_OWNERS` from the two
    non-billable dicts alone, so moving `_process_generation` back off the factory leaves that
    test green -- confirmed by trying it. Counting per function is what catches it, and counting
    rather than naming is what also catches a second dict added inside one of the two functions
    already on the list -- the same presence-not-count blind spot, one level down. Both reverts
    confirmed to fail here.
    """
    assert (
        _authorization_headers_by_function(LIBRARY_ROOT / "proxy" / "griptape_proxy_node.py") == INLINE_HEADER_FUNCTIONS
    )
