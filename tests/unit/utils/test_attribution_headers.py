from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

from griptape_nodes_library.utils.attribution_headers import build_attribution_headers

LIBRARY_ROOT = Path(__file__).parents[3] / "griptape_nodes_library"

# Files allowed to spell an Authorization header. `utils/attribution_headers.py` is the
# factory; the rest build headers for requests that consume no credits, so there is nothing
# to attribute: models list, buckets list, asset-access probe, and the proxy's two re-reads
# of a generation already paid for.
#
# `griptape_proxy_node.py` both calls the factory and keeps two inline dicts, so membership
# here is satisfied by the inline dicts alone -- reverting `_process_generation` to inline
# keeps this test green. `test_proxy_node_keeps_exactly_its_two_inline_header_dicts` closes
# that gap by counting per function.
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

    Reading the AST rather than grepping skips the JSON example string in
    ``json/json_schema_from_example.py`` and ``headers.get("Authorization", "")`` reads.
    A plain-constant value does not count either: that is the debug-log redaction in the
    two video nodes (``{**headers, "Authorization": "Bearer ***"}``), which carries no
    credential. Every real build interpolates the token.
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


# The two inline dicts `griptape_proxy_node.py` may keep, counted per function so a failure
# names the offender and a second dict inside a listed function still fails. Both re-read a
# generation already paid for at submit: `_fetch_generation_result` retrieves the finished
# result, `_refresh_async` backs the Refresh button.
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
    """The file-level test cannot see a third dict added here, because this file is an owner."""
    assert (
        _authorization_headers_by_function(LIBRARY_ROOT / "proxy" / "griptape_proxy_node.py") == INLINE_HEADER_FUNCTIONS
    )
