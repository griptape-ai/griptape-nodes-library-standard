from __future__ import annotations

import ast
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
# `griptape_proxy_node.py` is the weak entry, and knowingly so: it also calls the factory
# from `_process_generation`, so listing it here means this test can no longer tell its two
# intended inline dicts from a third one added later beside them. Nothing else in the set
# has that problem -- the other three never touch the factory. If a check on that file is
# wanted, it needs to count the literals, not just their presence.
#
# Set equality still holds the line elsewhere both ways: a new file building a header
# inline fails this test, and so does moving a billable call *out* of the factory.
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


def test_extra_headers_are_merged() -> None:
    """X-GTC-PROXY-AUTH-INFO rides in per request rather than being read off the node."""
    headers = build_attribution_headers("tok", extra={"X-GTC-PROXY-AUTH-INFO": "user-provider-key"})
    assert headers["X-GTC-PROXY-AUTH-INFO"] == "user-provider-key"
    assert headers["Authorization"] == "Bearer tok"


def test_extra_can_override_a_default() -> None:
    """`extra` merges last, so a caller with an unusual body type is not blocked on a new knob."""
    assert build_attribution_headers("tok", extra={"Content-Type": "multipart/form-data"})["Content-Type"] == (
        "multipart/form-data"
    )


def test_each_call_returns_a_fresh_dict() -> None:
    """Call sites mutate what they get back (`_submit_generation` adds the BYOK header)."""
    first = build_attribution_headers("tok")
    first["X-Mutated"] = "yes"
    assert "X-Mutated" not in build_attribution_headers("tok")


def _files_building_an_authorization_header() -> set[Path]:
    """Files that name an Authorization header key in a dict literal or a subscript store.

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
    matches: set[Path] = set()
    for path in sorted(LIBRARY_ROOT.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            # `keys` holds None for `**spread` entries; None is not an ast.Constant, so it drops out.
            if isinstance(node, ast.Dict) and any(
                isinstance(key, ast.Constant) and key.value == "Authorization" and not isinstance(value, ast.Constant)
                for key, value in zip(node.keys, node.values, strict=True)
            ):
                matches.add(path)
            elif (
                isinstance(node, ast.Subscript)
                and isinstance(node.ctx, ast.Store)
                and isinstance(node.slice, ast.Constant)
                and node.slice.value == "Authorization"
            ):
                matches.add(path)
    return matches


def test_only_the_header_factory_builds_an_authorization_header() -> None:
    """An inline copy silently opts its call site out of every future shared header."""
    assert _files_building_an_authorization_header() == AUTHORIZATION_HEADER_OWNERS
