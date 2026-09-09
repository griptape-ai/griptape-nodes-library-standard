from __future__ import annotations

import ast
from pathlib import Path

import pytest

from griptape_nodes_library.utils.griptape_cloud_headers import build_griptape_cloud_headers

LIBRARY_ROOT = Path(__file__).parents[3] / "griptape_nodes_library"

# Every Griptape Cloud header build in the library, and whether the call it belongs to incurs
# spend. Keyed by enclosing function so a failure names the offender; one flag per call, in source
# order, so a second call in a listed function lengthens the tuple instead of overwriting the
# first one's answer.
#
# The `False` entries consume no credits, so there is nothing to attribute: two model/bucket
# listings, an asset-access probe, and the proxy's two re-reads of a generation already paid
# for at submit. Flipping any `True` here to `False` is how spend silently stops being
# attributed, which is why the map is asserted whole rather than as an allowlist.
# The two `utils/` entries are the framework-driver bridge, where `cloud_driver_auth` passes this
# dict in as the driver's `headers`; `test_cloud_driver_auth.py` polices those construction sites.
CLOUD_HEADER_CALLS = {
    ("config/prompt/griptape_cloud_prompt.py", "_list_models"): (False,),
    ("proxy/griptape_proxy_node.py", "_fetch_generation_result"): (False,),
    ("proxy/griptape_proxy_node.py", "_process_generation"): (True,),
    ("proxy/griptape_proxy_node.py", "_refresh_async"): (False,),
    ("proxy/provider_asset_access.py", "check_provider_asset_access"): (False,),
    ("tools/file_manager_tool.py", "get_bucket_list"): (False,),
    ("utils/agent_utils.py", "build_tool_from_config"): (True,),
    ("utils/cloud_driver_auth.py", "cloud_driver_auth"): (True,),
    ("video/omnihuman_video_generation.py", "_auto_detect_masks"): (True,),
    ("video/seedance_common.py", "_append_private_asset"): (True,),
}


@pytest.mark.parametrize("attribution", [True, False])
def test_headers_are_bearer_and_json(attribution: bool) -> None:
    """Pins the exact dict every Cloud call sends, so a new shared header shows up in a diff.

    Both branches are identical today; the attribution header lands on the `True` branch
    under #601, at which point this test is what records the difference.
    """
    assert build_griptape_cloud_headers("tok", attribution=attribution) == {
        "Authorization": "Bearer tok",
        "Content-Type": "application/json",
    }


def test_attribution_must_be_stated() -> None:
    """No default, so a new call site cannot inherit one by omission.

    Defaulting to `False` would make an unattributed billable call the quiet outcome, and the
    platform emits no metric for a missing header -- the failure would be invisible on both
    ends. Defaulting to `True` only trades that for over-reporting, which is recoverable but
    still guesses. Ten call sites make stating it free.
    """
    with pytest.raises(TypeError):
        build_griptape_cloud_headers("tok")  # type: ignore[call-arg]  # pyright: ignore[reportCallIssue]


def test_each_call_returns_a_fresh_dict() -> None:
    """Call sites mutate what they get back (`_submit_generation` adds the BYOK header)."""
    first = build_griptape_cloud_headers("tok", attribution=True)
    first["X-Mutated"] = "yes"
    assert "X-Mutated" not in build_griptape_cloud_headers("tok", attribution=True)


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


def _cloud_header_calls() -> dict[tuple[str, str], tuple[bool | None, ...]]:
    """Every `build_griptape_cloud_headers` call: `{(file, function): (flag, per, call)}`.

    Accumulated rather than assigned, so two calls in one function stay two entries -- assigning
    would let the second inherit the first's recorded answer, and an unattributed billable call
    is invisible from the server.
    """
    found: dict[tuple[str, str], tuple[bool | None, ...]] = {}
    for path in sorted(LIBRARY_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text())
        scopes = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or getattr(node.func, "id", None) != "build_griptape_cloud_headers":
                continue
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
