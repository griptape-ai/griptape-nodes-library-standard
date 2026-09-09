"""Attribution reaches the `griptape` framework drivers, not just this library's own requests.

`test_griptape_cloud_headers.py` polices the requests this library builds itself. It cannot see
these: a node hands an `api_key` to a `griptape` driver and the driver builds `Authorization`
inside the framework, so the factory's dict never reaches the wire. `cloud_driver_auth` is the
bridge, and these tests pin the three things about it that are easy to get wrong -- both kwargs
travel together, the headers really do land on a constructed driver, and the one driver that
rejects the kwarg still does.
"""

from __future__ import annotations

import ast
from pathlib import Path

import attrs
import pytest
from griptape.drivers.file_manager.griptape_cloud import GriptapeCloudFileManagerDriver
from griptape.drivers.image_generation.griptape_cloud import GriptapeCloudImageGenerationDriver
from griptape.drivers.prompt.griptape_cloud import GriptapeCloudPromptDriver

import griptape_nodes_library
import griptape_nodes_library.utils.cloud_driver_auth as cloud_driver_auth_module
from griptape_nodes_library.utils.cloud_driver_auth import cloud_driver_auth
from griptape_nodes_library.utils.griptape_cloud_headers import build_griptape_cloud_headers

LIBRARY_ROOT = Path(griptape_nodes_library.__file__).parent

_TOKEN = "gt-the-credential"  # noqa: S105

# Constructions allowed to skip the `cloud_driver_auth()` spread: `(file, function) -> (how
# many, why)`. Keyed by function and counted, because either alone leaks wider than intended --
# by file the exemption would also cover `agent_utils.build_prompt_driver`, one `ProviderID`
# branch from a Cloud driver of its own; by function it would still cover a *second* construction
# inside `build_tool_from_config`, whose ~120-line `tool_type` dispatch is the likeliest thing
# here to grow one. Each exempt function must also reach the helper itself -- an exemption is a
# different shape, not a pass. All three are verified below.
UNSPREAD_CONSTRUCTIONS = {
    (
        LIBRARY_ROOT / "config" / "prompt" / "griptape_cloud_prompt.py",
        "process",
    ): (1, "**all_kwargs; helper lands via specific_args.update()"),
    (
        LIBRARY_ROOT / "config" / "image" / "griptape_cloud_image_driver.py",
        "process",
    ): (1, "**all_kwargs; helper lands via specific_args.update()"),
    (
        LIBRARY_ROOT / "utils" / "agent_utils.py",
        "build_tool_from_config",
    ): (1, "FileManagerDriver rejects headers=; assigned after construction"),
}


def _enclosing_function(scopes: list[ast.FunctionDef | ast.AsyncFunctionDef], lineno: int) -> str:
    """Name of the innermost function containing `lineno`, so a nested def reports as itself."""
    enclosing = [f for f in scopes if f.lineno <= lineno <= (f.end_lineno or f.lineno)]
    if not enclosing:
        return "<module>"
    return min(enclosing, key=lambda f: (f.end_lineno or f.lineno) - f.lineno).name


def _cloud_driver_constructions() -> dict[tuple[Path, str], list[tuple[int, bool]]]:
    """Every Cloud driver construction: `{(path, function): [(lineno, spreads_the_helper)]}`.

    Keyed by enclosing function to match `UNSPREAD_CONSTRUCTIONS`; the lineno is carried only so
    a failure can name the line, and is never what an exemption matches on. Ordered by line, so
    a failure that names several reads in the order the file does.

    Aliases are resolved from the `ImportFrom` binding rather than matched by name, because
    `griptape_cloud_prompt.py` imports the class `as GtGriptapeCloudPromptDriver`.
    """
    found: dict[tuple[Path, str], list[tuple[int, bool]]] = {}
    for path in sorted(LIBRARY_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text())
        bound = {
            alias.asname or alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("griptape.drivers")
            for alias in node.names
            if alias.name.startswith("GriptapeCloud") and alias.name.endswith("Driver")
        }
        scopes = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        # Sorted, because `ast.walk` is breadth-first: a construction nested inside an `if` is
        # yielded before a shallower one on a later line, so a failure message would name the
        # lines in an order the reader cannot find in the file.
        constructions = sorted(
            (
                n
                for n in ast.walk(tree)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in bound
            ),
            key=lambda n: (n.lineno, n.col_offset),
        )
        for node in constructions:
            spreads = any(
                kw.arg is None
                and isinstance(kw.value, ast.Call)
                and getattr(kw.value.func, "id", None) == "cloud_driver_auth"
                for kw in node.keywords
            )
            found.setdefault((path, _enclosing_function(scopes, node.lineno)), []).append((node.lineno, spreads))
    return found


def test_returns_the_two_kwargs_that_have_to_travel_together() -> None:
    """Passing `headers` alone re-triggers the `os.environ` default; see the module docstring."""
    assert cloud_driver_auth(_TOKEN) == {
        "api_key": _TOKEN,
        "headers": build_griptape_cloud_headers(_TOKEN, attribution=True),
    }


def test_resolves_the_credential_when_none_is_given(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cloud_driver_auth_module, "resolve_cloud_api_key", lambda: _TOKEN)

    assert cloud_driver_auth()["api_key"] == _TOKEN


def test_an_explicit_empty_token_is_honored_not_re_resolved(monkeypatch: pytest.MonkeyPatch) -> None:
    """A site that already decided the credential is absent keeps that answer."""
    monkeypatch.setattr(cloud_driver_auth_module, "resolve_cloud_api_key", lambda: "should-not-be-called")

    assert cloud_driver_auth("")["api_key"] == ""


def test_each_call_returns_a_fresh_headers_dict() -> None:
    first, second = cloud_driver_auth(_TOKEN), cloud_driver_auth(_TOKEN)

    assert first["headers"] == second["headers"]
    assert first["headers"] is not second["headers"]


@pytest.mark.parametrize("driver_class", [GriptapeCloudPromptDriver, GriptapeCloudImageGenerationDriver])
def test_headers_land_on_a_constructed_driver(driver_class: type) -> None:
    """The end-to-end claim: what the factory builds is what the driver will send."""
    driver = driver_class(model="a-model", **cloud_driver_auth(_TOKEN))

    assert driver.headers == build_griptape_cloud_headers(_TOKEN, attribution=True)
    assert driver.api_key == _TOKEN


def test_file_manager_driver_still_rejects_the_headers_kwarg() -> None:
    """Pins the upstream quirk that forces `agent_utils` to assign after construction.

    `GriptapeCloudFileManagerDriver` declares `headers` as `init=False` while the other two
    drivers accept it -- an inconsistency, given the image driver's own docstring advertises the
    override. If `griptape` fixes it this test fails, which is the signal to drop the workaround.
    """
    assert attrs.fields(GriptapeCloudFileManagerDriver).headers.init is False

    with pytest.raises(TypeError, match="headers"):
        # pyright flags this too, which is the point -- the kwarg does not exist.
        GriptapeCloudFileManagerDriver(api_key=_TOKEN, bucket_id="a-bucket", headers={})  # pyright: ignore[reportCallIssue]


def test_headers_survive_post_construction_assignment() -> None:
    """The `agent_utils` workaround: `@define` is slotted but not frozen, so this is allowed."""
    driver = GriptapeCloudFileManagerDriver.__new__(GriptapeCloudFileManagerDriver)
    driver.headers = build_griptape_cloud_headers(_TOKEN, attribution=True)

    assert driver.headers == build_griptape_cloud_headers(_TOKEN, attribution=True)


def test_every_cloud_driver_construction_carries_attribution() -> None:
    """A site built without the helper bills against `<system-defaults>`, silently and forever.

    Nothing warns: the platform emits no degradation metric for a missing attribution header, so
    an unconverted site looks exactly like a converted one from the outside.
    """
    unspread = {
        f"{path.relative_to(LIBRARY_ROOT)}:{lineno} ({function})"
        for (path, function), calls in _cloud_driver_constructions().items()
        for lineno, spreads in calls
        if not spreads and (path, function) not in UNSPREAD_CONSTRUCTIONS
    }

    assert unspread == set()


def test_every_exemption_still_reaches_the_helper() -> None:
    """An exemption is a different shape, not a pass -- so each one must still name the helper.

    Scoped to the exempt function, not its file: a sibling function's use of the helper says
    nothing about whether *this* construction is attributed.
    """
    for (path, function), (_expected, reason) in UNSPREAD_CONSTRUCTIONS.items():
        tree = ast.parse(path.read_text())
        scopes = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        names = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and _enclosing_function(scopes, node.lineno) == function
        }
        reaches = bool(names & {"cloud_driver_auth", "build_griptape_cloud_headers"})
        assert reaches, f"{path.relative_to(LIBRARY_ROOT)}:{function} is exempt ({reason}) but never builds the headers"


def test_exemptions_cover_exactly_the_constructions_they_were_written_for() -> None:
    """Too few and the exemption is stale; too many and it is silently covering a new site.

    The second half bites: a driver added to an already-exempt function inherits its pass, falls
    back to `os.environ["GT_CLOUD_API_KEY"]` -- which the engine plants as `""` -- and bills
    unattributed behind a 401, with no server-side metric for either.
    """
    for (path, function), (expected, reason) in UNSPREAD_CONSTRUCTIONS.items():
        calls = _cloud_driver_constructions().get((path, function), [])
        unspread = [lineno for lineno, spreads in calls if not spreads]
        assert len(unspread) == expected, (
            f"{path.relative_to(LIBRARY_ROOT)}:{function} is exempt for {expected} construction(s) "
            f"({reason}) but has {len(unspread)} at lines {unspread}"
        )


def test_driver_credentials_are_not_serialized() -> None:
    """The gap `cloud_driver_auth` cannot close, pinned so the workaround is not mistaken for a fix.

    Neither field carries `serializable` metadata, so `Agent.from_dict` alone rebuilds a Cloud
    driver from `os.environ` with no attribution header. `agent_utils._restored_cloud_credentials`
    covers the deserializing nodes by re-injecting both before `from_dict` runs -- a repair at
    load, which is the right layer: marking either field serializable would write the raw
    credential into every saved workflow JSON on disk.

    If this starts failing, upstream has changed its mind about that, and the security of every
    saved workflow needs re-examining before the injection is dropped.
    """
    for driver_class in (GriptapeCloudPromptDriver, GriptapeCloudImageGenerationDriver):
        fields = {f.name: f for f in attrs.fields(driver_class)}
        assert fields["api_key"].metadata.get("serializable") is None
        assert fields["headers"].metadata.get("serializable") is None
