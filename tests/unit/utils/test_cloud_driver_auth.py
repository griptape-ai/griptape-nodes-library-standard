"""Attribution reaches the `griptape` framework drivers, not just this library's own requests.

`test_attribution_headers.py` polices the requests this library builds itself. It cannot see
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
from griptape_nodes_library.utils.attribution_headers import build_attribution_headers
from griptape_nodes_library.utils.cloud_driver_auth import cloud_driver_auth

LIBRARY_ROOT = Path(griptape_nodes_library.__file__).parent

_TOKEN = "gt-the-credential"  # noqa: S105

# Files allowed to construct a Cloud driver without spreading `cloud_driver_auth()`, and why.
# Each still has to reference the helper somewhere -- an exemption is a different shape, not a
# pass. Verified below, so an entry cannot quietly become a hole.
UNSPREAD_CONSTRUCTIONS = {
    LIBRARY_ROOT
    / "config"
    / "prompt"
    / "griptape_cloud_prompt.py": "**all_kwargs; helper lands via specific_args.update()",
    LIBRARY_ROOT
    / "config"
    / "image"
    / "griptape_cloud_image_driver.py": "**all_kwargs; helper lands via specific_args.update()",
    LIBRARY_ROOT / "utils" / "agent_utils.py": "FileManagerDriver rejects headers=; assigned after construction",
}


def _cloud_driver_constructions() -> dict[Path, list[tuple[int, bool]]]:
    """Every Cloud driver construction in the library: `{path: [(lineno, spreads_the_helper)]}`.

    Aliases are resolved from the `ImportFrom` binding rather than matched by name, because
    `griptape_cloud_prompt.py` imports the class `as GtGriptapeCloudPromptDriver`.
    """
    found: dict[Path, list[tuple[int, bool]]] = {}
    for path in sorted(LIBRARY_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text())
        bound = {
            alias.asname or alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("griptape.drivers")
            for alias in node.names
            if alias.name.startswith("GriptapeCloud") and alias.name.endswith("Driver")
        }
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            if node.func.id not in bound:
                continue
            spreads = any(
                kw.arg is None
                and isinstance(kw.value, ast.Call)
                and getattr(kw.value.func, "id", None) == "cloud_driver_auth"
                for kw in node.keywords
            )
            found.setdefault(path, []).append((node.lineno, spreads))
    return found


def test_returns_the_two_kwargs_that_have_to_travel_together() -> None:
    """Passing `headers` alone re-triggers the `os.environ` default; see the module docstring."""
    assert cloud_driver_auth(_TOKEN) == {"api_key": _TOKEN, "headers": build_attribution_headers(_TOKEN)}


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

    assert driver.headers == build_attribution_headers(_TOKEN)
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
    driver.headers = build_attribution_headers(_TOKEN)

    assert driver.headers == build_attribution_headers(_TOKEN)


def test_every_cloud_driver_construction_carries_attribution() -> None:
    """A site built without the helper bills against `<system-defaults>`, silently and forever.

    Nothing warns: the platform emits no degradation metric for a missing attribution header, so
    an unconverted site looks exactly like a converted one from the outside.
    """
    unspread = {
        f"{path.relative_to(LIBRARY_ROOT)}:{lineno}"
        for path, calls in _cloud_driver_constructions().items()
        for lineno, spreads in calls
        if not spreads and path not in UNSPREAD_CONSTRUCTIONS
    }

    assert unspread == set()


def test_every_exemption_still_reaches_the_helper() -> None:
    """An exemption is a different shape, not a pass -- so each one must still name the helper."""
    for path, reason in UNSPREAD_CONSTRUCTIONS.items():
        source = path.read_text()
        reaches = "cloud_driver_auth" in source or "build_attribution_headers" in source
        assert reaches, f"{path.relative_to(LIBRARY_ROOT)} is exempt ({reason}) but never builds the headers"


def test_exemptions_are_only_for_sites_that_need_them() -> None:
    """An exempt file that has started spreading the helper everywhere should leave the list."""
    for path in UNSPREAD_CONSTRUCTIONS:
        calls = _cloud_driver_constructions().get(path, [])
        assert any(not spreads for _, spreads in calls), (
            f"{path.relative_to(LIBRARY_ROOT)} no longer needs its exemption; drop it"
        )


def test_driver_credentials_are_not_serialized() -> None:
    """The gap `cloud_driver_auth` cannot close, pinned so it is not mistaken for covered.

    Neither field carries `serializable` metadata, so `Agent.from_dict` rebuilds a Cloud driver
    from `os.environ` with no attribution header. Every node that deserializes an agent is still
    unattributed. Tracked in griptape-nodes-library-standard#595; if this starts failing, the
    upstream fix has landed and those sites can be revisited.
    """
    for driver_class in (GriptapeCloudPromptDriver, GriptapeCloudImageGenerationDriver):
        fields = {f.name: f for f in attrs.fields(driver_class)}
        assert fields["api_key"].metadata.get("serializable") is None
        assert fields["headers"].metadata.get("serializable") is None
