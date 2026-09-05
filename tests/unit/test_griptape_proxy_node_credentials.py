from __future__ import annotations

import ast
from pathlib import Path

import pytest

import griptape_nodes_library.proxy.provider_asset_access as access_module
from griptape_nodes_library.image.google_image_generation import GoogleImageGeneration
from griptape_nodes_library.proxy.provider_asset_access import (
    API_KEY_NAME,
    LICENSE_SECRET_NAME,
    PROXY_API_KEY_ENV_VAR,
)
from griptape_nodes_library.utils.attribution_headers import build_attribution_headers
from griptape_nodes_library.video.omnihuman_subject_detection import OmnihumanSubjectDetection

LIBRARY_ROOT = Path(__file__).parents[2] / "griptape_nodes_library"

# The two classes that may define a credential check: the config drivers' generic helper and the
# proxy node base. Every proxy node shares the latter so the failure message stays in one place.
CREDENTIAL_CHECK_OWNERS = {
    LIBRARY_ROOT / "config" / "base_driver.py",
    LIBRARY_ROOT / "proxy" / "griptape_proxy_node.py",
}


def _files_defining_validate_api_key() -> set[Path]:
    matches = set()
    for path in sorted(LIBRARY_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == "_validate_api_key":
                matches.add(path)
    return matches


def test_only_shared_bases_define_the_credential_check() -> None:
    """A per-node copy of the check drifts from the shared credential message."""
    assert _files_defining_validate_api_key() == CREDENTIAL_CHECK_OWNERS


def _stub_secrets(monkeypatch: pytest.MonkeyPatch, secrets: dict[str, str | None]) -> None:
    """Make credential resolution read from an in-memory dict instead of the real secret sources.

    An absent secret is None and a secret registered with a blank value is that blank string,
    matching ``SecretsManager.get_secret``. Stubbed rather than driven through the environment
    because ``get_secret`` also reads the workspace and global ``.env`` files, so a developer
    running against Griptape Cloud would otherwise hand these tests a real credential.
    """
    monkeypatch.setattr(
        access_module.GriptapeNodes,
        "SecretsManager",
        lambda: type("S", (), {"get_secret": lambda self, name, **_kwargs: secrets.get(name)})(),
    )


@pytest.fixture(autouse=True)
def unconfigured_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Present an environment where no credential is configured."""
    monkeypatch.delenv(PROXY_API_KEY_ENV_VAR, raising=False)
    _stub_secrets(monkeypatch, {LICENSE_SECRET_NAME: None, API_KEY_NAME: None})


@pytest.mark.parametrize("node_class", [GoogleImageGeneration, OmnihumanSubjectDetection])
def test_missing_credential_error_names_every_accepted_credential(node_class: type) -> None:
    node = node_class(name=node_class.__name__)

    with pytest.raises(ValueError) as error:  # noqa: PT011 -- the message under test is the point
        node._validate_api_key()

    message = str(error.value)
    assert node.name in message
    # A License-only user must not be told to set an API key they are deliberately not using.
    assert "license" in message.lower()
    assert API_KEY_NAME in message
    # The proxy calls the provider server-side, so the credential is a Griptape Cloud one.
    assert "Griptape Cloud" in message


def test_missing_credential_error_reports_a_blank_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """A blank secret looks configured in the config file, Settings, and the log."""
    _stub_secrets(monkeypatch, {LICENSE_SECRET_NAME: None, API_KEY_NAME: ""})
    node = GoogleImageGeneration(name="Google Nano Banana Image Generation")

    with pytest.raises(ValueError) as error:  # noqa: PT011 -- the message under test is the point
        node._validate_api_key()

    assert f"{API_KEY_NAME} is set to a blank value" in str(error.value)


def test_proxy_headers_carry_the_resolved_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    """The exact dict a billable proxy call sends, composed the way its three call sites compose it.

    The missing-credential path is covered by
    `test_missing_credential_error_names_every_accepted_credential`, which exercises the
    same `_validate_api_key` these sites now call directly.
    """
    _stub_secrets(monkeypatch, {LICENSE_SECRET_NAME: None, API_KEY_NAME: "gt-cloud-key"})
    node = GoogleImageGeneration(name="Google Nano Banana Image Generation")

    headers = build_attribution_headers(node._validate_api_key())

    assert headers == {"Authorization": "Bearer gt-cloud-key", "Content-Type": "application/json"}
