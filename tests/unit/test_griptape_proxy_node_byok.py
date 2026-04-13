from __future__ import annotations

import ast
import importlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from griptape_nodes.exe_types.param_components.artifact_url.public_artifact_url_parameter import (
    PublicArtifactUrlParameter,
)

from griptape_nodes_library.image.flux_2_image_generation import Flux2ImageGeneration
from griptape_nodes_library.proxy import (
    get_proxy_api_key_provider_config,
    is_proxy_api_key_provider_disabled,
)


def _iter_proxy_node_classes() -> Iterator[tuple[str, str]]:
    library_root = Path(__file__).parents[2] / "griptape_nodes_library"

    for path in sorted(library_root.rglob("*.py")):
        module_rel_path = path.relative_to(library_root.parent)
        module_name = ".".join(module_rel_path.with_suffix("").parts)
        tree = ast.parse(path.read_text())

        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue

            if not any(isinstance(base, ast.Name) and base.id == "GriptapeProxyNode" for base in node.bases):
                continue

            yield node.name, module_name


PROXY_NODE_CLASSES = tuple(_iter_proxy_node_classes())


@pytest.fixture(autouse=True)
def stub_public_artifact_bucket_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        PublicArtifactUrlParameter, "_get_bucket_id", staticmethod(lambda *_args, **_kwargs: "test-bucket")
    )


@pytest.mark.parametrize(("class_name", "_module_name"), PROXY_NODE_CLASSES)
def test_every_proxy_node_explicitly_declares_byok_support_state(class_name: str, _module_name: str) -> None:
    provider_config = get_proxy_api_key_provider_config(class_name)
    is_disabled = is_proxy_api_key_provider_disabled(class_name)
    assert (provider_config is not None) != is_disabled


@pytest.mark.parametrize(("class_name", "module_name"), PROXY_NODE_CLASSES)
def test_every_proxy_node_exposes_shared_byok_ui(class_name: str, module_name: str) -> None:
    provider_config = get_proxy_api_key_provider_config(class_name)
    module = importlib.import_module(module_name)
    node_class = getattr(module, class_name)
    node = node_class(name=class_name)

    api_key_provider_parameter = next(
        (parameter for parameter in node.parameters if parameter.name == "api_key_provider"), None
    )
    api_key_provider_message = node.get_message_by_name_or_element_id("api_key_provider_message")

    if is_proxy_api_key_provider_disabled(class_name):
        assert provider_config is None
        assert api_key_provider_parameter is None
        assert api_key_provider_message is None
        return

    assert provider_config is not None
    assert api_key_provider_parameter is not None
    assert api_key_provider_message is not None
    assert api_key_provider_message.button_link == f"#settings-secrets?filter={provider_config.api_key_name}"


@pytest.mark.asyncio
async def test_flux2_submission_keeps_proxy_bearer_auth_with_byok(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_request: dict[str, Any] = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, str]:
            return {"generation_id": "gen_123"}

    class FakeAsyncClient:
        async def __aenter__(self) -> FakeAsyncClient:
            return self

        async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            return None

        async def post(self, url: str, json: dict[str, Any], headers: dict[str, str], timeout: int) -> FakeResponse:
            captured_request["url"] = url
            captured_request["json"] = json
            captured_request["headers"] = headers
            captured_request["timeout"] = timeout
            return FakeResponse()

    monkeypatch.setattr("griptape_nodes_library.proxy.griptape_proxy_node.httpx.AsyncClient", FakeAsyncClient)

    node = Flux2ImageGeneration(name="Flux2")
    node.register_user_auth_info("user-bfl-key")

    generation_id = await node._submit_generation(
        payload={"prompt": "test"},
        headers={"Authorization": "Bearer gt-cloud-key", "Content-Type": "application/json"},
        api_model_id="flux-2-pro",
    )

    assert generation_id == "gen_123"
    assert captured_request["headers"]["Authorization"] == "Bearer gt-cloud-key"
    assert captured_request["headers"]["X-GTC-PROXY-AUTH-INFO"] == "user-bfl-key"
