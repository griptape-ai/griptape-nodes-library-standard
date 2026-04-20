from __future__ import annotations

import ast
import importlib
import logging
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


def test_elide_base64_in_payload() -> None:
    """Test that _elide_base64_in_payload elides base64 data URIs and truncates long strings."""
    node = Flux2ImageGeneration(name="Flux2")

    # Test base64 data URI elision
    payload_with_data_uri = {
        "prompt": "test",
        "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
    }
    elided = node._elide_base64_in_payload(payload_with_data_uri)
    assert "data:image/png;base64,[" in elided
    assert "chars]" in elided
    assert "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==" not in elided

    # Test long string truncation (>100 chars)
    long_string = "a" * 150
    payload_with_long_string = {
        "prompt": "test",
        "long_field": long_string,
    }
    elided = node._elide_base64_in_payload(payload_with_long_string)
    assert "a" * 100 in elided
    assert "[150 chars total]" in elided
    assert long_string not in elided

    # Test short strings are preserved
    payload_with_short_string = {
        "prompt": "test",
        "short_field": "short",
    }
    elided = node._elide_base64_in_payload(payload_with_short_string)
    assert '"short"' in elided

    # Test nested structures
    payload_nested = {
        "prompt": "test",
        "nested": {
            "image": "data:image/jpeg;base64," + "b" * 200,
            "long_value": "c" * 120,
        },
        "list": ["data:image/png;base64," + "d" * 150, "e" * 110],
    }
    elided = node._elide_base64_in_payload(payload_nested)
    assert "data:image/jpeg;base64,[200 chars]" in elided
    assert "c" * 100 in elided
    assert "[120 chars total]" in elided
    assert "data:image/png;base64,[150 chars]" in elided
    assert "e" * 100 in elided
    assert "[110 chars total]" in elided
    # Ensure raw data is not in logs
    assert "b" * 200 not in elided
    assert "c" * 120 not in elided
    assert "d" * 150 not in elided
    assert "e" * 110 not in elided


@pytest.mark.asyncio
async def test_submit_generation_logs_sanitized_payload(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    captured_request: dict[str, Any] = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, str]:
            return {"generation_id": "gen_456"}

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
    payload = {
        "prompt": "test",
        "image": "data:image/png;base64,RAW_IMAGE_BASE64_PAYLOAD",
        "nested": {"bytesBase64Encoded": "RAW_BYTES_BASE64_PAYLOAD"},
    }

    with caplog.at_level(logging.INFO, logger="griptape_nodes"):
        generation_id = await node._submit_generation(
            payload=payload,
            headers={"Authorization": "Bearer gt-cloud-key", "Content-Type": "application/json"},
            api_model_id="flux-2-pro",
        )

    assert generation_id == "gen_456"
    assert captured_request["json"] == payload
    assert "Request payload:" in caplog.text
    assert "RAW_IMAGE_BASE64_PAYLOAD" not in caplog.text
    assert "RAW_BYTES_BASE64_PAYLOAD" not in caplog.text
