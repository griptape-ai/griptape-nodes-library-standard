"""Tests that ``GenerateImage.process`` declares the model invocation before
running the model (issue #431).

``GenerateImage`` runs models through a griptape framework image generation
driver without ever declaring the call to the engine's permission layer.
``process`` now declares the invocation right before the network call, once
the driver's model is settled, and fails closed (raises) when the declaration
is denied.
"""

from __future__ import annotations

from typing import Any

import pytest

import griptape_nodes_library.image.create_image as create_image_module
from griptape_nodes_library.image.create_image import GenerateImage


class _FakeDeclaration:
    """Stand-in for the `ResultPayload` returned by `declare_model_invocation_sync`."""

    def __init__(self, *, ok: bool, details: str = "") -> None:
        self._ok = ok
        self.result_details = details

    def failed(self) -> bool:
        return not self._ok


def _stub_secret(monkeypatch: pytest.MonkeyPatch, value: str | None) -> None:
    class _FakeSecrets:
        def get_secret(self, _name: str) -> str | None:
            return value

    monkeypatch.setattr(create_image_module.GriptapeNodes, "SecretsManager", lambda: _FakeSecrets())


@pytest.fixture
def generate_image_node(monkeypatch: pytest.MonkeyPatch) -> GenerateImage:
    _stub_secret(monkeypatch, "gt-cloud-key")
    node = GenerateImage(name="GenerateImage")
    node.set_parameter_value("model", "gpt-image-1-mini")
    node.set_parameter_value("prompt", "a cat wearing a hat")
    # Leave enhance_prompt off (default) so process() reaches the image generation
    # driver declaration on the first yield without an extra prompt-model call.
    return node


def test_declares_invocation_with_selected_model_before_running(
    generate_image_node: GenerateImage, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    def _fake_declare(node: GenerateImage, api_model_id: str) -> _FakeDeclaration:
        captured["node"] = node
        captured["api_model_id"] = api_model_id
        return _FakeDeclaration(ok=True)

    monkeypatch.setattr(create_image_module, "declare_model_invocation_sync", _fake_declare)

    gen = generate_image_node.process()
    runner = next(gen)

    assert captured["api_model_id"] == "gpt-image-1-mini"
    assert captured["node"] is generate_image_node
    assert callable(runner)


def test_raises_before_running_when_declaration_is_denied(
    generate_image_node: GenerateImage, monkeypatch: pytest.MonkeyPatch
) -> None:
    ran = {"called": False}

    def _fake_create_image(self: GenerateImage, agent: Any, prompt: Any) -> None:  # pragma: no cover - must not run
        ran["called"] = True

    def _fake_declare(_node: GenerateImage, _api_model_id: str) -> _FakeDeclaration:
        return _FakeDeclaration(ok=False, details="denied by policy")

    monkeypatch.setattr(create_image_module, "declare_model_invocation_sync", _fake_declare)
    monkeypatch.setattr(GenerateImage, "_create_image", _fake_create_image)

    gen = generate_image_node.process()
    with pytest.raises(RuntimeError, match="denied by policy"):
        next(gen)

    assert ran["called"] is False
