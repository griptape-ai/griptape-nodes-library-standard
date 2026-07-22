"""Tests that ``DescribeImage.process`` declares the model invocation before
running the model (issue #431).

``DescribeImage`` runs models through a griptape framework prompt driver
without ever declaring the call to the engine's permission layer. ``process``
now declares the invocation right before the network call, once the driver's
model is settled, and fails closed (raises) when the declaration is denied.
"""

from __future__ import annotations

from typing import Any

import pytest

import griptape_nodes_library.image.describe_image as describe_image_module
from griptape_nodes_library.image.describe_image import DescribeImage


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

    monkeypatch.setattr(describe_image_module.GriptapeNodes, "SecretsManager", lambda: _FakeSecrets())


def _stub_images(node: DescribeImage, monkeypatch: pytest.MonkeyPatch, images: list[Any]) -> None:
    """Return a fixed image list for the ``images`` parameter without touching the real ParameterList."""
    original = node.get_parameter_value

    def _get(name: str) -> Any:
        if name == "images":
            return images
        return original(name)

    monkeypatch.setattr(node, "get_parameter_value", _get)


@pytest.fixture
def describe_image_node(monkeypatch: pytest.MonkeyPatch) -> DescribeImage:
    _stub_secret(monkeypatch, "gt-cloud-key")
    node = DescribeImage(name="DescribeImage")
    node.set_parameter_value("model", "gpt-5.2")
    _stub_images(node, monkeypatch, [object()])
    return node


def test_declares_invocation_with_selected_model_before_running(
    describe_image_node: DescribeImage, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    def _fake_declare(node: DescribeImage, api_model_id: str) -> _FakeDeclaration:
        captured["node"] = node
        captured["api_model_id"] = api_model_id
        return _FakeDeclaration(ok=True)

    monkeypatch.setattr(describe_image_module, "declare_model_invocation_sync", _fake_declare)

    gen = describe_image_node.process()
    runner = next(gen)

    assert captured["api_model_id"] == "gpt-5.2"
    assert captured["node"] is describe_image_node
    assert callable(runner)


def test_raises_before_running_when_declaration_is_denied(
    describe_image_node: DescribeImage, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _fake_declare(_node: DescribeImage, _api_model_id: str) -> _FakeDeclaration:
        return _FakeDeclaration(ok=False, details="denied by policy")

    monkeypatch.setattr(describe_image_module, "declare_model_invocation_sync", _fake_declare)

    gen = describe_image_node.process()
    with pytest.raises(RuntimeError, match="denied by policy"):
        next(gen)


def test_declares_connected_agents_model_over_stale_dropdown_value(
    describe_image_node: DescribeImage, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A connected Agent supplies the model that actually runs.

    The node's own `model` dropdown keeps its last value when an upstream Agent
    is connected -- the parameter is hidden, not cleared -- so the declaration
    must read the model from the restored agent's task driver, not from the
    stale parameter value.
    """
    from griptape.drivers.prompt.griptape_cloud import GriptapeCloudPromptDriver
    from griptape.structures import Agent as GtStructureAgent

    from griptape_nodes_library.utils.agent_utils import wrap_agent

    # Restoring the wrapper rebuilds a GriptapeCloudPromptDriver, whose api_key
    # default reads this env var.
    monkeypatch.setenv("GT_CLOUD_API_KEY", "fake-key")
    upstream = GtStructureAgent(prompt_driver=GriptapeCloudPromptDriver(model="gpt-4.1", api_key="fake-key"))
    describe_image_node.set_parameter_value("agent", wrap_agent(upstream.to_dict(), [], []))
    # The dropdown still holds its previous selection (set in the fixture).
    assert describe_image_node.get_parameter_value("model") == "gpt-5.2"

    captured: dict[str, Any] = {}

    def _fake_declare(node: DescribeImage, api_model_id: str) -> _FakeDeclaration:
        captured["api_model_id"] = api_model_id
        return _FakeDeclaration(ok=True)

    monkeypatch.setattr(describe_image_module, "declare_model_invocation_sync", _fake_declare)

    gen = describe_image_node.process()
    next(gen)

    assert captured["api_model_id"] == "gpt-4.1"
