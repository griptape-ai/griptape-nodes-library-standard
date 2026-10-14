"""Tests that ``DescribeImage.process`` declares the model invocation before
running the model (issue #431).

``process`` declares the invocation right before the network call, once the
model to run is settled, and fails closed (raises) when the declaration is denied.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
from griptape.artifacts import ImageArtifact
from griptape_nodes.exe_types.node_types import BaseNode
from griptape_nodes.node_library.library_registry import LibraryRegistry

import griptape_nodes_library.image.describe_image as describe_image_module
import griptape_nodes_library.utils.model_invocation as model_invocation_module
from griptape_nodes_library.image.describe_image import DescribeImage
from griptape_nodes_library.llm.agent_state import AgentState
from griptape_nodes_library.llm.model_config import ModelConfig, ModelProvider
from griptape_nodes_library.llm.models import override_model
from griptape_nodes_library.llm.testing import text_model

LIBRARY_NAME = "Griptape Nodes Library"


def _create_node(node_type: str) -> BaseNode:
    """Create a node through the library so its metadata carries `library` / `node_type`.

    `_get_selected_model_id` / `resolve_catalog_model_id` read those two metadata
    keys to resolve a node's declared models; a bare `NodeClass(name=...)`
    construction leaves every model-id resolution in `process()` returning `None`.
    """
    library = LibraryRegistry.get_library(name=LIBRARY_NAME)
    return library.create_node(node_type=node_type, name=node_type)


class _FakeDeclaration:
    """Stand-in for the `ResultPayload` returned by `declare_model_invocation_sync`."""

    def __init__(self, *, ok: bool, details: str = "") -> None:
        self._ok = ok
        self.result_details = details

    def failed(self) -> bool:
        return not self._ok


def _stub_secret(monkeypatch: pytest.MonkeyPatch, value: str | None) -> None:
    class _FakeSecrets:
        # Accepts **kwargs because the Cloud credential resolver passes
        # should_error_on_not_found= when probing for the License.
        def get_secret(self, _name: str, **_kwargs: object) -> str | None:
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
    node = cast(DescribeImage, _create_node("DescribeImage"))
    node.set_parameter_value("model", "gpt-5.2")
    _stub_images(node, monkeypatch, [ImageArtifact(b"\x89PNG", format="png", width=1, height=1)])
    return node


def test_declares_invocation_with_selected_model_before_running(
    describe_image_node: DescribeImage, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    def _fake_declare(node: DescribeImage, api_model_id: str) -> _FakeDeclaration:
        captured["node"] = node
        captured["api_model_id"] = api_model_id
        return _FakeDeclaration(ok=True)

    monkeypatch.setattr(model_invocation_module, "declare_model_invocation_sync", _fake_declare)

    with override_model(text_model("a cat")):
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

    monkeypatch.setattr(model_invocation_module, "declare_model_invocation_sync", _fake_declare)

    with override_model(text_model("a cat")):
        gen = describe_image_node.process()
        with pytest.raises(RuntimeError, match="denied by policy"):
            next(gen)


def test_declares_connected_agents_model_over_stale_dropdown_value(
    describe_image_node: DescribeImage, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A connected Agent supplies the model that actually runs.

    The node's own `model` dropdown keeps its last value when an upstream Agent
    is connected -- the parameter is hidden, not cleared -- so the declaration
    must read the model from the agent's model config, not from the
    stale parameter value.
    """
    upstream = AgentState(model=ModelConfig(provider=ModelProvider.GRIPTAPE_CLOUD, model="gpt-4.1"))
    describe_image_node.set_parameter_value("agent", upstream.to_wire())
    # The dropdown still holds its previous selection (set in the fixture).
    assert describe_image_node.get_parameter_value("model") == "gpt-5.2"

    captured: dict[str, Any] = {}

    def _fake_declare(node: DescribeImage, api_model_id: str) -> _FakeDeclaration:
        captured["api_model_id"] = api_model_id
        return _FakeDeclaration(ok=True)

    monkeypatch.setattr(model_invocation_module, "declare_model_invocation_sync", _fake_declare)

    with override_model(text_model("a cat")):
        gen = describe_image_node.process()
        next(gen)

    assert captured["api_model_id"] == "gpt-4.1"
