"""Tests for ``DescribeImage.process`` over the pydantic-ai adapter layer."""

from __future__ import annotations

import json
from typing import Any, cast

import pytest
from griptape.artifacts import ImageArtifact
from griptape_nodes.exe_types.node_types import BaseNode
from griptape_nodes.node_library.library_registry import LibraryRegistry
from griptape_nodes.retained_mode.events.agent_events import ProviderConfig
from pydantic_ai.messages import (
    BinaryContent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo

import griptape_nodes_library.image.describe_image as describe_image_module
import griptape_nodes_library.utils.model_invocation as model_invocation_module
from griptape_nodes_library.image.describe_image import DescribeImage
from griptape_nodes_library.llm.agent_state import AgentState
from griptape_nodes_library.llm.content import image_content
from griptape_nodes_library.llm.model_config import ModelConfig, ModelProvider
from griptape_nodes_library.llm.models import override_model
from griptape_nodes_library.llm.testing import fake_model, text_model

PNG_BYTES = b"\x89PNG\r\n\x1a\nfake"
IMAGE = ImageArtifact(PNG_BYTES, format="png", width=1, height=1)


class _AllowedDeclaration:
    result_details = ""

    def failed(self) -> bool:
        return False


class _FakeSecrets:
    def get_secret(self, _name: str, **_kwargs: object) -> str:
        return "gt-cloud-key"


@pytest.fixture
def node(monkeypatch: pytest.MonkeyPatch) -> DescribeImage:
    monkeypatch.setattr(describe_image_module.GriptapeNodes, "SecretsManager", lambda: _FakeSecrets())
    monkeypatch.setattr(model_invocation_module, "declare_model_invocation_sync", lambda *_a: _AllowedDeclaration())
    library = LibraryRegistry.get_library(name="Griptape Nodes Library")
    created = cast(DescribeImage, library.create_node(node_type="DescribeImage", name="DescribeImage"))
    monkeypatch.setattr(BaseNode, "get_parameter_value", _with_images([IMAGE]), raising=True)
    return created


def _with_images(images: list[Any]):
    original = BaseNode.get_parameter_value

    def _get(self: BaseNode, name: str) -> Any:
        if name == "images":
            return images
        return original(self, name)

    return _get


def _run(node: DescribeImage) -> None:
    gen = node.process()
    try:
        runner = next(gen)
        gen.send(runner())
    except StopIteration:
        pass


def test_describes_image_and_threads_text_only_history(node: DescribeImage) -> None:
    seen: list[list[ModelMessage]] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(messages)
        return ModelResponse(parts=[TextPart("a cat")])

    node.set_parameter_value("prompt", "What is it?")
    with override_model(fake_model(respond)):
        _run(node)

    request = seen[0][-1]
    assert isinstance(request, ModelRequest)
    prompt_part = next(p for p in request.parts if isinstance(p, UserPromptPart))
    assert isinstance(prompt_part.content, list)
    assert prompt_part.content[0] == "What is it?\n\nOutput image description only."
    assert prompt_part.content[1] == BinaryContent(data=PNG_BYTES, media_type="image/png")

    assert node.parameter_output_values["output"] == "a cat"
    state = AgentState.from_wire(node.parameter_output_values["agent"])
    assert state.runs() == [{"input": "What is it?\n\nOutput image description only.", "output": "a cat"}]
    assert PNG_BYTES.decode("latin-1") not in json.dumps(node.parameter_output_values["agent"])


def test_incoming_agent_history_is_sent_and_extended(node: DescribeImage) -> None:
    upstream = AgentState(
        model=ModelConfig(provider=ModelProvider.GRIPTAPE_CLOUD, model="gpt-4.1"),
        rulesets=[{"name": "Style", "rules": ["Be brief."]}],
    ).with_runs([{"input": "hello", "output": "hi"}])
    node.set_parameter_value("agent", upstream.to_wire())
    captured: dict[str, Any] = {}

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        captured["messages"] = messages
        return ModelResponse(parts=[TextPart("a dog")])

    with override_model(fake_model(respond)):
        _run(node)

    assert len(captured["messages"]) == 3
    state = AgentState.from_wire(node.parameter_output_values["agent"])
    assert [run["output"] for run in state.runs()] == ["hi", "a dog"]
    assert state.model == upstream.model
    assert state.rulesets == upstream.rulesets


def test_output_schema_returns_structured_output(node: DescribeImage) -> None:
    node.set_parameter_value(
        "output_schema",
        {"type": "object", "properties": {"subject": {"type": "string"}}, "required": ["subject"]},
    )

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        tool = info.output_tools[0]
        return ModelResponse(parts=[ToolCallPart(tool.name, {"subject": "cat"})])

    with override_model(fake_model(respond)):
        _run(node)

    assert node.parameter_output_values["output"] == {"subject": "cat"}
    state = AgentState.from_wire(node.parameter_output_values["agent"])
    assert json.loads(state.runs()[-1]["output"]) == {"subject": "cat"}


def test_no_images_short_circuits(node: DescribeImage, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(BaseNode, "get_parameter_value", _with_images([None, ""]), raising=True)

    with override_model(text_model("unused")):
        _run(node)

    assert node.parameter_output_values["output"] == "No image provided"


def test_non_object_output_schema_raises(node: DescribeImage) -> None:
    node.set_parameter_value("output_schema", ["not", "an", "object"])

    with pytest.raises(TypeError, match="must be a JSON schema object"):
        next(node.process())


class TestImageContent:
    def test_image_artifact(self) -> None:
        assert image_content(IMAGE) == BinaryContent(data=PNG_BYTES, media_type="image/png")

    def test_rejects_unknown_value(self) -> None:
        with pytest.raises(TypeError):
            image_content(object())


def test_third_party_provider_becomes_model_config(node: DescribeImage, monkeypatch: pytest.MonkeyPatch) -> None:
    provider = ProviderConfig(
        name="local", type="custom", model="", base_url="http://localhost:9/v1", api_key_secret_name="LOCAL_KEY"
    )
    monkeypatch.setattr(node._provider, "_fetch_providers", lambda: [provider])
    node.parameter_values["model_provider"] = "local"
    node.parameter_values["model"] = "llava"

    with override_model(text_model("ok")):
        _run(node)

    state = AgentState.from_wire(node.parameter_output_values["agent"])
    assert state.model == ModelConfig(
        provider=ModelProvider.OPENAI_COMPATIBLE,
        model="llava",
        base_url="http://localhost:9/v1",
        api_key_secret="LOCAL_KEY",
        settings={"temperature": 0.1},
    )
