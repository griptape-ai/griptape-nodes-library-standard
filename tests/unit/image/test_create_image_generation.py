"""Tests for the `GenerateImage` pipeline: prompt enhancement, image model selection, agent threading."""

from __future__ import annotations

from collections.abc import Generator
from typing import Any, cast

import pytest
from griptape_nodes.traits.options import Options
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo

import griptape_nodes_library.image.create_image as create_image_module
import griptape_nodes_library.utils.model_invocation as model_invocation_module
from griptape_nodes_library.image.create_image import ENHANCEMENT_MODEL, GenerateImage
from griptape_nodes_library.llm.agent_state import AgentState
from griptape_nodes_library.llm.image_generation import ImageGenerationConfig, ImageProvider
from griptape_nodes_library.llm.model_config import ModelConfig, ModelProvider, cloud_model_config
from griptape_nodes_library.llm.models import override_model
from griptape_nodes_library.llm.testing import fake_model


class _Allowed:
    result_details = ""

    def failed(self) -> bool:
        return False


class _FakeFile:
    def __init__(self, written: list[bytes]) -> None:
        self._written = written

    def write_bytes(self, data: bytes) -> Any:
        self._written.append(data)
        return type("Saved", (), {"location": "file:///generated.png"})()


@pytest.fixture
def declared(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    models: list[str] = []

    def _declare(_node: Any, api_model_id: str) -> _Allowed:
        models.append(api_model_id)
        return _Allowed()

    monkeypatch.setattr(model_invocation_module, "declare_model_invocation_sync", _declare)
    return models


@pytest.fixture
def generated(monkeypatch: pytest.MonkeyPatch) -> list[tuple[ImageGenerationConfig, str]]:
    calls: list[tuple[ImageGenerationConfig, str]] = []

    def _generate(config: ImageGenerationConfig, prompt: str) -> bytes:
        calls.append((config, prompt))
        return b"image-bytes"

    monkeypatch.setattr(create_image_module, "generate_image", _generate)
    return calls


@pytest.fixture
def written(monkeypatch: pytest.MonkeyPatch) -> list[bytes]:
    data: list[bytes] = []
    monkeypatch.setattr(GenerateImage, "publish_update_to_parameter", lambda *_a, **_k: None)
    monkeypatch.setattr(
        create_image_module.ProjectFileParameter, "build_file", lambda _self: _FakeFile(data), raising=False
    )
    return data


@pytest.fixture
def node() -> GenerateImage:
    created = GenerateImage(name="GenerateImage")
    created.set_parameter_value("model", "gpt-image-1-mini")
    created.set_parameter_value("prompt", "a cat")
    return created


def _drive(gen: Generator[Any, Any, None]) -> None:
    """Run a node's `process()` generator the way the engine does."""
    result = None
    try:
        while True:
            runner = gen.send(result)
            result = runner()
    except StopIteration:
        return


def test_generates_image_with_default_cloud_model_and_returns_agent_with_false_memory(
    node: GenerateImage, declared: list[str], generated: list[tuple[ImageGenerationConfig, str]], written: list[bytes]
) -> None:
    node.set_parameter_value("image_size", "1536x1024")

    _drive(node.process())

    config, prompt = generated[0]
    assert config == ImageGenerationConfig(
        provider=ImageProvider.GRIPTAPE_CLOUD, model="gpt-image-1-mini", image_size="1536x1024"
    )
    assert prompt == "\nUser:\na cat\n"
    assert declared == ["gpt-image-1-mini"]
    assert written == [b"image-bytes"]

    state = AgentState.from_wire(node.parameter_output_values["agent"])
    assert state.model == cloud_model_config(ENHANCEMENT_MODEL)
    runs = state.runs()
    assert runs[0]["input"] == "a cat"
    assert runs[0]["output"].startswith("I created an image based on your prompt.")


def test_connected_image_config_is_used_as_is(
    node: GenerateImage, declared: list[str], generated: list[tuple[ImageGenerationConfig, str]], written: list[bytes]
) -> None:
    connected = ImageGenerationConfig(provider=ImageProvider.OPENAI, model="gpt-image-1", quality="low")
    # A connection retypes the parameter to the source's output type and drops its dropdown.
    model_param = node.get_parameter_by_name("model")
    assert model_param is not None
    model_param.type = "Image Generation Driver"
    model_param.remove_trait(trait_type=model_param.find_elements_by_type(Options)[0])
    node.set_parameter_value("model", connected)

    _drive(node.process())

    assert generated[0][0] == connected
    assert declared == ["gpt-image-1"]
    assert written == [b"image-bytes"]


def test_agent_history_prefixes_prompt_and_is_preserved(
    node: GenerateImage, declared: list[str], generated: list[tuple[ImageGenerationConfig, str]], written: list[bytes]
) -> None:
    agent = AgentState(
        model=ModelConfig(provider=ModelProvider.OPENAI, model="gpt-5"),
        messages=[
            ModelRequest(parts=[UserPromptPart(content="hello")]),
            ModelResponse(parts=[TextPart(content="hi there")]),
        ],
        rulesets=[{"name": "r", "rules": ["be nice"]}],
    )
    node.set_parameter_value("agent", agent.to_wire())

    _drive(node.process())

    assert generated[0][1] == (
        "<Conversation History>\nUser: hello\nAssistant: hi there</Conversation History>\n\nUser:\na cat\n"
    )
    out = AgentState.from_wire(node.parameter_output_values["agent"])
    assert out.model == agent.model
    assert out.rulesets == agent.rulesets
    assert [run["input"] for run in out.runs()] == ["hello", "a cat"]
    assert written == [b"image-bytes"]


def test_enhance_prompt_runs_agent_model_and_feeds_result_to_image_model(
    node: GenerateImage, declared: list[str], generated: list[tuple[ImageGenerationConfig, str]], written: list[bytes]
) -> None:
    node.set_parameter_value("enhance_prompt", True)
    seen: list[list[ModelMessage]] = []

    def _respond(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        seen.append(messages)
        return ModelResponse(parts=[TextPart(content="a fluffy cat at golden hour")])

    with override_model(fake_model(_respond)):
        _drive(node.process())

    assert declared == [ENHANCEMENT_MODEL, "gpt-image-1-mini"]
    assert generated[0][1] == "a fluffy cat at golden hour"
    request = cast(ModelRequest, seen[0][0])
    user_text = "\n".join(
        item for p in request.parts if isinstance(p, UserPromptPart) for item in cast(list[str], p.content)
    )
    assert "Enhance the following prompt for an image generation engine" in user_text
    assert "User:\na cat" in user_text
    assert written == [b"image-bytes"]
    # The enhancement exchange is not left in the agent's memory.
    out = AgentState.from_wire(node.parameter_output_values["agent"])
    assert [run["input"] for run in out.runs()] == ["a cat"]
