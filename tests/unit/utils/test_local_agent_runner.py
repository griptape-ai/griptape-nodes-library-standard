"""Tests for ``LocalAgentRunner`` and the griptape tool adapter, run against pydantic-ai test models."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from types import SimpleNamespace

import pytest
from griptape.artifacts import ImageArtifact, ImageUrlArtifact
from griptape.tools import CalculatorTool
from pydantic_ai.messages import (
    BinaryContent,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel

import griptape_nodes_library.utils.local_agent_runner as local_runner_module
from griptape_nodes_library.utils.agent_runner import TextDelta, ToolCalled, ToolReturned, image_prompt_content
from griptape_nodes_library.utils.agent_state import AgentState
from griptape_nodes_library.utils.agent_tools import griptape_tool_to_pydantic
from griptape_nodes_library.utils.local_agent_runner import LocalAgentRunner

PNG = b"\x89PNG\r\n\x1a\nfake"


def _function_model(respond: Callable[[list[ModelMessage], AgentInfo], ModelResponse]) -> FunctionModel:
    """Stream whatever ``respond`` returns, since the runner always requests a stream."""

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        for index, part in enumerate(respond(messages, info).parts):
            if isinstance(part, TextPart):
                yield part.content
            elif isinstance(part, ToolCallPart):
                yield {
                    index: DeltaToolCall(
                        name=part.tool_name, json_args=part.args_as_json_str(), tool_call_id=part.tool_call_id
                    )
                }

    return FunctionModel(stream_function=stream)


def _echo_model(seen: list[list[ModelMessage]]) -> FunctionModel:
    def respond(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        seen.append(messages)
        return ModelResponse(parts=[TextPart(content="done")])

    return _function_model(respond)


def test_run_appends_turn_and_streams_text() -> None:
    events = []
    state = AgentState(model="m", rulesets=[{"name": "tone", "rules": ["Be brief"]}])

    run = LocalAgentRunner(model_override=TestModel(custom_output_text="Hello there")).run(
        state, ["Hi"], on_event=events.append
    )

    assert run.output == "Hello there"
    assert not run.cancelled
    assert [(t.prompt, t.response) for t in run.state.turns()] == [("Hi", "Hello there")]
    assert state.messages == []  # the input state is left as it was
    assert "".join(e.text for e in events if isinstance(e, TextDelta)) == "Hello there"


def test_rulesets_become_instructions() -> None:
    seen: list[list[ModelMessage]] = []
    state = AgentState(model="m", rulesets=[{"name": "tone", "rules": ["Be brief"]}])

    LocalAgentRunner(model_override=_echo_model(seen)).run(state, ["Hi"])

    request = seen[0][-1]
    assert isinstance(request, ModelRequest)
    assert request.instructions is not None
    assert "Be brief" in request.instructions


def test_output_schema_returns_structured_output() -> None:
    schema = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
    state = AgentState(model="m", output_schema=schema)

    run = LocalAgentRunner(model_override=TestModel(custom_output_args={"name": "Ada"})).run(state, ["Who?"])

    assert run.output == {"name": "Ada"}


def test_cancel_returns_the_state_unchanged() -> None:
    state = AgentState(model="m")

    run = LocalAgentRunner(model_override=TestModel()).run(state, ["Hi"], is_cancelled=lambda: True)

    assert run.cancelled
    assert run.state is state


def test_history_image_urls_are_inlined_for_the_model_and_kept_as_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        local_runner_module,
        "load_image_from_url_artifact",
        lambda artifact: ImageArtifact(PNG, format="png", width=1, height=1),
    )
    url = ImageUrl(url="http://localhost/a.png")
    state = AgentState(
        model="m",
        messages=[
            ModelRequest(parts=[UserPromptPart(content=["What is this?", url])]),
            ModelResponse(parts=[TextPart(content="A cat.")]),
        ],
    )
    seen: list[list[ModelMessage]] = []

    run = LocalAgentRunner(model_override=_echo_model(seen)).run(state, ["And this?", url])

    sent_history, sent_prompt = seen[0][0], seen[0][-1]
    assert isinstance(sent_history.parts[0], UserPromptPart)
    assert isinstance(sent_history.parts[0].content[1], BinaryContent)  # pyright: ignore[reportIndexIssue]
    assert isinstance(sent_prompt, ModelRequest)
    stored_prompt = run.state.turns()[-1].user_content
    assert list(stored_prompt) == ["And this?", url]


def test_prompt_image_bytes_are_stored_as_a_file_url(monkeypatch: pytest.MonkeyPatch) -> None:
    written: list[tuple[str, bytes]] = []

    class _Dest:
        def __init__(self, filename: str) -> None:
            self.filename = filename

        def write_bytes(self, data: bytes) -> SimpleNamespace:
            written.append((self.filename, data))
            return SimpleNamespace(location=f"http://localhost/static/{self.filename}")

    monkeypatch.setattr(
        local_runner_module.ProjectFileDestination,
        "from_situation",
        lambda filename, situation: _Dest(filename),
    )
    image = BinaryContent(data=PNG, media_type="image/png")
    seen: list[list[ModelMessage]] = []

    run = LocalAgentRunner(model_override=_echo_model(seen)).run(AgentState(model="m"), ["What is this?", image])

    sent_prompt = seen[0][-1]
    assert isinstance(sent_prompt, ModelRequest)
    assert isinstance(sent_prompt.parts[0], UserPromptPart)
    assert sent_prompt.parts[0].content[1] == image  # pyright: ignore[reportIndexIssue]
    assert written == [("agent_input.png", PNG)]
    stored_prompt = run.state.turns()[-1].user_content
    assert list(stored_prompt) == ["What is this?", ImageUrl(url="http://localhost/static/agent_input.png")]
    assert "base64" not in json.dumps(run.state.to_wire()) and PNG.hex() not in json.dumps(run.state.to_wire())


def test_tool_calls_run_griptape_activities_and_report_events() -> None:
    events = []
    calls = {"count": 0}

    def respond(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        calls["count"] += 1
        if calls["count"] == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(tool_name="CalculatorTool_calculate", args={"expression": "6*7"}, tool_call_id="c1")
                ]
            )
        tool_return = messages[-1].parts[0]
        assert isinstance(tool_return, ToolReturnPart)
        return ModelResponse(parts=[TextPart(content=f"It is {tool_return.content}")])

    runner = LocalAgentRunner(
        extra_tools=griptape_tool_to_pydantic(CalculatorTool()), model_override=_function_model(respond)
    )
    run = runner.run(AgentState(model="m"), ["What is 6*7?"], on_event=events.append)

    assert run.output == "It is 42"
    assert ToolCalled("CalculatorTool_calculate", '{"expression":"6*7"}') in events
    assert ToolReturned("CalculatorTool_calculate", "42") in events


def test_tool_configs_on_the_state_are_rebuilt() -> None:
    state = AgentState(model="m", tools=[{"tool_type": "Calculator"}])
    seen_tools: list[str] = []

    def respond(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen_tools.extend(tool.name for tool in info.function_tools)
        return ModelResponse(parts=[TextPart(content="ok")])

    LocalAgentRunner(model_override=_function_model(respond)).run(state, ["Hi"])

    assert "CalculatorTool_calculate" in seen_tools


def test_tool_adapter_rejects_non_tools() -> None:
    with pytest.raises(TypeError):
        griptape_tool_to_pydantic(object())


def test_image_prompt_content() -> None:
    assert image_prompt_content(ImageUrlArtifact("http://h/a.png")) == ImageUrl(url="http://h/a.png")
    assert image_prompt_content(" http://h/b.png ") == ImageUrl(url="http://h/b.png")
    binary = image_prompt_content(ImageArtifact(PNG, format="png", width=1, height=1))
    assert isinstance(binary, BinaryContent)
    assert binary.data == PNG
    assert image_prompt_content("") is None
    assert image_prompt_content(None) is None
