"""Scripted pydantic-ai models for node tests. Use with `griptape_nodes_library.llm.models.override_model`."""

from collections.abc import AsyncIterator, Callable

from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel

Respond = Callable[[list[ModelMessage], AgentInfo], ModelResponse]


def fake_model(respond: Respond) -> FunctionModel:
    """A `FunctionModel` that answers streamed and non-streamed requests with `respond`."""

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

    return FunctionModel(respond, stream_function=stream)


def text_model(text: str) -> FunctionModel:
    """Always answer `text`."""
    return fake_model(lambda messages, info: ModelResponse(parts=[TextPart(text)]))
