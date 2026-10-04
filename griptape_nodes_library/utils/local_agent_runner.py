"""Run an :class:`~griptape_nodes_library.utils.agent_state.AgentState` with pydantic-ai in this process."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from griptape.artifacts import ImageUrlArtifact
from griptape_nodes.drivers.cloud_models import OLLAMA_DEFAULT_BASE_URL, model_settings_for
from griptape_nodes.files.project_file import ProjectFileDestination
from openai import AsyncOpenAI
from pydantic_ai import Agent, StructuredDict
from pydantic_ai.messages import (
    AgentStreamEvent,
    BinaryContent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    PartDeltaEvent,
    PartStartEvent,
    RetryPromptPart,
    TextPart,
    TextPartDelta,
    UserContent,
    UserPromptPart,
)
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.run import AgentRunResultEvent
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import Tool

from griptape_nodes_library.utils.agent_runner import AgentRunEvent, AgentRunOutput, TextDelta, ToolCalled, ToolReturned
from griptape_nodes_library.utils.agent_state import AgentState, ProviderKind, ProviderRef
from griptape_nodes_library.utils.agent_tools import build_pydantic_tools
from griptape_nodes_library.utils.cloud_credential_utils import missing_credential_message, resolve_cloud_api_key
from griptape_nodes_library.utils.griptape_cloud_headers import build_griptape_cloud_headers
from griptape_nodes_library.utils.image_utils import load_image_from_url_artifact

GRIPTAPE_CLOUD_BASE_URL = "https://cloud.griptape.ai"


# ---------------------------------------------------------------------------
# Local pydantic-ai runner
# ---------------------------------------------------------------------------


@dataclass
class LocalAgentRunner:
    """Runs pydantic-ai in this process.

    Attributes:
        extra_tools: Tools that exist only for this run: live griptape tools connected to
            the node that have no config to rebuild them from. They are not added to the
            returned state.
        model_override: Model to call instead of the one the state names. For tests.
    """

    extra_tools: list[Tool] = field(default_factory=list)
    model_override: Model | None = None

    def run(
        self,
        state: AgentState,
        prompt: Sequence[UserContent],
        *,
        on_event: Callable[[AgentRunEvent], None] | None = None,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> AgentRunOutput:
        # Built before the event loop starts: resolving a Cloud credential and rebuilding
        # tools both call into the engine synchronously.
        model = self.model_override or build_model(state.provider, state.model, state.model_settings)
        output_type: Any = (
            StructuredDict(state.output_schema, name="output") if state.output_schema is not None else str
        )
        agent = Agent(
            model,
            output_type=output_type,
            instructions=state.instructions(),
            tools=[*build_pydantic_tools(state.tools), *self.extra_tools],
        )
        return asyncio.run(_run(agent, state, prompt, on_event=on_event, is_cancelled=is_cancelled))


async def _run(
    agent: Agent[None, Any],
    state: AgentState,
    prompt: Sequence[UserContent],
    *,
    on_event: Callable[[AgentRunEvent], None] | None,
    is_cancelled: Callable[[], bool] | None,
) -> AgentRunOutput:
    tool_names: dict[str, str] = {}
    async with agent.run_stream_events(list(prompt), message_history=_inline_image_urls(state.messages)) as events:
        async for event in events:
            if is_cancelled is not None and is_cancelled():
                return AgentRunOutput(output="", state=state, cancelled=True)
            if isinstance(event, AgentRunResultEvent):
                new_messages = _store_image_urls(event.result.new_messages(), prompt)
                result_state = state.model_copy(update={"messages": [*state.messages, *new_messages]})
                return AgentRunOutput(output=event.result.output, state=result_state)
            if on_event is not None:
                for run_event in _to_run_events(event, tool_names):
                    on_event(run_event)
    msg = "Agent run ended without a result."
    raise RuntimeError(msg)


def _to_run_events(event: AgentStreamEvent, tool_names: dict[str, str]) -> list[AgentRunEvent]:
    if isinstance(event, PartStartEvent) and isinstance(event.part, TextPart) and event.part.content:
        return [TextDelta(event.part.content)]
    if isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
        return [TextDelta(event.delta.content_delta)]
    if isinstance(event, FunctionToolCallEvent):
        tool_names[event.part.tool_call_id] = event.part.tool_name
        return [ToolCalled(event.part.tool_name, event.part.args_as_json_str())]
    if isinstance(event, FunctionToolResultEvent):
        part = event.part
        name = tool_names.get(part.tool_call_id, part.tool_name or "")
        if isinstance(part, RetryPromptPart):
            return [ToolReturned(name, part.model_response(), is_error=True)]
        return [ToolReturned(name, part.model_response_str())]
    return []


# ---------------------------------------------------------------------------
# Images in history
# ---------------------------------------------------------------------------


def _inline_image_urls(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Download every ``ImageUrl`` in the user prompts of ``messages`` into ``BinaryContent``.

    History keeps an image the user connected by URL as that URL, so the saved workflow
    doesn't carry the bytes. The URL is often a local static-file address only this machine
    can reach, so the bytes are what goes to the model.
    """
    result: list[ModelMessage] = []
    for message in messages:
        if not isinstance(message, ModelRequest) or not any(
            isinstance(part, UserPromptPart) and not isinstance(part.content, str) for part in message.parts
        ):
            result.append(message)
            continue
        parts = [
            UserPromptPart(content=[_inline(item) for item in part.content], timestamp=part.timestamp)
            if isinstance(part, UserPromptPart) and not isinstance(part.content, str)
            else part
            for part in message.parts
        ]
        result.append(ModelRequest(parts=parts, instructions=message.instructions))
    return result


def _inline(item: UserContent) -> UserContent:
    if isinstance(item, ImageUrl):
        artifact = load_image_from_url_artifact(ImageUrlArtifact(item.url))
        return BinaryContent(data=artifact.value, media_type=artifact.mime_type)
    return item


def _store_image_urls(new_messages: list[ModelMessage], prompt: Sequence[UserContent]) -> list[ModelMessage]:
    """Return ``new_messages`` with the run's prompt stored by reference.

    URLs stay as given. Image bytes are written to a project file and stored as its URL,
    so history, and the workflow it's saved in, never carries image data.
    """
    if not new_messages or not isinstance(new_messages[0], ModelRequest):
        return new_messages
    first = new_messages[0]
    stored = [_to_file_url(item) for item in prompt]
    parts = [
        UserPromptPart(content=stored, timestamp=part.timestamp) if isinstance(part, UserPromptPart) else part
        for part in first.parts
    ]
    return [ModelRequest(parts=parts, instructions=first.instructions), *new_messages[1:]]


def _to_file_url(item: UserContent) -> UserContent:
    if isinstance(item, BinaryContent) and item.is_image:
        dest = ProjectFileDestination.from_situation(filename=f"agent_input.{item.format}", situation="save_file")
        return ImageUrl(url=dest.write_bytes(item.data).location)
    return item


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def build_model(provider: ProviderRef, model_name: str, settings: dict[str, Any] | None = None) -> Model:
    """Build the pydantic-ai model a :class:`ProviderRef` points at, resolving its credential.

    Raises:
        KeyError: The provider is Griptape Cloud and no License or API key is set.
    """
    match provider.kind:
        case ProviderKind.GRIPTAPE_CLOUD:
            api_key = resolve_cloud_api_key()
            if not api_key:
                raise KeyError(missing_credential_message("run the Agent"))
            root = os.environ.get("GT_CLOUD_BASE_URL", GRIPTAPE_CLOUD_BASE_URL).rstrip("/")
            client = AsyncOpenAI(
                base_url=f"{root}/api/v1",
                api_key=api_key,
                default_headers=build_griptape_cloud_headers(api_key, attribution=True),
            )
            preset = model_settings_for(model_name) or {}
            return OpenAIChatModel(
                model_name,
                provider=OpenAIProvider(openai_client=client),
                settings=ModelSettings(**{**preset, **(settings or {})}),
            )
        case ProviderKind.OLLAMA:
            base_url = (provider.base_url or OLLAMA_DEFAULT_BASE_URL).rstrip("/")
            # Ollama needs no key, but the OpenAI client rejects an empty one.
            client = AsyncOpenAI(base_url=base_url, api_key="ollama")
            return OpenAIChatModel(
                model_name, provider=OpenAIProvider(openai_client=client), settings=ModelSettings(**(settings or {}))
            )
        case ProviderKind.OPENAI_COMPATIBLE:
            client = AsyncOpenAI(base_url=provider.base_url or None, api_key=provider.resolve_api_key())
            return OpenAIChatModel(
                model_name, provider=OpenAIProvider(openai_client=client), settings=ModelSettings(**(settings or {}))
            )
        case _:
            msg = f"Unknown provider kind: {provider.kind!r}"
            raise ValueError(msg)
