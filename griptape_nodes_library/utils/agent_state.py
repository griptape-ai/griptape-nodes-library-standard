"""The Agent value passed between nodes.

An :class:`AgentState` is plain data: which model to call and how to reach it, the rulesets
and tool configs the agent carries, an optional output schema, and the conversation so far
as pydantic-ai messages. Nothing in it is a live object, so every node that receives one
rebuilds the model and tools fresh, and the value survives saving the workflow.

Credentials never travel. :class:`ProviderRef` names the secret (or the provider config) a
credential comes from, and the runner resolves it at the point of use.

Saved workflows hold the older griptape wrapper (``{"agent": Agent.to_dict(), "tools": [...],
"rulesets": [...], "provider": {...}}``) or a bare ``Agent.to_dict()``.
:meth:`AgentState.from_wire` reads both, so those workflows keep working unchanged.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

from griptape_nodes.drivers.cloud_models import ProviderID
from griptape_nodes.retained_mode.events.agent_events import ListAgentProvidersRequest, ListAgentProvidersResultSuccess
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from pydantic import BaseModel, Field
from pydantic_ai.messages import (
    BinaryContent,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserContent,
    UserPromptPart,
)

AGENT_STATE_FORMAT = "griptape-nodes-agent/v1"
"""Tag every :class:`AgentState` wire dict carries, so readers can tell it from a legacy wrapper."""


class ProviderKind(StrEnum):
    """How the runner reaches the model."""

    GRIPTAPE_CLOUD = "griptape_cloud"
    OPENAI_COMPATIBLE = "openai_compatible"
    OLLAMA = "ollama"


class ProviderRef(BaseModel):
    """Where a model is served and where its credential comes from.

    Attributes:
        kind: Which client the runner builds.
        name: The provider config name selected in the node's provider dropdown. For a
            non-Cloud provider with no ``api_key_secret``, the runner looks the config up
            by this name to find its secret.
        base_url: Endpoint root for OpenAI-compatible and Ollama providers.
        api_key_secret: Name of the secret holding the API key, resolved through the
            engine's secrets manager when the agent runs.
    """

    kind: ProviderKind = ProviderKind.GRIPTAPE_CLOUD
    name: str = ProviderID.GRIPTAPE_CLOUD
    base_url: str = ""
    api_key_secret: str | None = None

    def resolve_api_key(self) -> str:
        """Return the API key for an OpenAI-compatible provider, or ``"not-needed"`` if it has none.

        The OpenAI client rejects an empty key, and local endpoints accept any value.
        """
        secret_name = self.api_key_secret or self._provider_config_secret()
        if not secret_name:
            return "not-needed"
        secrets = GriptapeNodes.SecretsManager()
        return secrets.get_secret(secret_name, should_error_on_not_found=False) or "not-needed"

    def _provider_config_secret(self) -> str | None:
        result = GriptapeNodes.handle_request(ListAgentProvidersRequest())
        if not isinstance(result, ListAgentProvidersResultSuccess):
            return None
        config = next((p for p in result.providers or [] if p.name == self.name), None)
        return config.api_key_secret_name if config else None


class AgentState(BaseModel):
    """Everything a node needs to run or edit an agent it received."""

    format: Literal["griptape-nodes-agent/v1"] = AGENT_STATE_FORMAT
    provider: ProviderRef = Field(default_factory=ProviderRef)
    model: str
    model_settings: dict[str, Any] = Field(default_factory=dict)
    rulesets: list[dict[str, Any]] = Field(default_factory=list)
    tools: list[dict[str, Any]] = Field(default_factory=list)
    output_schema: dict[str, Any] | None = None
    messages: list[ModelMessage] = Field(default_factory=list)

    @classmethod
    def from_wire(cls, value: object) -> AgentState | None:
        """Read an Agent parameter value in any format a node may receive.

        Returns ``None`` for anything that isn't an agent (``None``, a string, an empty dict).
        """
        if not isinstance(value, dict) or not value:
            return None
        if value.get("format") == AGENT_STATE_FORMAT:
            return cls.model_validate(value)
        return _from_legacy_wrapper(value)

    def to_wire(self) -> dict[str, Any]:
        """Return the JSON-safe dict stored as the Agent parameter value."""
        return self.model_dump(mode="json")

    def instructions(self) -> str | None:
        """Render the rulesets as the system instructions sent with every request."""
        sections = []
        for ruleset in self.rulesets:
            rules = [str(rule) for rule in ruleset.get("rules", []) if str(rule).strip()]
            if not rules:
                continue
            lines = "\n".join(f"- {rule}" for rule in rules)
            sections.append(f'Ruleset "{ruleset.get("name", "rules")}":\n{lines}')
        if not sections:
            return None
        return "Follow every rule below.\n\n" + "\n\n".join(sections)

    def turns(self) -> list[ConversationTurn]:
        """Split the history into user turns, in order. See :class:`ConversationTurn`."""
        return _split_turns(self.messages)

    def replace_turn(self, index: int, *, prompt: str | None = None, response: str | None = None) -> None:
        """Replace one turn with a single prompt and a single text response.

        ``None`` keeps that side of the turn as it was. Tool calls made during the turn are
        dropped, since the edited response no longer follows from them.

        Raises:
            IndexError: ``index`` is not a turn in the history.
        """
        turns = self.turns()
        turn = turns[index]
        user_content: str | Sequence[UserContent] = prompt if prompt is not None else turn.user_content
        response_text = response if response is not None else turn.response
        replacement: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content=user_content)]),
            ModelResponse(parts=[TextPart(content=response_text)]),
        ]
        self.messages[turn.start : turn.end] = replacement

    def replace_history(self, prompt: str, response: str) -> None:
        """Replace the whole history with one prompt/response turn."""
        self.messages = [
            ModelRequest(parts=[UserPromptPart(content=prompt)]),
            ModelResponse(parts=[TextPart(content=response)]),
        ]


@dataclass
class ConversationTurn:
    """One user prompt and the agent's final reply to it.

    A turn spans every message from the request carrying the user's prompt up to the next
    one, so tool calls and their results belong to the turn that triggered them.

    Attributes:
        start: Index of the turn's first message in :attr:`AgentState.messages`.
        end: Index one past the turn's last message.
        user_content: The user prompt as sent: a string, or a list mixing text and media.
        prompt: The prompt's text, with each image or file shown as ``[image]`` / ``[file]``.
        response: The text of the turn's last text response, or ``""`` if it has none.
    """

    start: int
    end: int
    user_content: str | Sequence[UserContent]
    prompt: str
    response: str


def user_prompt_text(content: str | Sequence[UserContent] | Any) -> str:
    """Render a user prompt as text, standing a marker in for each piece of media."""
    if isinstance(content, str):
        return content
    if not isinstance(content, (list, tuple)):
        return str(content)
    pieces = []
    for item in content:
        if isinstance(item, str):
            pieces.append(item)
        elif isinstance(item, (BinaryContent, ImageUrl)) and _is_image(item):
            pieces.append("[image]")
        else:
            pieces.append("[file]")
    return "\n".join(pieces)


def _is_image(item: BinaryContent | ImageUrl) -> bool:
    if isinstance(item, ImageUrl):
        return True
    return item.is_image


def _split_turns(messages: list[ModelMessage]) -> list[ConversationTurn]:
    starts = [
        i
        for i, message in enumerate(messages)
        if isinstance(message, ModelRequest) and any(isinstance(part, UserPromptPart) for part in message.parts)
    ]
    turns = []
    for position, start in enumerate(starts):
        end = starts[position + 1] if position + 1 < len(starts) else len(messages)
        request = messages[start]
        user_part = next(part for part in request.parts if isinstance(part, UserPromptPart))
        response = ""
        for message in messages[start:end]:
            if isinstance(message, ModelResponse):
                texts = [part.content for part in message.parts if isinstance(part, TextPart)]
                if texts:
                    response = "".join(texts)
        turns.append(
            ConversationTurn(
                start=start,
                end=end,
                user_content=user_part.content,
                prompt=user_prompt_text(user_part.content),
                response=response,
            )
        )
    return turns


# ---------------------------------------------------------------------------
# Legacy griptape wrapper
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DirectProvider:
    """A provider a Prompt Model Config node calls directly, reached through its OpenAI-compatible API."""

    base_url: str
    api_key_secret: str


DIRECT_PROVIDERS: dict[str, DirectProvider] = {
    "AnthropicPromptDriver": DirectProvider("https://api.anthropic.com/v1/", "ANTHROPIC_API_KEY"),
    "CoherePromptDriver": DirectProvider("https://api.cohere.ai/compatibility/v1", "COHERE_API_KEY"),
    "GrokPromptDriver": DirectProvider("https://api.x.ai/v1", "GROK_API_KEY"),
    "OpenAiChatPromptDriver": DirectProvider("https://api.openai.com/v1", "OPENAI_API_KEY"),
}
"""Endpoint and secret for each griptape driver type a Prompt Model Config node produces.

Keyed by the driver's ``type`` tag, as ``to_dict()`` writes it. An ``OpenAiChatPromptDriver``
with its own ``base_url`` (Groq, NIM, a custom endpoint) keeps that URL; the entry here is
the fallback.
"""


def _from_legacy_wrapper(value: dict[str, Any]) -> AgentState | None:
    """Convert a griptape ``Agent.to_dict()``, wrapped or bare, into an :class:`AgentState`.

    Only text survives: griptape memory stores each run's input and output as artifacts,
    and the wrapper already flattened non-text outputs to text. A plaintext ``api_key`` in
    the wrapper's ``provider`` blob is dropped; the provider is looked up by name instead.
    """
    is_wrapper = "agent" in value and "tools" in value
    agent_dict = value["agent"] if is_wrapper else value
    if not isinstance(agent_dict, dict):
        return None
    tasks = agent_dict.get("tasks") or []
    driver = (tasks[0].get("prompt_driver") if tasks and isinstance(tasks[0], dict) else None) or {}
    model = driver.get("model")
    if not model:
        return None

    provider_blob = value.get("provider") if is_wrapper else None
    provider = _legacy_provider(driver, provider_blob if isinstance(provider_blob, dict) else None)

    settings: dict[str, Any] = {}
    if driver.get("temperature") is not None:
        settings["temperature"] = driver["temperature"]
    if driver.get("max_tokens") is not None:
        settings["max_tokens"] = driver["max_tokens"]

    messages: list[ModelMessage] = []
    for run in (agent_dict.get("conversation_memory") or {}).get("runs", []):
        if not isinstance(run, dict):
            continue
        messages.append(ModelRequest(parts=[UserPromptPart(content=_artifact_text(run.get("input")))]))
        messages.append(ModelResponse(parts=[TextPart(content=_artifact_text(run.get("output")))]))

    return AgentState(
        provider=provider,
        model=model,
        model_settings=settings,
        rulesets=list(value.get("rulesets", [])) if is_wrapper else [],
        tools=list(value.get("tools", [])) if is_wrapper else [],
        messages=messages,
    )


def _legacy_provider(driver: dict[str, Any], provider_blob: dict[str, Any] | None) -> ProviderRef:
    if provider_blob and provider_blob.get("name") in DIRECT_PROVIDERS:
        return direct_provider_ref(provider_blob["name"], provider_blob.get("base_url"))
    if provider_blob:
        kind = ProviderKind.OLLAMA if provider_blob.get("type") == ProviderID.OLLAMA else ProviderKind.OPENAI_COMPATIBLE
        return ProviderRef(
            kind=kind,
            name=provider_blob.get("name") or "",
            base_url=provider_blob.get("base_url") or "",
        )
    driver_type = str(driver.get("type", ""))
    match driver_type:
        case t if t.startswith("GriptapeCloud"):
            return ProviderRef()
        case "OllamaPromptDriver":
            host = driver.get("host") or ""
            return ProviderRef(kind=ProviderKind.OLLAMA, name=ProviderID.OLLAMA, base_url=f"{host}/v1" if host else "")
        case _:
            return direct_provider_ref(driver_type, driver.get("base_url"))


def direct_provider_ref(driver_type: str, base_url: str | None) -> ProviderRef:
    """Return the :class:`ProviderRef` for a Prompt Model Config driver, by its type name."""
    known = DIRECT_PROVIDERS.get(driver_type)
    return ProviderRef(
        kind=ProviderKind.OPENAI_COMPATIBLE,
        name=driver_type,
        base_url=base_url or (known.base_url if known else ""),
        api_key_secret=known.api_key_secret if known else None,
    )


def _artifact_text(artifact: object) -> str:
    if isinstance(artifact, dict):
        value = artifact.get("value", "")
        if isinstance(value, list):
            return "\n".join(_artifact_text(item) for item in value)
        return value if isinstance(value, str) else json.dumps(value)
    if isinstance(artifact, str):
        return artifact
    return ""
