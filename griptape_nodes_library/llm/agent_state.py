"""The `Agent` parameter value: model, message history, tools, and rulesets.

Wire format (plain JSON, safe to persist; holds no secret values)::

    {
        "format": "pydantic_ai_agent",
        "version": 1,
        "model": {...ModelConfig...} | None,
        "messages": [...pydantic-ai ModelMessage JSON...],
        "tools": [...tool config dicts...],
        "rulesets": [{"name": str, "rules": [str]}],
    }

:meth:`AgentState.from_wire` also reads the griptape-era formats saved by older
workflows: the `{"agent": <Agent.to_dict()>, "tools", "rulesets", "provider"}`
wrapper and a bare `Agent.to_dict()`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Any

from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from griptape_nodes_library.llm.model_config import ModelConfig, model_config_from_legacy_driver
from griptape_nodes_library.llm.rulesets import ruleset_to_config

WIRE_FORMAT = "pydantic_ai_agent"
WIRE_VERSION = 1
FINAL_RESULT_TOOL_PREFIX = "final_result"


@dataclass
class AgentState:
    model: ModelConfig | None = None
    messages: list[ModelMessage] = field(default_factory=list)
    tools: list[dict] = field(default_factory=list)
    rulesets: list[dict] = field(default_factory=list)

    def to_wire(self) -> dict[str, Any]:
        return {
            "format": WIRE_FORMAT,
            "version": WIRE_VERSION,
            "model": self.model.to_wire() if self.model else None,
            "messages": ModelMessagesTypeAdapter.dump_python(self.messages, mode="json"),
            "tools": list(self.tools),
            "rulesets": list(self.rulesets),
        }

    @classmethod
    def from_wire(cls, value: Any) -> AgentState:
        """Read any `Agent` value. Unrecognized input yields an empty state."""
        if isinstance(value, AgentState):
            return replace(value)
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                return cls()
        if not isinstance(value, dict):
            return cls()
        if value.get("format") == WIRE_FORMAT:
            return cls(
                model=ModelConfig.from_wire(value.get("model")),
                messages=ModelMessagesTypeAdapter.validate_python(value.get("messages") or []),
                tools=list(value.get("tools") or []),
                rulesets=list(value.get("rulesets") or []),
            )
        return _from_legacy(value)

    def runs(self) -> list[dict[str, str]]:
        return runs_from_messages(self.messages)

    def with_runs(self, runs: list[dict[str, Any]]) -> AgentState:
        return replace(self, messages=messages_from_runs(runs))


def is_agent_value(value: Any) -> bool:
    return isinstance(value, (AgentState, dict)) and bool(value)


def _user_text(part: UserPromptPart) -> str:
    content = part.content
    if isinstance(content, str):
        return content
    return "\n".join(item for item in content if isinstance(item, str))


def _response_text(response: ModelResponse) -> str:
    texts = [p.content for p in response.parts if isinstance(p, TextPart)]
    if texts:
        return "".join(texts)
    for part in response.parts:
        if isinstance(part, ToolCallPart) and part.tool_name.startswith(FINAL_RESULT_TOOL_PREFIX):
            return part.args_as_json_str()
    return ""


def runs_from_messages(messages: list[ModelMessage]) -> list[dict[str, str]]:
    runs: list[dict[str, str]] = []
    for message in messages:
        if isinstance(message, ModelRequest):
            prompts = [p for p in message.parts if isinstance(p, UserPromptPart)]
            if prompts:
                runs.append({"input": "\n".join(_user_text(p) for p in prompts), "output": ""})
        elif isinstance(message, ModelResponse) and runs:
            text = _response_text(message)
            if text:
                runs[-1]["output"] = text
    return runs


_NON_TEXT_ARTIFACTS = frozenset(
    {"ImageArtifact", "ImageUrlArtifact", "BlobArtifact", "AudioArtifact", "AudioUrlArtifact"}
)


def _as_text(value: Any) -> str:
    if isinstance(value, dict):
        if value.get("type") in _NON_TEXT_ARTIFACTS:
            return ""
        inner = value.get("value", "")
        # ModelArtifact / JsonArtifact hold a dict value: render it, don't unwrap it.
        if "type" in value and isinstance(inner, dict):
            return json.dumps(inner)
        return _as_text(inner)
    if isinstance(value, list):
        return "\n".join(t for t in (_as_text(v) for v in value) if t)
    if value is None:
        return ""
    return value if isinstance(value, str) else json.dumps(value)


_TOOL_RESULT_PREVIEW = 400


def _tool_exchange(calls: list[ToolCallPart], results: dict[str, str]) -> str:
    lines = ["[Verified tool use:"]
    for call in calls:
        lines.append(f"  Tool: {call.tool_name}")
        args = call.args_as_json_str()
        if args and args != "{}":
            lines.append(f"  Input: {args}")
        result = results.get(call.tool_call_id)
        if result is not None:
            if len(result) > _TOOL_RESULT_PREVIEW:
                result = result[:_TOOL_RESULT_PREVIEW] + "…"
            lines.append(f"  Result: {result}")
    lines.append("]")
    return "\n".join(lines) + "\n\n"


def compact_messages(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Collapse each run to user text and assistant text, recording tool use inline.

    Raw tool call/return parts break replay on a downstream agent whose tools differ:
    Anthropic and Bedrock reject `tool_use` blocks without matching tool definitions.
    """
    runs: list[dict[str, str]] = []
    calls: list[ToolCallPart] = []
    results: dict[str, str] = {}
    for message in messages:
        if isinstance(message, ModelRequest):
            prompts = [p for p in message.parts if isinstance(p, UserPromptPart)]
            if prompts:
                if runs and calls:
                    runs[-1]["output"] = _tool_exchange(calls, results) + runs[-1]["output"]
                calls, results = [], {}
                runs.append({"input": "\n".join(_user_text(p) for p in prompts), "output": ""})
            for part in message.parts:
                if isinstance(part, ToolReturnPart):
                    results[part.tool_call_id] = part.model_response_str()
        elif isinstance(message, ModelResponse) and runs:
            calls.extend(
                p
                for p in message.parts
                if isinstance(p, ToolCallPart) and not p.tool_name.startswith(FINAL_RESULT_TOOL_PREFIX)
            )
            text = _response_text(message)
            if text:
                runs[-1]["output"] = text
    if runs and calls:
        runs[-1]["output"] = _tool_exchange(calls, results) + runs[-1]["output"]
    return messages_from_runs(runs)


def messages_from_runs(runs: list[dict[str, Any]]) -> list[ModelMessage]:
    messages: list[ModelMessage] = []
    for run in runs:
        if not isinstance(run, dict):
            continue
        messages.append(ModelRequest(parts=[UserPromptPart(content=_as_text(run.get("input")))]))
        messages.append(ModelResponse(parts=[TextPart(content=_as_text(run.get("output")))]))
    return messages


def find_runs(data: Any) -> list[dict[str, Any]]:
    """Find a `runs` list anywhere in `data`, or treat a single `{input, output}` dict as one run."""
    if isinstance(data, dict):
        if isinstance(data.get("runs"), list):
            return data["runs"]
        for value in data.values():
            found = find_runs(value)
            if found:
                return found
        if "input" in data and "output" in data:
            return [data]
    elif isinstance(data, list):
        for item in data:
            found = find_runs(item)
            if found:
                return found
    return []


def _legacy_driver(agent_dict: dict) -> dict | None:
    for task in agent_dict.get("tasks") or []:
        if isinstance(task, dict) and isinstance(task.get("prompt_driver"), dict):
            return task["prompt_driver"]
    driver = agent_dict.get("prompt_driver")
    return driver if isinstance(driver, dict) else None


def _from_legacy(value: dict) -> AgentState:
    if "agent" in value and "tools" in value:
        agent_dict = value.get("agent") or {}
        tools = list(value.get("tools") or [])
        rulesets = list(value.get("rulesets") or [])
        provider = value.get("provider")
    else:
        agent_dict, tools, provider = value, [], None
        rulesets = [c for c in (ruleset_to_config(r) for r in value.get("rulesets") or []) if c]
    if not isinstance(agent_dict, dict):
        return AgentState(tools=tools, rulesets=rulesets)

    driver = _legacy_driver(agent_dict)
    model = model_config_from_legacy_driver(driver, provider) if driver else None
    memory = agent_dict.get("conversation_memory") or {}
    runs = memory.get("runs", []) if isinstance(memory, dict) else []
    return AgentState(model=model, messages=messages_from_runs(runs), tools=tools, rulesets=rulesets)
