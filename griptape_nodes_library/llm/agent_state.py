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
    UserPromptPart,
)

from griptape_nodes_library.llm.model_config import ModelConfig, model_config_from_legacy_driver
from griptape_nodes_library.llm.rulesets import ruleset_to_config

AGENT_TYPE = "Agent"
WIRE_FORMAT = "pydantic_ai_agent"
WIRE_VERSION = 1
FINAL_RESULT_TOOL_PREFIX = "final_result"


@dataclass
class AgentState:
    model: ModelConfig | None = None
    messages: list[ModelMessage] = field(default_factory=list)
    tools: list[dict] = field(default_factory=list)
    rulesets: list[dict] = field(default_factory=list)

    # --- Wire ---

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

    # --- Runs view: one entry per user prompt and the final answer to it ---

    def runs(self) -> list[dict[str, str]]:
        return runs_from_messages(self.messages)

    def with_runs(self, runs: list[dict[str, Any]]) -> AgentState:
        return replace(self, messages=messages_from_runs(runs))


def is_agent_value(value: Any) -> bool:
    return isinstance(value, (AgentState, dict)) and bool(value)


# ---------------------------------------------------------------------------
# Messages <-> runs
# ---------------------------------------------------------------------------


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


def _as_text(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("value", "")
    elif isinstance(value, list):
        return "\n".join(_as_text(v) for v in value)
    if value is None:
        return ""
    return value if isinstance(value, str) else json.dumps(value)


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


# ---------------------------------------------------------------------------
# Legacy griptape formats
# ---------------------------------------------------------------------------


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
        # A bare `Agent.to_dict()` keeps its rulesets inline.
        agent_dict, tools, provider = value, [], None
        rulesets = [c for c in (ruleset_to_config(r) for r in value.get("rulesets") or []) if c]
    if not isinstance(agent_dict, dict):
        return AgentState(tools=tools, rulesets=rulesets)

    driver = _legacy_driver(agent_dict)
    model = model_config_from_legacy_driver(driver, provider) if driver else None
    memory = agent_dict.get("conversation_memory") or {}
    runs = memory.get("runs", []) if isinstance(memory, dict) else []
    return AgentState(model=model, messages=messages_from_runs(runs), tools=tools, rulesets=rulesets)
