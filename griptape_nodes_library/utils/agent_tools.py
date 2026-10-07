"""Expose griptape tools to a pydantic-ai agent.

The tool nodes produce griptape tools, or config dicts that rebuild into them (see
:func:`~griptape_nodes_library.utils.agent_utils.build_tool_from_config`). Each activity of
a griptape tool becomes one pydantic-ai tool with the same name, description, and argument
schema the griptape prompt drivers send, so the model sees the tools it saw before.

Griptape task memory doesn't apply here: an ``off_prompt`` tool's output goes to the model
like any other tool's.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from griptape.artifacts import BaseArtifact, ErrorArtifact
from griptape.tools import BaseTool
from pydantic_ai import ModelRetry
from pydantic_ai.tools import Tool

from griptape_nodes_library.utils.agent_utils import build_tool_from_config

_JSON_SCHEMA_ID = "http://json-schema.org/draft-07/schema#"


def build_pydantic_tools(tool_configs: list[dict[str, Any]]) -> list[Tool]:
    """Rebuild each tool config and expose its activities as pydantic-ai tools."""
    tools: list[Tool] = []
    for config in tool_configs:
        tools.extend(griptape_tool_to_pydantic(build_tool_from_config(config)))
    return tools


def griptape_tool_to_pydantic(tool: object) -> list[Tool]:
    """Return one pydantic-ai tool per activity of a live griptape tool.

    Raises:
        TypeError: ``tool`` is not a griptape tool.
    """
    if not isinstance(tool, BaseTool):
        msg = f"Expected a griptape tool, got {type(tool).__name__}."
        raise TypeError(msg)
    tools = []
    for activity in tool.activities():
        schema = tool.to_activity_json_schema(activity, _JSON_SCHEMA_ID)
        schema.pop("$id", None)
        schema.pop("$schema", None)
        tools.append(
            Tool.from_schema(
                _activity_function(activity),
                name=tool.to_native_tool_name(activity),
                description=tool.activity_description(activity),
                json_schema=schema,
            )
        )
    return tools


def _activity_function(activity: Callable[..., Any]) -> Callable[..., str]:
    def call(**kwargs: Any) -> str:
        # Activities unpack their arguments from a {"values": ...} dict.
        result = activity({"values": kwargs})
        if isinstance(result, ErrorArtifact):
            raise ModelRetry(result.to_text())
        if isinstance(result, BaseArtifact):
            return result.to_text()
        return "" if result is None else str(result)

    return call
