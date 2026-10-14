from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from griptape_nodes_library.llm.model_config import ModelConfig
from griptape_nodes_library.llm.runner import RunCallbacks, build_agent, output_to_text, run_agent

if TYPE_CHECKING:
    from pydantic_ai.messages import ModelMessage
    from pydantic_ai.toolsets import AbstractToolset
    from pydantic_ai.usage import UsageLimits

    from griptape_nodes_library.llm.runner import Prompt


@dataclass
class TaskRunResult:
    output: Any
    text: str
    messages: list[ModelMessage] = field(default_factory=list)


def run_task_agent(  # noqa: PLR0913
    model_config: ModelConfig,
    prompt: Prompt | None,
    *,
    instructions: str | None = None,
    rulesets: Sequence[dict] = (),
    toolsets: Sequence[AbstractToolset[Any]] = (),
    output_type: Any = str,
    message_history: list[ModelMessage] | None = None,
    on_text: Callable[[str], None] | None = None,
    on_tool_call: Callable[[str, str], None] | None = None,
    usage_limits: UsageLimits | None = None,
) -> TaskRunResult:
    agent = build_agent(
        model_config, instructions=instructions, rulesets=rulesets, toolsets=toolsets, output_type=output_type
    )
    result = run_agent(
        agent,
        prompt,
        message_history=message_history,
        callbacks=RunCallbacks(on_text=on_text, on_tool_call=on_tool_call),
        usage_limits=usage_limits,
    )
    return TaskRunResult(
        output=result.output,
        text=output_to_text(result.output),
        # Pruning trims what the model sees; memory keeps every run.
        messages=[*(message_history or []), *result.new_messages()],
    )
