from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from griptape_nodes_library.llm.model_config import ModelConfig, ModelProvider
from griptape_nodes_library.llm.runner import RunCallbacks, build_agent, output_to_text, run_agent

if TYPE_CHECKING:
    from pydantic_ai.messages import ModelMessage
    from pydantic_ai.toolsets import AbstractToolset

    from griptape_nodes_library.llm.runner import Prompt


def cloud_model_config(model: str) -> ModelConfig:
    return ModelConfig(provider=ModelProvider.GRIPTAPE_CLOUD, model=model)


def model_config_for_provider(
    provider_type: str | None,
    model: str,
    *,
    base_url: str | None,
    api_key_secret: str | None,
    api_key: str | None = None,
) -> ModelConfig:
    """Map an engine chat provider entry (`ollama`, `lmstudio`, or an OpenAI-compatible `custom`) to a `ModelConfig`."""
    match provider_type:
        case "ollama":
            provider = ModelProvider.OLLAMA
        case "lmstudio":
            provider = ModelProvider.LMSTUDIO
        case _:
            provider = ModelProvider.OPENAI_COMPATIBLE
    return ModelConfig(
        provider=provider,
        model=model,
        base_url=base_url or None,
        api_key_secret=api_key_secret or None,
        api_key=api_key or None,
    )


@dataclass
class TaskRunResult:
    output: Any
    text: str
    tool_results: list[str] = field(default_factory=list)
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
) -> TaskRunResult:
    agent = build_agent(
        model_config, instructions=instructions, rulesets=rulesets, toolsets=toolsets, output_type=output_type
    )
    tool_results: list[str] = []

    def collect_tool_result(_name: str, result: str) -> None:
        tool_results.append(result)

    result = run_agent(
        agent,
        prompt,
        message_history=message_history,
        callbacks=RunCallbacks(on_text=on_text, on_tool_call=on_tool_call, on_tool_result=collect_tool_result),
    )
    return TaskRunResult(
        output=result.output,
        text=output_to_text(result.output),
        tool_results=tool_results,
        messages=result.all_messages(),
    )
