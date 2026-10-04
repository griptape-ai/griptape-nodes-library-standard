"""The interface nodes use to run an :class:`~griptape_nodes_library.utils.agent_state.AgentState`.

Nodes depend on the :class:`AgentRunner` protocol, not on how the run happens.
:class:`~griptape_nodes_library.utils.local_agent_runner.LocalAgentRunner` runs pydantic-ai
inside the node's own process. A runner that hands the state to the engine instead
implements the same ``run`` and changes nothing in the nodes.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from griptape.artifacts import ImageArtifact, ImageUrlArtifact
from pydantic_ai.messages import BinaryContent, ImageUrl, UserContent

if TYPE_CHECKING:
    from griptape_nodes_library.utils.agent_state import AgentState


# ---------------------------------------------------------------------------
# Events and results
# ---------------------------------------------------------------------------


@dataclass
class TextDelta:
    """A chunk of the agent's text response."""

    text: str


@dataclass
class ToolCalled:
    """The model called a tool."""

    tool_name: str
    args: str


@dataclass
class ToolReturned:
    """A tool call finished. ``is_error`` when the tool raised and the model was asked to retry."""

    tool_name: str
    content: str
    is_error: bool = False


AgentRunEvent = TextDelta | ToolCalled | ToolReturned


@dataclass
class AgentRunOutput:
    """What one run produced.

    Attributes:
        output: The final response: text, or the parsed object when the state has an
            output schema.
        state: The input state with this run's messages appended. On cancel, the state
            as it was before the run.
        cancelled: The run stopped because the node was cancelled.
    """

    output: str | dict[str, Any]
    state: AgentState
    cancelled: bool = False


class AgentRunner(Protocol):
    """Runs an agent for one prompt and reports progress as it goes."""

    def run(
        self,
        state: AgentState,
        prompt: Sequence[UserContent],
        *,
        on_event: Callable[[AgentRunEvent], None] | None = None,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> AgentRunOutput: ...


def image_prompt_content(image: object) -> UserContent | None:
    """Convert an image a node received into prompt content.

    URL artifacts and path strings stay URLs. An ``ImageArtifact`` is sent as its bytes;
    the runner stores those in history as a file URL, not inline.
    """
    if isinstance(image, ImageUrlArtifact):
        return ImageUrl(url=image.value)
    if isinstance(image, ImageArtifact):
        return BinaryContent(data=image.value, media_type=image.mime_type)
    if isinstance(image, str) and image.strip():
        return ImageUrl(url=image.strip())
    return None


def format_output(output: str | dict[str, Any]) -> str:
    """Render a run's output for a string parameter."""
    return output if isinstance(output, str) else json.dumps(output, ensure_ascii=False)
