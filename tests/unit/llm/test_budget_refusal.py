"""A Griptape Cloud budget refusal stops the run, with the budgets named, on the pydantic-ai path.

Cloud refuses an over-budget call with HTTP 403 and a body naming the budgets. These run
against a real socket so the SDK reads and unwraps the body the way production does.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from unittest.mock import MagicMock

import pytest
from griptape_nodes.node_library.library_registry import LibraryRegistry
from griptape_nodes.utils.budget_refusal import BUDGET_HALT_PREFIX, BudgetExceededError
from pydantic_ai import Agent as PydanticAgent
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

import griptape_nodes_library.utils.model_invocation as model_invocation_module
from griptape_nodes_library.agents.agent import Agent
from griptape_nodes_library.agents.memory.summarize_agent_memory import SummarizeAgentMemory
from griptape_nodes_library.llm.agent_state import AgentState, messages_from_runs
from griptape_nodes_library.llm.image_generation import ImageGenerationConfig, ImageProvider, generate_image
from griptape_nodes_library.llm.model_config import ModelConfig, ModelProvider
from griptape_nodes_library.llm.runner import prompt_model
from griptape_nodes_library.llm.tools import build_toolset
from griptape_nodes_library.number.askulator import Askulator
from griptape_nodes_library.text.search_web import SearchWeb
from griptape_nodes_library.video.split_video import SplitVideo

LIBRARY_NAME = "Griptape Nodes Library"

REFUSAL: dict[str, Any] = {
    "message": "Budget limit reached (tight).",
    "code": "budget_exceeded",
    "blocked_by": [
        {
            "budget_id": "3f1c6b4e-0000-4000-8000-000000000001",
            "budget_name": "tight",
            "scope_type": "ORGANIZATION",
            "reset_period": "MONTHLY",
            "enforcement": "HARD",
            "limit_credits": 100,
            "spent_credits": 90,
            "remaining_credits": 10,
            "requested_credits": 50,
            "frozen": False,
        }
    ],
    "effective_remaining_credits": 10,
    "spend_id": "5b2d9c10-0000-4000-8000-000000000002",
}
# The chat endpoint is OpenAI-compatible and nests the refusal under `error`; the image endpoint is flat.
CHAT_REFUSAL = {"error": {**REFUSAL, "type": "insufficient_quota", "param": None}}
IMAGE_REFUSAL = {**{k: v for k, v in REFUSAL.items() if k != "code"}, "error": "budget_exceeded"}
OTHER_403 = {"error": {"message": "You do not have permission.", "type": "forbidden", "code": "forbidden"}}


class _Cloud:
    """A stand-in Cloud that refuses every POST, and counts the calls."""

    def __init__(self, chat_body: dict, image_body: dict) -> None:
        self.paths: list[str] = []
        cloud = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                cloud.paths.append(self.path)
                payload = json.dumps(image_body if "images" in self.path else chat_body).encode()
                self.send_response(403)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args: object) -> None:
                """Silence the access log."""

        self._server = HTTPServer(("127.0.0.1", 0), Handler)
        self.base_url = f"http://127.0.0.1:{self._server.server_address[1]}"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def refusing_cloud(monkeypatch: pytest.MonkeyPatch) -> Iterator[_Cloud]:
    cloud = _Cloud(CHAT_REFUSAL, IMAGE_REFUSAL)
    monkeypatch.setenv("GT_CLOUD_BASE_URL", cloud.base_url)
    monkeypatch.setenv("GT_CLOUD_API_KEY", "key")
    yield cloud
    cloud.close()


class _Allowed:
    result_details = ""

    def failed(self) -> bool:
        return False


@pytest.fixture(autouse=True)
def _allow_model_invocation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(model_invocation_module, "declare_model_invocation_sync", lambda _node, _model: _Allowed())


CLOUD = ModelConfig(provider=ModelProvider.GRIPTAPE_CLOUD, model="gpt-4.1")


def _assert_names_the_budget(halt: BaseException | None) -> None:
    assert isinstance(halt, BudgetExceededError)
    assert str(halt).startswith(BUDGET_HALT_PREFIX)
    assert "tight" in str(halt)
    # Raised several frames below the node; `NodeManager` names the node on the way out.
    assert halt.node_name is None


def _drive(node: Any) -> None:
    gen = node.process()
    if gen is None:
        return
    try:
        func = next(gen)
        while True:
            func = gen.send(func())
    except StopIteration:
        return


class TestTheModelCallStops:
    def test_the_refusal_halts_with_the_budget_named(self, refusing_cloud: _Cloud) -> None:
        with pytest.raises(BudgetExceededError) as caught:
            prompt_model(CLOUD, "hello")
        _assert_names_the_budget(caught.value)

    def test_a_settled_refusal_is_asked_once(self, refusing_cloud: _Cloud) -> None:
        with pytest.raises(BudgetExceededError):
            prompt_model(CLOUD, "hello")
        assert refusing_cloud.paths == ["/api/v1/chat/completions"]

    def test_a_403_that_is_not_a_budget_refusal_is_left_alone(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cloud = _Cloud(OTHER_403, OTHER_403)
        monkeypatch.setenv("GT_CLOUD_BASE_URL", cloud.base_url)
        monkeypatch.setenv("GT_CLOUD_API_KEY", "key")
        try:
            with pytest.raises(ModelHTTPError):
                prompt_model(CLOUD, "hello")
        finally:
            cloud.close()


class TestNodesRaiseTheHalt:
    def test_agent(self, refusing_cloud: _Cloud) -> None:
        node = Agent(name="Agent")
        node.set_parameter_value("model", "gpt-4.1")
        node.set_parameter_value("prompt", "hello")
        with pytest.raises(BudgetExceededError) as caught:
            _drive(node)
        _assert_names_the_budget(caught.value)

    @pytest.mark.parametrize(
        ("node_class", "values"),
        [
            (SearchWeb, {"prompt": "griptape", "summarize": True}),
            (Askulator, {"instruction": "2+2"}),
        ],
        ids=["the shared task run, behind Search Web, Date and Time and Summarize Text", "Askulator's own run"],
    )
    def test_task_nodes(self, refusing_cloud: _Cloud, node_class: type, values: dict[str, Any]) -> None:
        node = node_class(name="refused_task")
        for name, value in values.items():
            node.set_parameter_value(name, value)
        with pytest.raises(BudgetExceededError) as caught:
            _drive(node)
        _assert_names_the_budget(caught.value)

    def test_summarize_agent_memory(self, refusing_cloud: _Cloud) -> None:
        node = SummarizeAgentMemory(name="Summarize")
        state = AgentState(model=CLOUD, messages=messages_from_runs([{"input": "q", "output": "a"}]))
        node.set_parameter_value("agent", state.to_wire())
        with pytest.raises(BudgetExceededError):
            node.process()

    def test_split_video_keeps_the_halt_on_its_wrapped_error(self, refusing_cloud: _Cloud) -> None:
        node = SplitVideo(name="Split")
        with pytest.raises(ValueError, match="parse timecodes") as caught:
            node._parse_timecodes_with_agent("00:00:00:00-00:00:01:00")
        _assert_names_the_budget(caught.value.__cause__)


class TestAHaltInsideAToolStopsTheRun:
    def test_an_agent_tool_refusal_is_raised_instead_of_answered_around(self, refusing_cloud: _Cloud) -> None:
        """A tool's ordinary error goes back to the model as text; a budget refusal does not."""
        toolset = build_toolset(
            {"tool_type": "AgentTool", "name": "Helper", "agent_dict": AgentState(model=CLOUD).to_wire()}
        )
        assert toolset is not None

        def respond(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[ToolCallPart("Helper", {"input": "hi"})])

        with pytest.raises(BudgetExceededError) as caught:
            PydanticAgent(FunctionModel(respond), toolsets=[toolset]).run_sync("hello")
        _assert_names_the_budget(caught.value)


class TestImageGenerationStops:
    def test_the_refusal_halts_with_the_budget_named(self, refusing_cloud: _Cloud) -> None:
        config = ImageGenerationConfig(provider=ImageProvider.GRIPTAPE_CLOUD, model="gpt-image-1-mini")
        with pytest.raises(BudgetExceededError) as caught:
            generate_image(config, "a cat")
        _assert_names_the_budget(caught.value)
        assert refusing_cloud.paths == ["/api/images/generations"]

    def test_generate_image_saves_nothing(self, refusing_cloud: _Cloud, monkeypatch: pytest.MonkeyPatch) -> None:
        """A refused or failed generation writes no file and uses no output version number."""
        library = LibraryRegistry.get_library(name=LIBRARY_NAME)
        node: Any = library.create_node(node_type="GenerateImage", name="GenerateImage")
        output_file = MagicMock()
        monkeypatch.setattr(node, "_output_file", output_file)
        monkeypatch.setattr(node, "publish_update_to_parameter", MagicMock())
        config = ImageGenerationConfig(provider=ImageProvider.GRIPTAPE_CLOUD, model="gpt-image-1-mini")

        with pytest.raises(BudgetExceededError):
            node._create_image(config, "a cat")

        output_file.build_file.assert_not_called()
        node.publish_update_to_parameter.assert_not_called()
