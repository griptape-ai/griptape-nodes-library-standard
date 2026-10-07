"""Load and run workflows saved by the griptape-era library (main @ d19c834) on this branch."""

from __future__ import annotations

import ast
import asyncio
import json
import logging
import pickle
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import openai
import pytest
from griptape_nodes.node_library.library_registry import LibraryRegistry
from griptape_nodes.retained_mode.events.agent_events import (
    CreateAgentProviderRequest,
    CreateProviderPayload,
    DeleteAgentProviderRequest,
)
from griptape_nodes.retained_mode.events.base_events import EventRequest, ExecutionGriptapeNodeEvent
from griptape_nodes.retained_mode.events.execution_events import (
    ResolveNodeRequest,
    StartFlowRequest,
    UnresolveFlowRequest,
)
from griptape_nodes.retained_mode.events.node_events import UnresolveNodeRequest
from griptape_nodes.retained_mode.events.workflow_events import (
    RunWorkflowFromScratchRequest,
    RunWorkflowFromScratchResultSuccess,
    WorkflowStatus,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from pydantic_ai.messages import (
    BinaryContent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo

import griptape_nodes_library.llm.tools as tools_module
from griptape_nodes_library.llm.models import override_model
from griptape_nodes_library.llm.testing import fake_model

from .fixture_tools import rewrite_pickled_strings

HERE = Path(__file__).parent
FIXTURES = HERE / "fixtures" / "main"
MEDIA = HERE / "fixtures" / "media"
MCP_SERVER = HERE / "mcp_compat_server.py"
LIBRARY = "Griptape Nodes Library"

# Tiny valid PNG, returned by the patched image generator.
PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d4944415478da63f8cfc0f01f0005000201a5e2a6e70000000049454e44ae426082"
)


@dataclass
class ModelCall:
    tools: list[str]
    instructions: str
    user_prompts: list[str]
    output_tools: list[str]
    images: int


# Canned replies for nodes that parse the model's text, keyed by a prompt substring.
SCRIPTED = {
    "Please parse the timecodes": "00:00:00:00-00:00:02:00|Segment 1:\n00:00:02:00-00:00:04:00|Segment 2:",
}


@dataclass
class FakeLLM:
    calls: list[ModelCall] = field(default_factory=list)

    def respond(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        tools = sorted(t.name for t in info.function_tools)
        instructions = "\n".join(
            [info.instructions or ""]
            + [
                p.content
                for m in messages
                if isinstance(m, ModelRequest)
                for p in m.parts
                if isinstance(p, SystemPromptPart)
            ]
        )
        user_prompts = [
            p.content if isinstance(p.content, str) else " ".join(c for c in p.content if isinstance(c, str))
            for m in messages
            if isinstance(m, ModelRequest)
            for p in m.parts
            if isinstance(p, UserPromptPart)
        ]
        images = sum(
            isinstance(c, BinaryContent)
            for m in messages
            if isinstance(m, ModelRequest)
            for p in m.parts
            if isinstance(p, UserPromptPart) and not isinstance(p.content, str)
            for c in p.content
        )
        self.calls.append(ModelCall(tools, instructions, user_prompts, [t.name for t in info.output_tools], images))
        last = messages[-1]
        returned = isinstance(last, ModelRequest) and any(isinstance(p, ToolReturnPart) for p in last.parts)
        if not returned:
            wanted_calls = (
                ("shout", {"text": "hello"}),
                ("calculate", {"expression": "6*7"}),
                ("search", {"query": user_prompts[-1] if user_prompts else ""}),
                ("get_content", {"url": "https://example.com"}),
            )
            for wanted, args in wanted_calls:
                for name in tools:
                    if name == wanted or name.endswith(f"_{wanted}"):
                        call = ToolCallPart(tool_name=name, args=args, tool_call_id=f"call-{name}")
                        return ModelResponse(parts=[call])
        for needle, reply in SCRIPTED.items():
            if user_prompts and needle in user_prompts[-1]:
                return ModelResponse(parts=[TextPart(reply)])
        output_object = info.model_request_parameters.output_object
        if output_object is not None:
            return ModelResponse(parts=[TextPart(json.dumps(_example(output_object.json_schema)))])
        if info.output_tools and not info.allow_text_output:
            tool = info.output_tools[0]
            return ModelResponse(
                parts=[
                    ToolCallPart(tool_name=tool.name, args=_example(tool.parameters_json_schema), tool_call_id="out")
                ]
            )
        tool_text = "".join(
            f" [{p.tool_name}={p.model_response_str()}]"
            for m in messages
            if isinstance(m, ModelRequest)
            for p in m.parts
            if isinstance(p, ToolReturnPart)
        )
        return ModelResponse(parts=[TextPart(f"FAKE#{len(self.calls)}{tool_text}")])


def _example(schema: dict[str, Any], defs: dict[str, Any] | None = None) -> Any:
    known: dict[str, Any] = defs if defs is not None else schema.get("$defs") or {}
    if "$ref" in schema:
        return _example(known[schema["$ref"].split("/")[-1]], known)
    match schema.get("type"):
        case "object":
            return {k: _example(v, known) for k, v in (schema.get("properties") or {}).items()}
        case "array":
            return [_example(schema.get("items") or {"type": "string"}, known)]
        case "integer":
            return 1
        case "number":
            return 1.5
        case "boolean":
            return True
        case _:
            return "x"


@pytest.fixture
def fake_llm() -> Iterator[FakeLLM]:
    llm = FakeLLM()
    with override_model(fake_model(llm.respond)):
        yield llm


@dataclass
class CompatEnv:
    workspace: Path
    image_requests: list[tuple[Any, str]]


@pytest.fixture
def compat_engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_llm: FakeLLM) -> Iterator[CompatEnv]:
    """Configure the engine like a user machine: a workspace, the compat MCP server, a Cloud key."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = GriptapeNodes.ConfigManager()
    config.set_config_value("workspace_directory", str(workspace))
    config.set_config_value(
        "mcp_servers",
        [{"name": "compat", "transport": "stdio", "command": sys.executable, "args": [str(MCP_SERVER)]}],
    )
    provider = CreateProviderPayload(
        name="compat-openai",
        type="custom",
        model="gpt-4.1-mini",
        base_url="http://127.0.0.1:18999/v1",
        api_key_secret_name="OPENAI_API_KEY",
    )
    # The agent manager's config outlives the per-test engine reset, so add the provider once per test and remove it after.
    GriptapeNodes.handle_request(DeleteAgentProviderRequest(name=provider.name))
    created = GriptapeNodes.handle_request(CreateAgentProviderRequest(provider=provider))
    assert created.succeeded(), created.result_details
    monkeypatch.setenv("GT_CLOUD_API_KEY", "gt-compat-fake")
    for key in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "COHERE_API_KEY",
        "GROK_API_KEY",
        "GROQ_API_KEY",
        "NVIDIA_API_KEY",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "EXA_API_KEY",
    ):
        monkeypatch.setenv(key, "sk-compat-fake")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    # The engine imports node files under its own module names; patch the modules it loaded.
    library = LibraryRegistry.get_library(LIBRARY)
    create_image_module = sys.modules[library.get_node_class("GenerateImage").__module__]
    image_requests: list[tuple[Any, str]] = []

    def fake_generate_image(config: Any, prompt: str) -> bytes:
        image_requests.append((config, prompt))
        return PNG

    monkeypatch.setattr(create_image_module, "generate_image", fake_generate_image)
    monkeypatch.setattr(tools_module, "search_web", lambda query, engine: [{"title": f"{engine}:{query}", "url": "u"}])
    monkeypatch.setattr(tools_module, "scrape_url", lambda url: f"scraped {url}")

    async def fake_transcription(self: Any) -> None:
        await self._parse_result({"text": "fake transcript", "language": "en", "duration": 2.0}, "gen-compat")

    monkeypatch.setattr(library.get_node_class("TranscribeAudio"), "_process_generation", fake_transcription)
    # Constructors that call out to a provider, as they did on main.
    monkeypatch.setattr(
        library.get_node_class("FileManager"), "get_bucket_list", lambda _self: [("compat-bucket", "bucket-1")]
    )
    real_get = httpx.get

    def ollama_get(url: str, *args: Any, **kwargs: Any) -> Any:
        if str(url).endswith("/api/tags"):
            return httpx.Response(
                200, json={"models": [{"name": "llama3.2:latest"}]}, request=httpx.Request("GET", url)
            )
        return real_get(url, *args, **kwargs)

    monkeypatch.setattr(httpx, "get", ollama_get)
    models = SimpleNamespace(data=[SimpleNamespace(id=m) for m in ("gpt-4.1", "gpt-4.1-mini", "gpt-4o")])
    monkeypatch.setattr(
        openai, "Client", lambda *_a, **_k: SimpleNamespace(models=SimpleNamespace(list=lambda: models))
    )
    yield CompatEnv(workspace, image_requests)
    GriptapeNodes.handle_request(DeleteAgentProviderRequest(name=provider.name))


def fixture_source(name: str) -> str:
    return rewrite_pickled_strings(
        (FIXTURES / f"{name}.py").read_text(),
        {"__COMPAT_MEDIA__": str(MEDIA), "__COMPAT_PYTHON__": sys.executable, "__COMPAT_MCP_SERVER__": str(MCP_SERVER)},
    )


def materialize(name: str, dest: Path) -> Path:
    path = dest / f"{name}.py"
    path.write_text(fixture_source(name))
    return path


class _Records(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


async def load(path: Path) -> list[str]:
    """Load a saved workflow; return warning/error log lines raised while loading."""
    handler = _Records()
    logger = logging.getLogger("griptape_nodes")
    logger.addHandler(handler)
    try:
        result = await GriptapeNodes.ahandle_request(RunWorkflowFromScratchRequest(file_path=str(path)))
    finally:
        logger.removeHandler(handler)
    assert isinstance(result, RunWorkflowFromScratchResultSuccess), result.result_details
    assert result.status == WorkflowStatus.GOOD, result.result_details
    return [r.getMessage() for r in handler.records]


async def run_flow(*, fresh: bool = True, timeout: float = 120) -> None:
    """Run the loaded flow. `fresh` re-runs every node; otherwise saved resolved nodes keep their outputs."""
    flow_name = GriptapeNodes.ContextManager().get_current_flow().name
    if fresh:
        result = await GriptapeNodes.ahandle_request(UnresolveFlowRequest(flow_name=flow_name))
        assert result.succeeded(), result.result_details
    await _execute(StartFlowRequest(flow_name=flow_name), timeout)


async def run_node(name: str, timeout: float = 120) -> None:
    """Re-run one node; its upstream nodes keep the outputs saved in the workflow."""
    result = await GriptapeNodes.ahandle_request(UnresolveNodeRequest(node_name=name))
    assert result.succeeded(), result.result_details
    await _execute(ResolveNodeRequest(node_name=name), timeout)


async def _execute(request: Any, timeout: float) -> None:
    events = GriptapeNodes.EventManager()
    events.initialize_queue()
    result = await GriptapeNodes.ahandle_request(request)
    assert result.succeeded(), result.result_details

    async def wait() -> None:
        queue = events.event_queue
        while True:
            event = await queue.get()
            if isinstance(event, EventRequest):
                await GriptapeNodes.ahandle_request(event.request)
            elif isinstance(event, ExecutionGriptapeNodeEvent):
                name = type(event.wrapped_event.payload).__name__
                if name == "ControlFlowResolvedEvent":
                    return
                if name == "ControlFlowCancelledEvent":
                    msg = f"Flow cancelled: {event.wrapped_event.payload}"
                    raise AssertionError(msg)
            queue.task_done()

    await asyncio.wait_for(wait(), timeout)


def node(name: str) -> Any:
    return GriptapeNodes.NodeManager().get_node_by_name(name)


def out(name: str, param: str) -> Any:
    n = node(name)
    if param in n.parameter_output_values:
        return n.parameter_output_values[param]
    return n.get_parameter_value(param)


def saved_values(name: str | Path) -> dict[tuple[str, str, bool], Any]:
    """Values saved in a fixture name or workflow file, keyed by (node, parameter, is_output)."""
    tree = ast.parse(name.read_text() if isinstance(name, Path) else fixture_source(name))
    unique: dict[str, Any] = {}
    names: dict[str, str] = {}
    calls: list[ast.Call] = []
    for item in ast.walk(tree):
        if isinstance(item, ast.Assign) and isinstance(item.targets[0], ast.Name):
            target = item.targets[0].id
            if target == "top_level_unique_values_dict":
                unique = eval(compile(ast.Expression(item.value), str(name), "eval"), {"pickle": pickle})  # noqa: S307
            elif target.endswith("_name"):
                for call in ast.walk(item.value):
                    if isinstance(call, ast.Call) and getattr(call.func, "id", "") == "CreateNodeRequest":
                        node_name = next(k.value for k in call.keywords if k.arg == "node_name")
                        names[target] = ast.literal_eval(node_name)
        elif isinstance(item, ast.Call) and getattr(item.func, "id", "") == "SetParameterValueRequest":
            calls.append(item)
    values: dict[tuple[str, str, bool], Any] = {}
    for call in calls:
        kwargs = {k.arg: k.value for k in call.keywords}
        node_var = kwargs["node_name"]
        value = kwargs["value"]
        assert isinstance(node_var, ast.Name)
        assert isinstance(value, ast.Subscript)
        key = (names[node_var.id], ast.literal_eval(kwargs["parameter_name"]), ast.literal_eval(kwargs["is_output"]))
        values[key] = unique[ast.literal_eval(value.slice)]
    return values
