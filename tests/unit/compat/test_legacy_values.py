"""Griptape-era `Agent`, memory, ruleset, and driver values read by the pydantic-ai adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from griptape.artifacts import (
    ImageArtifact,
    JsonArtifact,
    ListArtifact,
    ModelArtifact,
    TextArtifact,
)
from griptape.drivers.prompt.amazon_bedrock import AmazonBedrockPromptDriver
from griptape.drivers.prompt.anthropic import AnthropicPromptDriver
from griptape.drivers.prompt.cohere import CoherePromptDriver
from griptape.drivers.prompt.griptape_cloud import GriptapeCloudPromptDriver
from griptape.drivers.prompt.grok import GrokPromptDriver
from griptape.drivers.prompt.ollama import OllamaPromptDriver
from griptape.drivers.prompt.openai import OpenAiChatPromptDriver
from griptape.memory.structure import ConversationMemory, Run
from griptape.rules import Rule, Ruleset
from griptape.structures import Agent as GtAgent
from pydantic import BaseModel
from pydantic_ai import Agent as PydanticAgent
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from griptape_nodes_library.llm.agent_state import AgentState, find_runs, messages_from_runs, runs_from_messages
from griptape_nodes_library.llm.model_config import (
    DEFAULT_BASE_URLS,
    ModelConfig,
    ModelProvider,
    model_config_from_legacy_driver,
)
from griptape_nodes_library.llm.tools import build_toolset

from .harness import FIXTURES, saved_values

TEMPLATES = Path(__file__).parents[3] / "workflows" / "templates"


def _is_legacy_agent(value: Any) -> bool:
    return isinstance(value, dict) and (
        ("agent" in value and "tools" in value) or value.get("type") in {"Agent", "GriptapeNodesAgent"}
    )


def _saved_agents() -> list[Any]:
    params = []
    sources: list[str | Path] = [p.stem for p in sorted(FIXTURES.glob("*.py"))]
    sources += [p for p in sorted(TEMPLATES.glob("*.py")) if p.name != "__init__.py"]
    for source in sources:
        label = source if isinstance(source, str) else f"template-{source.stem}"
        seen: set[str] = set()
        for (node, param, is_output), value in saved_values(source).items():
            key = json.dumps(value, sort_keys=True, default=str)
            if _is_legacy_agent(value) and key not in seen:
                seen.add(key)
                params.append(pytest.param(value, id=f"{label}-{node}.{param}{'-out' if is_output else ''}"))
    return params


def _text(artifact: dict) -> str:
    value = artifact.get("value")
    if isinstance(value, list):
        return "\n".join(_text(v) for v in value if v.get("type") == "TextArtifact")
    return value if isinstance(value, str) else json.dumps(value)


@pytest.mark.parametrize("value", _saved_agents())
def test_every_saved_agent_value_reads(value: dict) -> None:
    """Every agent value main saved in the fixtures keeps its history, tools, rulesets, and model."""
    state = AgentState.from_wire(value)
    agent = value.get("agent", value)
    runs = agent["conversation_memory"]["runs"]
    assert state.runs() == [{"input": _text(r["input"]), "output": _text(r["output"])} for r in runs]
    assert state.tools == value.get("tools", [])
    if "tools" in value:
        assert state.rulesets == value["rulesets"]
    else:  # bare `Agent.to_dict()` keeps griptape Ruleset dicts inline
        assert state.rulesets == [
            {"name": r["name"], "rules": [rule["value"] for rule in r["rules"]]} for r in value.get("rulesets", [])
        ]
    driver = agent["tasks"][0]["prompt_driver"]
    assert state.model is not None
    assert state.model.model == driver["model"]
    # Round trip through the wire format loses nothing the branch reads.
    assert AgentState.from_wire(state.to_wire()).runs() == state.runs()


def test_list_artifact_memory_from_generate_image() -> None:
    """GenerateImage saved a run whose input was a ListArtifact of text and image."""
    value = saved_values("image_generation__gen_plain")[("gen_plain", "agent", True)]
    run = value["agent"]["conversation_memory"]["runs"][0]
    assert run["input"]["type"] == "ListArtifact"
    assert AgentState.from_wire(value).runs()[0]["input"] == _text(run["input"])


def _agent_dict(**kwargs: Any) -> dict:
    return GtAgent(**kwargs).to_dict()


class TestBareAgentToDict:
    """`Agent.to_dict()` values, as saved by versions before the wrapper existed."""

    def test_memory_artifact_shapes(self) -> None:
        memory = ConversationMemory(
            runs=[
                Run(input=TextArtifact("plain"), output=TextArtifact("reply")),
                Run(
                    input=ListArtifact(
                        [TextArtifact("look at"), ImageArtifact(b"\x89PNG", format="png", width=1, height=1)]
                    ),
                    output=ModelArtifact(_Person(name="Ada", age=36)),
                ),
                Run(input=TextArtifact("as json"), output=JsonArtifact({"ok": True})),
            ]
        )
        state = AgentState.from_wire(
            _agent_dict(
                prompt_driver=GriptapeCloudPromptDriver(model="gpt-4.1", api_key="k"), conversation_memory=memory
            )
        )
        assert state.runs() == [
            {"input": "plain", "output": "reply"},
            {"input": "look at", "output": json.dumps({"name": "Ada", "age": 36})},
            {"input": "as json", "output": json.dumps({"ok": True})},
        ]
        assert state.model == ModelConfig(
            provider=ModelProvider.GRIPTAPE_CLOUD, model="gpt-4.1", settings={"temperature": 0.1}
        )

    def test_inline_rulesets_with_rule_dicts(self) -> None:
        value = _agent_dict(
            prompt_driver=GriptapeCloudPromptDriver(model="gpt-4.1", api_key="k"),
            rulesets=[Ruleset(name="Tone", rules=[Rule("Be kind."), Rule("Be brief.")])],
        )
        assert value["rulesets"][0]["rules"][0]["type"] == "Rule"
        assert AgentState.from_wire(value).rulesets == [{"name": "Tone", "rules": ["Be kind.", "Be brief."]}]

    def test_json_string(self) -> None:
        value = _agent_dict(prompt_driver=GriptapeCloudPromptDriver(model="gpt-4.1", api_key="k"))
        assert AgentState.from_wire(json.dumps(value)).model is not None


class _Person(BaseModel):
    name: str
    age: int


LEGACY_DRIVERS = [
    pytest.param(
        GriptapeCloudPromptDriver(model="gpt-4.1-mini", api_key="k", temperature=0.2, extra_params={"top_p": 0.8}),
        ModelConfig(
            provider=ModelProvider.GRIPTAPE_CLOUD, model="gpt-4.1-mini", settings={"temperature": 0.2, "top_p": 0.8}
        ),
        id="griptape-cloud",
    ),
    pytest.param(
        OpenAiChatPromptDriver(model="gpt-4.1", api_key="k", max_tokens=300, seed=5, extra_params={"top_p": 0.8}),
        ModelConfig(
            provider=ModelProvider.OPENAI,
            model="gpt-4.1",
            settings={"temperature": 0.1, "max_tokens": 300, "seed": 5, "top_p": 0.8},
        ),
        id="openai",
    ),
    pytest.param(
        OpenAiChatPromptDriver(model="llama-3.1-8b-instant", api_key="k", base_url="https://api.groq.com/openai/v1"),
        ModelConfig(
            provider=ModelProvider.GROQ,
            model="llama-3.1-8b-instant",
            base_url=DEFAULT_BASE_URLS[ModelProvider.GROQ],
            settings={"temperature": 0.1},
        ),
        id="groq",
    ),
    pytest.param(
        OpenAiChatPromptDriver(
            model="meta/llama3-8b-instruct", api_key="k", base_url="https://integrate.api.nvidia.com/v1"
        ),
        ModelConfig(
            provider=ModelProvider.NIM,
            model="meta/llama3-8b-instruct",
            base_url=DEFAULT_BASE_URLS[ModelProvider.NIM],
            settings={"temperature": 0.1},
        ),
        id="nim",
    ),
    pytest.param(
        AnthropicPromptDriver(
            model="claude-haiku-4-5", api_key="k", temperature=0.3, max_tokens=300, top_p=0.8, top_k=40
        ),
        ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model="claude-haiku-4-5",
            settings={"temperature": 0.3, "max_tokens": 300, "top_p": 0.8, "top_k": 40},
        ),
        id="anthropic",
    ),
    pytest.param(
        CoherePromptDriver(model="command-r-plus", api_key="k", extra_params={"p": 0.8, "k": 40}),
        ModelConfig(
            provider=ModelProvider.COHERE,
            model="command-r-plus",
            settings={"temperature": 0.1, "top_p": 0.8, "top_k": 40},
        ),
        id="cohere",
    ),
    pytest.param(
        GrokPromptDriver(model="grok-3-beta", api_key="k"),
        ModelConfig(
            provider=ModelProvider.GROK,
            model="grok-3-beta",
            base_url="https://api.x.ai/v1",
            settings={"temperature": 0.1},
        ),
        id="grok",
    ),
    pytest.param(
        OllamaPromptDriver(model="llama3.2", host="http://127.0.0.1:11434"),
        ModelConfig(
            provider=ModelProvider.OLLAMA,
            model="llama3.2",
            base_url="http://127.0.0.1:11434/v1",
            settings={"temperature": 0.1},
        ),
        id="ollama",
    ),
]


@pytest.mark.parametrize(("driver", "expected"), LEGACY_DRIVERS)
def test_legacy_driver_dicts(driver: Any, expected: ModelConfig) -> None:
    assert model_config_from_legacy_driver(driver.to_dict()) == expected


def test_legacy_bedrock_driver() -> None:
    boto3 = pytest.importorskip("boto3")
    driver = AmazonBedrockPromptDriver(
        model="us.anthropic.claude-haiku-4-5", session=boto3.Session(region_name="us-east-1")
    )
    config = model_config_from_legacy_driver(driver.to_dict())
    assert (config.provider, config.model) == (ModelProvider.BEDROCK, "us.anthropic.claude-haiku-4-5")


class TestWrapper:
    def _wrapper(self, **extra: Any) -> dict:
        agent = _agent_dict(
            prompt_driver=OpenAiChatPromptDriver(model="gpt-4.1-mini", api_key="k", base_url="http://localhost:9/v1"),
            conversation_memory=ConversationMemory(runs=[Run(input=TextArtifact("hi"), output=TextArtifact("yo"))]),
        )
        return {
            "agent": agent,
            "tools": [{"tool_type": "Calculator", "off_prompt": False}],
            "rulesets": [{"name": "R", "rules": ["x"]}],
            **extra,
        }

    def test_provider_blob_with_raw_key(self) -> None:
        provider = {"name": "local", "type": "custom", "base_url": "http://localhost:9/v1", "api_key": "sk-raw"}
        state = AgentState.from_wire(self._wrapper(provider=provider))
        assert state.model is not None
        assert (state.model.provider, state.model.base_url, state.model.api_key) == (
            ModelProvider.OPENAI_COMPATIBLE,
            "http://localhost:9/v1",
            "sk-raw",
        )
        assert "sk-raw" not in json.dumps(state.to_wire())
        assert state.model.options == {"engine_provider": "local"}

    def test_provider_blob_without_type(self) -> None:
        """Wrappers written before the provider `type` key existed fall back to OpenAI-compatible, as main did."""
        state = AgentState.from_wire(
            self._wrapper(provider={"name": "old", "base_url": "http://localhost:9/v1", "api_key": "k"})
        )
        assert state.model is not None
        assert state.model.provider == ModelProvider.OPENAI_COMPATIBLE

    def test_ollama_provider_blob(self) -> None:
        state = AgentState.from_wire(
            self._wrapper(provider={"name": "o", "type": "ollama", "base_url": "http://localhost:11434/v1"})
        )
        assert state.model is not None
        assert state.model.provider == ModelProvider.OLLAMA

    def test_tools_and_rulesets(self) -> None:
        state = AgentState.from_wire(self._wrapper())
        assert state.tools == [{"tool_type": "Calculator", "off_prompt": False}]
        assert state.rulesets == [{"name": "R", "rules": ["x"]}]
        assert state.runs() == [{"input": "hi", "output": "yo"}]


@pytest.mark.parametrize(
    "value",
    [
        None,
        "",
        "not json",
        "[]",
        [],
        42,
        {},
        {"agent": None, "tools": None},
        {"agent": "nope", "tools": []},
        {"agent": {"tasks": "bad", "conversation_memory": "bad"}, "tools": [], "rulesets": None},
        {"agent": {"conversation_memory": {"runs": ["bad", {"input": None, "output": None}]}}, "tools": []},
        {"type": "Agent", "tasks": [], "rulesets": [{"no_name": True}, "bad"]},
    ],
)
def test_malformed_agent_values_do_not_raise(value: Any) -> None:
    state = AgentState.from_wire(value)
    assert isinstance(state.runs(), list)


@pytest.mark.parametrize(
    "value",
    [None, "", "gpt-4.1", 3, {}, {"provider": "openai"}, {"model": "x"}],
)
def test_model_config_from_wire_ignores_non_configs(value: Any) -> None:
    assert ModelConfig.from_wire(value) is None


def test_model_config_from_wire_reads_wire() -> None:
    config = ModelConfig(
        provider=ModelProvider.ANTHROPIC, model="claude-haiku-4-5", settings={"top_k": 1}, max_retries=3
    )
    assert ModelConfig.from_wire(config.to_wire()) == config


class TestAgentMemoryFormats:
    """`agent_memory` parameter values: the simplified form and griptape's full form."""

    FULL = ConversationMemory(runs=[Run(input=TextArtifact("q1"), output=TextArtifact("a1"))]).to_dict()

    @pytest.mark.parametrize(
        ("memory", "expected"),
        [
            ({"runs": [{"input": "q1", "output": "a1"}]}, [{"input": "q1", "output": "a1"}]),
            ({"runs": [{"input": {"value": "q1"}, "output": {"value": "a1"}}]}, [{"input": "q1", "output": "a1"}]),
            ({"input": "q1", "output": "a1"}, [{"input": "q1", "output": "a1"}]),
            (FULL, [{"input": "q1", "output": "a1"}]),
            ({"conversation_memory": FULL}, [{"input": "q1", "output": "a1"}]),
            ({"runs": []}, []),
            ({"something": "else"}, []),
            ({"runs": [{"foo": 1}, {"input": "q1", "output": "a1"}]}, [{"input": "q1", "output": "a1"}]),
        ],
    )
    def test_formats(self, memory: dict, expected: list[dict]) -> None:
        assert runs_from_messages(messages_from_runs(find_runs(memory))) == expected


def test_tool_errors_go_back_to_the_model() -> None:
    """griptape returned tool exceptions to the model; a bad argument must not fail the run."""

    def respond(messages: list, _info: AgentInfo) -> ModelResponse:
        returns = [p for m in messages if isinstance(m, ModelRequest) for p in m.parts if isinstance(p, ToolReturnPart)]
        if returns:
            return ModelResponse(parts=[TextPart(str(returns[0].content))])
        return ModelResponse(parts=[ToolCallPart("add_timedelta", {"timedelta_kwargs": {"years": 4}})])

    toolset = build_toolset({"tool_type": "DateTime"})
    assert toolset is not None
    result = PydanticAgent(FunctionModel(respond), toolsets=[toolset]).run_sync("in four years")
    assert result.output.startswith("Error: 'years' is an invalid keyword argument")
