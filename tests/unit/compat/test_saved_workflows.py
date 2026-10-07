"""Workflows saved by the griptape-era library (main @ d19c834), loaded and run on pydantic-ai.

Fixtures under `fixtures/main/` were built, run, and saved by the engine with the main
library installed (see `fixtures/README.md`). "Fresh" runs re-execute every node;
"resumed" runs keep the nodes main saved as resolved, so their griptape-era output
values feed the nodes that do run.
"""

import json
import shutil
from typing import Any

import pytest
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

from griptape_nodes_library.llm.agent_state import AgentState
from griptape_nodes_library.llm.image_generation import ImageGenerationConfig, ImageProvider
from griptape_nodes_library.llm.model_config import DEFAULT_BASE_URLS, ModelConfig, ModelProvider
from griptape_nodes_library.llm.tools import build_toolset

from .harness import (
    FIXTURES,
    CompatEnv,
    FakeLLM,
    load,
    materialize,
    node,
    out,
    run_flow,
    run_node,
    saved_values,
)

pytestmark = pytest.mark.asyncio

ALL = sorted(p.stem for p in FIXTURES.glob("*.py"))
DATETIME_TOOLS = ["add_timedelta", "get_current_datetime", "get_datetime_diff", "get_relative_datetime"]


def state(node_name: str, param: str = "agent") -> AgentState:
    return AgentState.from_wire(out(node_name, param))


def model(node_name: str) -> ModelConfig:
    config = state(node_name).model
    assert config is not None
    return config


@pytest.mark.parametrize("name", ALL)
async def test_fixture_loads_cleanly(name: str, compat_engine: CompatEnv, tmp_path) -> None:
    assert await load(materialize(name, tmp_path)) == []


@pytest.mark.parametrize("name", ALL)
async def test_fixture_runs_fresh(name: str, compat_engine: CompatEnv, tmp_path, fake_llm: FakeLLM) -> None:
    await load(materialize(name, tmp_path))
    await run_flow()


@pytest.mark.parametrize("name", ALL)
async def test_fixture_runs_resumed(name: str, compat_engine: CompatEnv, tmp_path, fake_llm: FakeLLM) -> None:
    await load(materialize(name, tmp_path))
    await run_flow(fresh=False)


class TestAgentToolsRules:
    async def test_fresh(self, compat_engine: CompatEnv, tmp_path, fake_llm: FakeLLM) -> None:
        await load(materialize("agent_tools_rules", tmp_path))
        await run_flow()

        a1_call, a2_call, a3_call = (
            next(c for c in fake_llm.calls if c.user_prompts[-1].startswith(prefix))
            for prefix in ("Use the calculator", "Add 1", "What is 2+2")
        )
        assert a1_call.tools == sorted(["calculate", "get_content", "search", *DATETIME_TOOLS])
        assert a1_call.user_prompts == [
            "Use the calculator tool to compute 1234*5678. Reply with just the number.\nThis is a math question."
        ]
        for text in (
            "Ruleset name: Style",
            "Always answer in uppercase.\nBe brief.",
            "Ruleset name: Persona",
            "You are a pirate.",
        ):
            assert text in a1_call.instructions
        assert out("a1", "output") == f"FAKE#{fake_llm.calls.index(a1_call) + 2} [calculate=42]"

        # The downstream agent keeps the upstream agent's model, tools, rulesets, and history.
        assert a2_call.tools == a1_call.tools
        assert a2_call.instructions == a1_call.instructions
        assert a2_call.user_prompts[0] == a1_call.user_prompts[0]
        assert model("a2") == model("a1")
        assert (model("a1").provider, model("a1").model) == (ModelProvider.GRIPTAPE_CLOUD, "gpt-4.1-mini")
        assert out("show", "text") == out("a2", "output")

        # Direct tool and ruleset connections; the saved display name "GPT-4.1 nano" resolves to its model id.
        assert a3_call.tools == ["calculate"]
        assert "Style" in a3_call.instructions
        assert "pirate" not in a3_call.instructions
        assert model("a3").model == "gpt-4.1-nano"

    async def test_resumed_from_saved_wrapper(self, compat_engine: CompatEnv, tmp_path, fake_llm: FakeLLM) -> None:
        """`a2` reruns against the griptape wrapper main saved for `a1`'s output."""
        saved = saved_values("agent_tools_rules")
        await load(materialize("agent_tools_rules", tmp_path))
        await run_flow(fresh=False)

        call = next(c for c in fake_llm.calls if c.user_prompts[-1].startswith("What is 2+2"))
        assert call.tools == ["calculate"]
        # a2 and show stayed resolved, so their saved outputs survive.
        assert out("a2", "output") == saved[("a2", "output", True)]
        assert out("show", "text") == saved[("show", "text", True)]


class TestAgentMemory:
    async def test_memory_nodes_on_saved_agents(self, compat_engine: CompatEnv, tmp_path, fake_llm: FakeLLM) -> None:
        saved = saved_values("agent_memory")
        await load(materialize("agent_memory", tmp_path))

        await run_node("display")
        assert saved[("replace", "memory_to_replace", False)] == "0: My name is Zed. Say hi in five words or fewer."
        assert out("display", "memory") == saved[("display", "memory", True)]

        await run_node("replace")
        replaced = state("replace").runs()
        assert replaced[0] == {"input": "My name is Kai.", "output": "Hi Kai!"}
        assert replaced[1]["input"] == "What is my name? One word."

        await run_node("summarize")
        summary_call = fake_llm.calls[-1]
        assert summary_call.user_prompts[:2] == ["My name is Kai.", "What is my name? One word."]
        assert summary_call.user_prompts[-1] == "Summarize our conversation in under ten words."
        assert state("summarize").runs() == [{"input": "conversation summary", "output": out("summarize", "summary")}]
        assert model("summarize") == ModelConfig(
            provider=ModelProvider.GRIPTAPE_CLOUD, model="gpt-4.1-mini", settings={"temperature": 0.1}
        )

        await run_node("clear")
        assert state("clear").runs() == []
        assert model("clear") == model("summarize")

        await run_node("a3")
        assert fake_llm.calls[-1].user_prompts == ["Say the word fresh."]

    async def test_chained_agent_reads_saved_history(
        self, compat_engine: CompatEnv, tmp_path, fake_llm: FakeLLM
    ) -> None:
        saved = saved_values("agent_memory")
        main_runs = saved[("a1", "agent", True)]["agent"]["conversation_memory"]["runs"]
        await load(materialize("agent_memory", tmp_path))

        await run_node("a2")
        call = fake_llm.calls[-1]
        assert call.user_prompts == [main_runs[0]["input"]["value"], "What is my name? One word."]
        runs = state("a2").runs()
        assert runs[0]["output"] == main_runs[0]["output"]["value"]
        assert runs[1] == {"input": "What is my name? One word.", "output": out("a2", "output")}

    @pytest.mark.parametrize(
        ("node_name", "history"),
        [
            ("full_mem", ["What's the secret word?", "And the secret number?"]),
            ("simple_mem", ["Remember: my favourite colour is teal."]),
        ],
    )
    async def test_agent_memory_parameter(
        self, node_name: str, history: list[str], compat_engine: CompatEnv, tmp_path, fake_llm: FakeLLM
    ) -> None:
        await load(materialize("agent_memory", tmp_path))
        await run_node(node_name)
        assert fake_llm.calls[-1].user_prompts[:-1] == history
        assert len(state(node_name).runs()) == len(history) + 1


class TestThirdPartyProvider:
    async def test_fresh(self, compat_engine: CompatEnv, tmp_path, fake_llm: FakeLLM) -> None:
        await load(materialize("third_party_provider", tmp_path))
        await run_flow()
        expected = ModelConfig(
            provider=ModelProvider.OPENAI_COMPATIBLE,
            model="gpt-4.1-mini",
            base_url="http://127.0.0.1:18999/v1",
            api_key_secret="OPENAI_API_KEY",
        )
        assert model("a1") == expected
        assert model("a3") == expected
        assert fake_llm.calls[-1].user_prompts == [
            "Reply with the word alpha.",
            "Reply with the word beta.",
            "What two words have you said? Answer briefly.",
        ]

    async def test_resumed_from_wrapper_with_raw_key(
        self, compat_engine: CompatEnv, tmp_path, fake_llm: FakeLLM
    ) -> None:
        """main forwarded the provider's raw API key in the wrapper; the branch must not persist it."""
        saved = saved_values("third_party_provider")
        assert saved[("a1", "agent", True)]["provider"]["api_key"] == "sk-compat-fake"
        await load(materialize("third_party_provider", tmp_path))

        await run_node("a2")
        assert fake_llm.calls[-1].user_prompts[0] == "Reply with the word alpha."
        config = model("a2")
        assert (config.provider, config.model, config.base_url) == (
            ModelProvider.OPENAI_COMPATIBLE,
            "gpt-4.1-mini",
            "http://127.0.0.1:18999/v1",
        )
        assert config.options == {"engine_provider": "compat-openai"}
        assert "sk-compat-fake" not in repr(out("a2", "agent"))


PROMPT_CONFIGS: dict[str, ModelConfig] = {
    "GriptapeCloudPrompt": ModelConfig(
        provider=ModelProvider.GRIPTAPE_CLOUD,
        model="gpt-4.1-nano",
        settings={"temperature": 0.3, "max_tokens": 300, "top_p": 0.8},
    ),
    "OpenAiPrompt": ModelConfig(
        provider=ModelProvider.OPENAI, model="gpt-4.1", settings={"temperature": 0.3, "max_tokens": 300, "top_p": 0.8}
    ),
    "AnthropicPrompt": ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        model="claude-haiku-4-5",
        settings={"temperature": 0.3, "max_tokens": 300, "top_p": 0.8, "top_k": 40},
    ),
    "CoherePrompt": ModelConfig(
        provider=ModelProvider.COHERE,
        model="command-r-plus",
        settings={"temperature": 0.3, "max_tokens": 300, "top_p": 0.8, "top_k": 40},
    ),
    "GrokPrompt": ModelConfig(
        provider=ModelProvider.GROK,
        model="grok-3-mini-beta",
        settings={"temperature": 0.3, "max_tokens": 300, "top_p": 0.8},
    ),
    "GroqPrompt": ModelConfig(
        provider=ModelProvider.GROQ,
        model="llama-3.1-8b-instant",
        base_url=DEFAULT_BASE_URLS[ModelProvider.GROQ],
        settings={"temperature": 0.3, "max_tokens": 300, "top_p": 0.8},
    ),
    "NimPrompt": ModelConfig(
        provider=ModelProvider.NIM,
        model="meta/llama3-8b-instruct",
        base_url=DEFAULT_BASE_URLS[ModelProvider.NIM],
        settings={"temperature": 0.3, "max_tokens": 300, "top_p": 0.8},
    ),
    "AmazonBedrockPrompt": ModelConfig(
        provider=ModelProvider.BEDROCK,
        model="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        settings={"temperature": 0.3, "max_tokens": 300},
    ),
    "OllamaPrompt": ModelConfig(
        provider=ModelProvider.OLLAMA,
        model="llama3.2:latest",
        base_url="http://127.0.0.1:11434/v1",
        settings={"temperature": 0.3, "max_tokens": 300},
    ),
}


@pytest.mark.parametrize("node_type", sorted(PROMPT_CONFIGS))
async def test_prompt_config_feeds_agent(node_type: str, compat_engine: CompatEnv, tmp_path, fake_llm: FakeLLM) -> None:
    await load(materialize(f"prompt_{node_type}", tmp_path))
    await run_flow()
    got = model("a1")
    expected = PROMPT_CONFIGS[node_type]
    assert (got.provider, got.model, got.base_url) == (expected.provider, expected.model, expected.base_url)
    assert {k: got.settings.get(k) for k in expected.settings} == expected.settings
    assert got.max_retries == 2
    assert model("a2") == got
    assert fake_llm.calls[-1].user_prompts == [
        "Say hello in exactly three words.",
        "Now say goodbye in exactly three words.",
    ]
    assert out("show", "text") == out("a2", "output")


class TestMcpAndAgentTools:
    async def test_fresh(self, compat_engine: CompatEnv, tmp_path, fake_llm: FakeLLM) -> None:
        await load(materialize("mcp_and_agent_tools", tmp_path))
        await run_flow()

        boss = next(c for c in fake_llm.calls if c.user_prompts[-1].startswith("Use the shout tool on the word hello"))
        assert boss.tools == [
            "Poet",
            "list_files_from_disk",
            "load_files_from_disk",
            "mcpCompat_shout",
            "save_content_to_file",
        ]
        assert "[mcpCompat_shout=HELLO!]" in out("boss", "output")
        assert "[mcpCompat_shout=HELLO!]" in out("task", "output")
        assert out("task", "was_successful") is True
        assert out("poet_tool", "tool")["name"] == "Poet"

    async def test_mcp_task_on_saved_agent(self, compat_engine: CompatEnv, tmp_path, fake_llm: FakeLLM) -> None:
        await load(materialize("mcp_and_agent_tools", tmp_path))
        await run_node("task_agent")
        call = fake_llm.calls[0]
        assert call.user_prompts == ["Remember the codeword: owl.", "Shout the codeword I told you."]
        assert "mcpCompat_shout" in call.tools
        assert out("task_agent", "was_successful") is True
        assert [r["input"] for r in state("task_agent").runs()] == call.user_prompts

    async def test_agent_tool_from_saved_config(self, compat_engine: CompatEnv, tmp_path, fake_llm: FakeLLM) -> None:
        """The saved AgentToTool config embeds a griptape agent dict; calling it runs that agent."""
        tool = saved_values("mcp_and_agent_tools")[("poet_tool", "tool", True)]
        assert tool["agent_dict"]["agent"]["type"] == "GriptapeNodesAgent"

        toolset = build_toolset(tool)
        assert toolset is not None
        assert list(toolset.tools) == ["Poet"]  # type: ignore[attr-defined]
        result = await toolset.tools["Poet"].function("cats")  # type: ignore[attr-defined]
        assert result.startswith("FAKE#")
        assert "Only ever reply with a two-line rhyme." in fake_llm.calls[-1].instructions


class TestImageGeneration:
    @pytest.mark.parametrize(
        ("name", "node_name", "expected"),
        [
            (
                "image_generation__gtc_img",
                "gen_gtc",
                ImageGenerationConfig(
                    provider=ImageProvider.GRIPTAPE_CLOUD,
                    model="gpt-image-1-mini",
                    image_size="1024x1536",
                    quality="low",
                ),
            ),
            (
                "image_generation__oai_img",
                "gen_oai",
                ImageGenerationConfig(
                    provider=ImageProvider.OPENAI,
                    model="gpt-image-1",
                    image_size="1024x1024",
                    quality="low",
                    style="vivid",
                    background="transparent",
                    moderation="low",
                    output_format="png",  # main sends compression only for jpeg
                ),
            ),
            (
                "image_generation__grok_img",
                "gen_grok",
                ImageGenerationConfig(provider=ImageProvider.GROK, model="grok-2-image-1212"),
            ),
        ],
    )
    async def test_driver_feeds_generate_image(
        self,
        name: str,
        node_name: str,
        expected: ImageGenerationConfig,
        compat_engine: CompatEnv,
        tmp_path,
        fake_llm: FakeLLM,
    ) -> None:
        await load(materialize(name, tmp_path))
        await run_flow()
        ((config, _prompt),) = compat_engine.image_requests
        assert config.model_copy(update={"api_key_secret": None, "base_url": None}) == expected
        assert out(node_name, "output") is not None

    async def test_describe_generated_image(self, compat_engine: CompatEnv, tmp_path, fake_llm: FakeLLM) -> None:
        await load(materialize("image_generation__gtc_img", tmp_path))
        await run_flow()
        assert fake_llm.calls[-1].images == 1
        assert out("describe", "output").startswith("FAKE#")

    async def test_plain_model_with_enhanced_prompt(
        self, compat_engine: CompatEnv, tmp_path, fake_llm: FakeLLM
    ) -> None:
        await load(materialize("image_generation__gen_plain", tmp_path))
        await run_flow()
        ((config, prompt),) = compat_engine.image_requests
        assert (config.provider, config.model, config.image_size) == (
            ImageProvider.GRIPTAPE_CLOUD,
            "gpt-image-1-mini",
            "1536x1024",
        )
        assert prompt.startswith("FAKE#")  # enhanced by the model
        assert "A yellow star" in fake_llm.calls[0].user_prompts[-1]


class TestMedia:
    async def test_describe_image(self, compat_engine: CompatEnv, tmp_path, fake_llm: FakeLLM) -> None:
        await load(materialize("media__img", tmp_path))
        await run_flow()
        call = fake_llm.calls[-1]
        assert call.images == 1
        assert call.user_prompts == ["Name the colour. One word.\n\nOutput image description only."]
        assert out("describe", "output") == "FAKE#1"

    async def test_describe_image_with_saved_agent(self, compat_engine: CompatEnv, tmp_path, fake_llm: FakeLLM) -> None:
        await load(materialize("media__img2", tmp_path))
        await run_flow(fresh=False)
        call = fake_llm.calls[-1]
        assert call.user_prompts[0] == "Remember: answer in French."
        assert call.images == 1

    async def test_describe_image_with_prompt_config(
        self, compat_engine: CompatEnv, tmp_path, fake_llm: FakeLLM
    ) -> None:
        await load(materialize("media__img3", tmp_path))
        await run_flow()
        assert (model("describe_cfg").provider, model("describe_cfg").model) == (ModelProvider.OPENAI, "gpt-4.1-mini")

    async def test_describe_image_with_schema(self, compat_engine: CompatEnv, tmp_path, fake_llm: FakeLLM) -> None:
        await load(materialize("media__img4", tmp_path))
        await run_flow()
        assert fake_llm.calls[-1].output_tools
        output: Any = out("describe_schema", "output")
        assert output in (
            {"colour": "x", "is_square": True},
            '{"colour": "x", "is_square": true}',
            '{"colour":"x","is_square":true}',
        )

    async def test_transcribe_audio(self, compat_engine: CompatEnv, tmp_path, fake_llm: FakeLLM) -> None:
        await load(materialize("media__audio", tmp_path))
        await run_flow()
        assert out("transcribe", "output") == "fake transcript"
        runs = state("transcribe").runs()
        assert runs[-1]["input"] == "I'm passing you some audio to transcribe."
        assert "fake transcript" in runs[-1]["output"]

    async def test_split_video(self, compat_engine: CompatEnv, tmp_path, fake_llm: FakeLLM) -> None:
        await load(materialize("media__video", tmp_path))
        await run_flow()
        assert len(out("split", "split_videos")) == 2
        assert out("split", "was_successful") is True


class TestTasksAndText:
    @pytest.mark.parametrize(
        ("name", "node_name", "param"),
        [
            ("tasks_text__random", "random", "output"),
            ("tasks_text__schema", "schema", "schema"),
        ],
    )
    async def test_deterministic_outputs_match_main(
        self, name: str, node_name: str, param: str, compat_engine: CompatEnv, tmp_path, fake_llm: FakeLLM
    ) -> None:
        saved = saved_values(name)[(node_name, param, True)]
        await load(materialize(name, tmp_path))
        await run_flow()
        assert out(node_name, param) == saved

    @pytest.mark.parametrize(
        ("name", "node_name", "prompt"),
        [
            ("tasks_text__askulator", "askulator", "What is 15% of 80?"),
            ("tasks_text__date", "date", "the first of march 2030 at noon"),
            ("tasks_text__evaluate", "evaluate", None),
            ("tasks_text__scrape", "scrape", "What is the title of https://example.com ?"),
            ("tasks_text__search", "search", "Griptape Nodes"),
            ("tasks_text__summarize", "summarize", None),
        ],
    )
    async def test_task_nodes_run(
        self, name: str, node_name: str, prompt: str | None, compat_engine: CompatEnv, tmp_path, fake_llm: FakeLLM
    ) -> None:
        await load(materialize(name, tmp_path))
        await run_flow()
        assert fake_llm.calls, "node did not call the model"
        if prompt is not None:
            assert prompt in fake_llm.calls[0].user_prompts[-1]

    async def test_search_without_summary_skips_model(
        self, compat_engine: CompatEnv, tmp_path, fake_llm: FakeLLM
    ) -> None:
        await load(materialize("tasks_text__search_raw", tmp_path))
        await run_flow()
        assert out("search_raw", "output").startswith(
            "[{'title': 'DuckDuckGo:Search the web for pydantic ai'"
        ) or "DuckDuckGo:Search the web for pydantic ai" in out("search_raw", "output")

    async def test_schema_agent(self, compat_engine: CompatEnv, tmp_path, fake_llm: FakeLLM) -> None:
        await load(materialize("tasks_text__schema", tmp_path))
        await run_flow()
        assert fake_llm.calls[-1].output_tools
        # main emitted structured output as a JSON string.
        output = out("schema_agent", "output")
        assert isinstance(output, str)
        assert set(json.loads(output)) == {"name", "age"}


TEMPLATES = sorted(p for p in (FIXTURES.parents[4] / "workflows" / "templates").glob("*.py") if p.name != "__init__.py")
# Warnings main raises loading these templates too; neither involves an LLM node.
TEMPLATE_LOAD_WARNINGS = {
    "fill_in_the_story": "Seedream Image Generation.",
    "flux_2_-_replace_a_face": "ParameterImage 'input_image': Conflicting values for 'hide_property'",
}
LLM_NODE_TYPES = {"Agent", "DescribeImage", "GenerateImage", "MCPTaskNode", "SummarizeAgentMemory", "AgentToTool"}


@pytest.mark.parametrize("path", TEMPLATES, ids=lambda p: p.stem)
async def test_template_loads_and_llm_nodes_run(path, compat_engine: CompatEnv, tmp_path, fake_llm: FakeLLM) -> None:
    copy = tmp_path / path.name
    shutil.copy(path, copy)
    expected = TEMPLATE_LOAD_WARNINGS.get(path.stem)
    assert [p for p in await load(copy) if not (expected and expected in p)] == []
    manager = GriptapeNodes.NodeManager()
    llm_nodes = [
        n for n in manager._name_to_parent_flow_name if type(manager.get_node_by_name(n)).__name__ in LLM_NODE_TYPES
    ]
    for name in llm_nodes:
        await run_node(name)
        if type(node(name)).__name__ == "Agent":
            assert AgentState.from_wire(out(name, "agent")).model is not None
