"""Workflow specs for compat fixtures. nodes: name -> (type, values); conns: (src, param, dst, param)."""

from pathlib import Path

from griptape.artifacts import TextArtifact
from griptape.memory.structure import ConversationMemory, Run

MEDIA = str(Path(__file__).parents[1] / "fixtures" / "media")

FULL_MEMORY = ConversationMemory(
    runs=[
        Run(input=TextArtifact("What's the secret word?"), output=TextArtifact("The secret word is banana.")),
        Run(input=TextArtifact("And the secret number?"), output=TextArtifact("The secret number is 42.")),
    ]
).to_dict()
SIMPLE_MEMORY = {"runs": [{"input": "Remember: my favourite colour is teal.", "output": "Got it, teal."}]}

GTC_MODEL = "gpt-4.1-mini"

SPECS: dict[str, dict] = {}

SPECS["agent_tools_rules"] = {
    "nodes": {
        "calc": ("Calculator", {}),
        "dt": ("DateTime", {"off_prompt": False}),
        "scraper": ("WebScraper", {}),
        "search": ("WebSearch", {"search_engine": "DuckDuckGo"}),
        "tools": ("ToolList", {}),
        "style": ("Ruleset", {"name": "Style", "rules": "Always answer in uppercase.\nBe brief."}),
        "persona": ("Ruleset", {"name": "Persona", "rules": "You are a pirate."}),
        "rules": ("RulesetList", {}),
        "a1": (
            "Agent",
            {
                "model": GTC_MODEL,
                "prompt": "Use the calculator tool to compute 1234*5678. Reply with just the number.",
                "additional_context": "This is a math question.",
                "include_details": True,
            },
        ),
        "a2": ("Agent", {"prompt": "Add 1 to the number you just gave. Reply with only the number."}),
        "a3": ("Agent", {"model": "GPT-4.1 nano", "prompt": "What is 2+2? Use the calculator."}),
        "show": ("DisplayText", {}),
    },
    "conns": [
        ("calc", "tool", "tools", "tool_1"),
        ("dt", "tool", "tools", "tool_2"),
        ("scraper", "tool", "tools", "tool_3"),
        ("search", "tool", "tools", "tool_4"),
        ("style", "ruleset", "rules", "ruleset_1"),
        ("persona", "ruleset", "rules", "ruleset_2"),
        ("tools", "tool_list", "a1", "tools"),
        ("rules", "rulesets", "a1", "rulesets"),
        ("a1", "agent", "a2", "agent"),
        ("calc", "tool", "a3", "tools"),
        ("style", "ruleset", "a3", "rulesets"),
        ("a2", "output", "show", "text"),
    ],
}

SPECS["agent_memory"] = {
    "nodes": {
        "a1": ("Agent", {"model": GTC_MODEL, "prompt": "My name is Zed. Say hi in five words or fewer."}),
        "a2": ("Agent", {"prompt": "What is my name? One word."}),
        "display": ("DisplayAgentMemory", {}),
        "replace": ("ReplaceItemInAgentMemory", {"new_input": "My name is Kai.", "new_output": "Hi Kai!"}),
        "summarize": ("SummarizeAgentMemory", {"prompt": "Summarize our conversation in under ten words."}),
        "clear": ("ClearAgentMemory", {}),
        "a3": ("Agent", {"prompt": "Say the word fresh."}),
        "full_mem": (
            "Agent",
            {"model": GTC_MODEL, "agent_memory": FULL_MEMORY, "prompt": "What was the secret word? One word."},
        ),
        "simple_mem": (
            "Agent",
            {"model": "gtc_gpt_4_1_mini", "agent_memory": SIMPLE_MEMORY, "prompt": "What is my favourite colour?"},
        ),
    },
    "conns": [
        ("a1", "agent", "a2", "agent"),
        ("a2", "agent", "display", "agent"),
        ("a2", "agent", "replace", "agent"),
        ("replace", "agent", "summarize", "agent"),
        ("summarize", "agent", "clear", "agent"),
        ("clear", "agent", "a3", "agent"),
    ],
}

PROMPT_CONFIGS = {
    "GriptapeCloudPrompt": {"model": "gpt-4.1-nano", "top_p": 0.8},
    "OpenAiPrompt": {"model": "gpt-4.1", "top_p": 0.8},
    "AnthropicPrompt": {"model": "claude-haiku-4-5", "top_p": 0.8, "top_k": 40},
    "CoherePrompt": {"model": "command-r-plus", "p": 0.8, "k": 40},
    "GrokPrompt": {"model": "grok-3-mini-beta", "top_p": 0.8},
    "GroqPrompt": {"model": "llama-3.1-8b-instant", "top_p": 0.8},
    "NimPrompt": {"model": "meta/llama3-8b-instruct", "top_p": 0.8},
    "AmazonBedrockPrompt": {"model": "us.anthropic.claude-haiku-4-5-20251001-v1:0"},
    "OllamaPrompt": {"base_url": "http://127.0.0.1", "port": "11434"},
}
for node_type, extra in PROMPT_CONFIGS.items():
    SPECS[f"prompt_{node_type}"] = {
        "nodes": {
            "config": (
                node_type,
                {"temperature": 0.3, "max_attempts_on_fail": 3, "max_tokens": 300, "stream": False, **extra},
            ),
            "a1": ("Agent", {"prompt": "Say hello in exactly three words."}),
            "a2": ("Agent", {"prompt": "Now say goodbye in exactly three words."}),
            "show": ("DisplayText", {}),
        },
        "conns": [
            ("config", "prompt_model_config", "a1", "model"),
            ("a1", "agent", "a2", "agent"),
            ("a2", "output", "show", "text"),
        ],
    }

SPECS["third_party_provider"] = {
    "nodes": {
        "a1": ("Agent", {"model_provider": "compat-openai", "prompt": "Reply with the word alpha."}),
        "a2": ("Agent", {"prompt": "Reply with the word beta."}),
        "a3": ("Agent", {"prompt": "What two words have you said? Answer briefly."}),
    },
    "after": {"a1": {"model": "gpt-4.1-mini"}},
    "conns": [("a1", "agent", "a2", "agent"), ("a2", "agent", "a3", "agent")],
}

SPECS["image_generation"] = {
    "nodes": {
        "gtc_img": ("GriptapeCloudImage", {"model": "gpt-image-1-mini", "image_size": "1024x1536", "quality": "low"}),
        "gen_gtc": ("GenerateImage", {"prompt": "A flat red square icon", "output_file": "gtc_square.png"}),
        "oai_img": (
            "OpenAiImage",
            {
                "model": "gpt-image-1",
                "quality": "low",
                "background": "transparent",
                "output_format": "png",
                "moderation": "low",
            },
        ),
        "gen_oai": ("GenerateImage", {"prompt": "A flat blue circle icon", "output_file": "oai_circle.png"}),
        "grok_img": ("GrokImage", {"model": "grok-2-image-1212"}),
        "gen_grok": ("GenerateImage", {"prompt": "A flat green triangle icon", "output_file": "grok_triangle.png"}),
        "gen_plain": (
            "GenerateImage",
            {
                "model": "gpt-image-1-mini",
                "prompt": "A yellow star",
                "image_size": "1536x1024",
                "enhance_prompt": True,
                "include_details": True,
            },
        ),
        "describe": ("DescribeImage", {"model": GTC_MODEL, "prompt": "What shape and colour is this? Five words."}),
    },
    "conns": [
        ("gtc_img", "image_model_config", "gen_gtc", "model"),
        ("oai_img", "image_model_config", "gen_oai", "model"),
        ("grok_img", "image_model_config", "gen_grok", "model"),
        ("gen_gtc", "output", "describe", "image"),
    ],
}

SPECS["media"] = {
    "nodes": {
        "img": ("LoadImage", {"path": f"{MEDIA}/red.png"}),
        "img2": ("LoadImage", {"path": f"{MEDIA}/red.png"}),
        "img3": ("LoadImage", {"path": f"{MEDIA}/red.png"}),
        "img4": ("LoadImage", {"path": f"{MEDIA}/red.png"}),
        "describe": (
            "DescribeImage",
            {"model": GTC_MODEL, "prompt": "Name the colour. One word.", "description_only": True},
        ),
        "desc_agent_src": ("Agent", {"model": GTC_MODEL, "prompt": "Remember: answer in French."}),
        "describe_agent": ("DescribeImage", {"prompt": "What colour is this? One word."}),
        "openai_cfg": ("OpenAiPrompt", {"model": "gpt-4.1-mini"}),
        "describe_cfg": ("DescribeImage", {"prompt": "Is this image mostly red? yes or no."}),
        "describe_schema": (
            "DescribeImage",
            {
                "model": GTC_MODEL,
                "prompt": "Describe the image.",
                "output_schema": {
                    "type": "object",
                    "properties": {"colour": {"type": "string"}, "is_square": {"type": "boolean"}},
                    "required": ["colour", "is_square"],
                },
            },
        ),
        "audio": ("LoadAudio", {"path": f"{MEDIA}/hello.mp3"}),
        "transcribe": (
            "TranscribeAudio",
            {
                "model": "whisper-1",
                "language": "en",
                "prompt": "compatibility",
                "response_format": "verbose_json",
                "temperature": 0.1,
            },
        ),
        "video": ("LoadVideo", {"path": f"{MEDIA}/clip.mp4"}),
        "split": (
            "SplitVideo",
            {
                "split_by": "timecode",
                "timecodes": "00:00:00:00-00:00:02:00\n00:00:02:00-00:00:04:00",
                "output_file": "part.mp4",
            },
        ),
    },
    "conns": [
        ("img", "image", "describe", "image"),
        ("img2", "image", "describe_agent", "image"),
        ("desc_agent_src", "agent", "describe_agent", "agent"),
        ("img3", "image", "describe_cfg", "image"),
        ("openai_cfg", "prompt_model_config", "describe_cfg", "model"),
        ("img4", "image", "describe_schema", "image"),
        ("audio", "audio", "transcribe", "audio"),
        ("video", "video", "split", "video"),
    ],
}

SPECS["tasks_text"] = {
    "nodes": {
        "askulator": ("Askulator", {"instruction": "What is 15% of 80?", "model": "gpt-4.1-nano"}),
        "date": (
            "DateAndTime",
            {"prompt": "the first of march 2030 at noon", "format": "2024-06-15 12:00:00", "model": "gpt-4.1-nano"},
        ),
        "date_custom": (
            "DateAndTime",
            {"prompt": "christmas 2031", "format": "Custom format", "custom_format": "%d/%m/%Y"},
        ),
        "evaluate": (
            "EvaluateTextResult",
            {
                "input": "What is the capital of France?",
                "expected_output": "Paris",
                "actual_output": "The capital of France is Paris.",
                "criteria": "Is the actual output factually equivalent to the expected output?",
                "model": "gpt-4.1-mini",
            },
        ),
        "scrape": ("ScrapeWeb", {"prompt": "What is the title of https://example.com ?", "model": "gpt-4.1-mini"}),
        "search": (
            "SearchWeb",
            {"prompt": "Griptape Nodes", "summarize": True, "search_engine": "DuckDuckGo", "model": "gpt-4.1-mini"},
        ),
        "search_raw": ("SearchWeb", {"prompt": "pydantic ai", "summarize": False}),
        "summarize": (
            "SummarizeText",
            {
                "prompt": "The quick brown fox jumps over the lazy dog. " * 20 + "The dog did not mind.",
                "model": "gpt-4.1-nano",
            },
        ),
        "random": ("RandomText", {"input_text": "One. Two. Three. Four.", "seed": 7, "selection_type": "sentence"}),
        "schema": (
            "CreateAgentSchema",
            {"example_template": "Custom", "example": '{"name": "Ada", "age": 36}', "ruleset_example": ""},
        ),
        "schema_agent": ("Agent", {"model": GTC_MODEL, "prompt": "Invent a person."}),
    },
    "conns": [
        ("schema", "schema", "schema_agent", "output_schema"),
        ("schema", "agent_ruleset", "schema_agent", "rulesets"),
    ],
}

SPECS["mcp_and_agent_tools"] = {
    "nodes": {
        "mcp_tool": ("MCPToolNode", {"mcp_server_name": "compat"}),
        "fm": ("FileManager", {"file_location": "Workspace Directory"}),
        "poet": ("Agent", {"model": GTC_MODEL}),
        "poet_rules": ("Ruleset", {"name": "Poet", "rules": "Only ever reply with a two-line rhyme."}),
        "poet_tool": ("AgentToTool", {"name": "Poet", "description": "Writes a two line rhyme about a topic."}),
        "tools": ("ToolList", {}),
        "boss": (
            "Agent",
            {
                "model": GTC_MODEL,
                "prompt": "Use the shout tool on the word hello, then ask the Poet for a rhyme about cats. Report both.",
            },
        ),
        "task": (
            "MCPTaskNode",
            {
                "mcp_server_name": "compat",
                "prompt": "Use the shout tool on the text 'mcp works'.",
                "max_subtasks": 5,
                "model": GTC_MODEL,
            },
        ),
        "task_agent_src": ("Agent", {"model": GTC_MODEL, "prompt": "Remember the codeword: owl."}),
        "task_agent": ("MCPTaskNode", {"mcp_server_name": "compat", "prompt": "Shout the codeword I told you."}),
    },
    "conns": [
        ("poet_rules", "ruleset", "poet", "rulesets"),
        ("poet", "agent", "poet_tool", "agent"),
        ("mcp_tool", "tool", "tools", "tool_1"),
        ("poet_tool", "tool", "tools", "tool_2"),
        ("fm", "tool", "tools", "tool_3"),
        ("tools", "tool_list", "boss", "tools"),
        ("task_agent_src", "agent", "task_agent", "agent"),
    ],
}


def _split(name: str) -> None:
    """Split a spec into one spec per connected component so a failing node does not block the rest."""
    spec = SPECS.pop(name)
    parent = {n: n for n in spec["nodes"]}

    def find(n: str) -> str:
        while parent[n] != n:
            n = parent[n]
        return n

    for src, _, dst, _ in spec.get("conns", []):
        parent[find(src)] = find(dst)
    groups: dict[str, list[str]] = {}
    for n in spec["nodes"]:
        groups.setdefault(find(n), []).append(n)
    for members in groups.values():
        key = f"{name}__{members[0]}"
        SPECS[key] = {
            "nodes": {n: spec["nodes"][n] for n in members},
            "conns": [c for c in spec.get("conns", []) if c[0] in members],
            "after": {n: v for n, v in spec.get("after", {}).items() if n in members},
        }


_split("tasks_text")
_split("media")
_split("image_generation")
