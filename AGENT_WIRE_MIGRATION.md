# Agent Wire Redesign — Phase 1 Complete

## Problem

Tool nodes, prompt driver nodes, and ruleset nodes all output live Python objects that
cannot survive `agent.to_dict()` / `from_dict()` round-trips:

- `MCPTool` holds a live connection object — not serializable
- `WebSearchDriver` is similarly broken
- `BasePromptDriver` subclasses have API keys baked in at creation time — keys don't
  survive serialization, so chained agents fail to authenticate
- Any tool, ruleset, or driver passed through a chained agent breaks on the second run

## Core Principle

> Build all framework objects fresh in memory at the point of use. Never wire Python objects
> across node boundaries. Wire serializable config dicts instead.

## The Agent Wire Format

Every `"Agent"` parameter carries a **wrapper dict**, not a raw griptape agent dict:

```python
{
    "agent": {
        # griptape agent.to_dict() output
        # conversation_memory is kept here — it's pure text, already serializable
        # tools, rulesets stripped (live in the fields below)
    },
    "tools": [
        {"tool_type": "WebSearch", "engine": "Exa", "off_prompt": False},
        {"tool_type": "MCPTool", "mcp_server_name": "blender", "server_config": {...}, "rules": "..."},
    ],
    "rulesets": [
        {"name": "my_rules", "rules": ["Be concise", "Always cite sources"]},
    ],
    # "prompt_driver" key reserved for Phase 2 — not yet populated
}
```

### Backward Compatibility

Old format (raw griptape dict without `"tools"` key) is detected and treated as having
no tools or rulesets:

```python
def unwrap_agent(value: dict) -> tuple[dict, list, list]:
    if "agent" in value and "tools" in value:
        return value["agent"], value.get("tools", []), value.get("rulesets", [])
    return value, [], []   # old format
```

## Tool Config Formats

### Simple tools (no secrets)
```python
{"tool_type": "Calculator",          "off_prompt": False}
{"tool_type": "WebScraper",          "off_prompt": False}
{"tool_type": "DateTime",            "off_prompt": False}
{"tool_type": "FileManager",         "off_prompt": False}
{"tool_type": "AudioTranscription",  "off_prompt": False}
{"tool_type": "PromptSummary",       "off_prompt": False}
```

### WebSearch
```python
{"tool_type": "WebSearch", "engine": "Exa",        "off_prompt": False}
{"tool_type": "WebSearch", "engine": "DuckDuckGo",  "off_prompt": False}  # broken upstream — see griptape #2198
{"tool_type": "WebSearch", "engine": "Google",      "off_prompt": False}
```
Secrets (`EXA_API_KEY`, `GOOGLE_API_KEY`, etc.) are fetched from SecretsManager at build time
and never stored in the config.

### MCPToolNode
```python
{
    "tool_type": "MCPTool",
    "mcp_server_name": "blender",
    "server_config": {
        "transport": "stdio",
        "command": "uv",
        "args": [...],
        # full connection config from engine's get_server_config()
    },
    "rules": "Only use Blender tools for 3D operations.",
}
```

### AgentToTool
```python
{
    "tool_type": "AgentTool",
    "agent_dict": {...},      # the wrapped agent dict (already serializable)
    "name": "ResearchAgent",
    "description": "Searches and summarises research papers.",
    "off_prompt": False,
}
```

## Ruleset Config Format

```python
{"name": "my_ruleset", "rules": ["Be concise", "Always cite sources"]}
```

## Utility Helpers (`utils/agent_utils.py`)

```python
wrap_agent(agent_dict, tool_configs, ruleset_configs) -> dict
unwrap_agent(value) -> tuple[dict, list, list]

build_tool_from_config(config) -> GtBaseTool
build_tools(tool_configs) -> list[GtBaseTool]
build_ruleset_from_config(config) -> Ruleset
build_rulesets_from_configs(ruleset_configs) -> list[Ruleset]
extract_rulesets_from_tool_configs(tool_configs) -> list[dict]
```

Every node that touches the Agent wire uses the same helpers:
- **Input**: `agent_core_dict, tool_configs, ruleset_configs = unwrap_agent(agent_input)`
- **Reconstruct**: `GriptapeNodesAgent().from_dict(agent_core_dict)`
- **Output**: clear tools from task, then `wrap_agent(agent.to_dict(), tool_configs, ruleset_configs)`

Always use `GriptapeNodesAgent().from_dict()`, never griptape's base `Agent.from_dict()` — the
base class rejects the wrapper dict keys as unexpected constructor arguments.

## Griptape Agent Dict — What Gets Stripped

`wrap_agent` strips live objects from the griptape dict before storing:

```python
agent_dict["rulesets"] = []
agent_dict["rules"] = []
for task in agent_dict.get("tasks", []):
    task["tools"] = []
    task["rulesets"] = []
    task["rules"] = []
```

`conversation_memory` stays intact — it's pure text and already serializable.

## ModelArtifact Coercion

When an agent uses `output_schema`, griptape stores the result as a `ModelArtifact` with a
dict value in conversation memory. This breaks the Anthropic API when a second agent reads
it (dict keys are passed as raw content block fields instead of `{"text": "..."}`).

`wrap_agent` coerces any non-`TextArtifact` memory output to a plain JSON string:

```python
for run in memory.get("runs", []):
    output = run.get("output", {})
    if isinstance(output, dict) and output.get("type") != "TextArtifact":
        value = output.get("value", "")
        if not isinstance(value, str):
            value = json.dumps(value)
        run["output"] = {"type": "TextArtifact", "value": value}
```

## Exa Monkey-Patch

exa-py v2 removed `use_autoprompt` from `search_and_contents()`. Griptape's
`ExaWebSearchDriver` still passes it, causing a `TypeError` at runtime.

A monkey-patch in `agent_utils.py` replaces `ExaWebSearchDriver.search` with a version that
omits the argument. Remove it when [griptape-ai/griptape#2200](https://github.com/griptape-ai/griptape/pull/2200)
merges and a release is cut.

## DuckDuckGo Status

DuckDuckGo web search is currently broken — the `duckduckgo_search` package was renamed to
`ddgs` and griptape's driver still imports the old name. The `web_search_tool.py` node shows
a warning badge on the DuckDuckGo option. Tracked in [griptape-ai/griptape#2198](https://github.com/griptape-ai/griptape/issues/2198).

## Files Changed (Phase 1)

### Tool nodes → output config dict
| File | Notes |
|---|---|
| `tools/calculator_tool.py` | Outputs `{"tool_type": "Calculator", ...}` |
| `tools/web_scraper_tool.py` | Outputs `{"tool_type": "WebScraper", ...}` |
| `tools/date_time_tool.py` | Outputs `{"tool_type": "DateTime", ...}` |
| `tools/file_manager_tool.py` | Outputs `{"tool_type": "FileManager", ...}` |
| `tools/audio_transcription_tool.py` | Outputs `{"tool_type": "AudioTranscription", ...}` |
| `tools/prompt_summary_tool.py` | Outputs `{"tool_type": "PromptSummary", ...}` |
| `tools/web_search_tool.py` | Engine variants; Exa default; DuckDuckGo warning badge |
| `tools/mcp_tool.py` | Re-enabled; embeds server config + rules |
| `convert/agent_to_tool.py` | Embeds wrapped agent dict + name + description |

### Ruleset nodes → output config dict
| File | Notes |
|---|---|
| `rules/create_ruleset.py` | Outputs `{"name": ..., "rules": [...]}` |
| `rules/ruleset_list.py` | Pass-through aggregator — no change needed |

### Agent-consuming nodes → unwrap/wrap
| File | Notes |
|---|---|
| `agents/agent.py` | Main node; unwrap → build tools + rulesets → run → strip → wrap |
| `image/describe_image.py` | unwrap → build tools + rulesets → wrap |
| `image/create_image.py` | unwrap → run → clear tools → wrap |
| `audio/transcribe_audio.py` | unwrap → run → clear tools → wrap |
| `tasks/mcp_task.py` | unwrap → `GriptapeNodesAgent.from_dict` → run → wrap |
| `agents/memory/display_agent_memory.py` | Reads memory directly from unwrapped dict — no reconstruction |
| `agents/memory/summarize_agent_memory.py` | unwrap → reconstruct → summarize → wrap |
| `agents/memory/clear_agent_memory.py` | unwrap → clear runs → wrap |
| `agents/memory/replace_item_in_agent_memory.py` | unwrap → replace run → wrap |

### Unchanged
| File | Reason |
|---|---|
| `tools/tool_list.py` | Aggregates whatever it receives — dicts pass through unchanged |
| `tools/extraction_tool.py` | Not yet enabled; takes a live `prompt_driver` object — needs Phase 2 before enabling |
| `tools/rag_tool.py` | Not yet enabled; takes a live `rag_engine` object — needs design before enabling |
| `tools/vector_store_tool.py` | Not yet enabled; takes a live vector store driver — needs design before enabling |

---

## Phase 2: Prompt Driver Migration (not yet started)

Prompt drivers don't currently *break* — API key values survive `to_dict()`/`from_dict()`
as plain strings. Phase 2 applies the same config dict pattern to keep secrets out of
serialized workflow state.

### Prompt Driver Config Format

```python
{
    "driver_type":  "GriptapeCloudPromptDriver",
    "model":        "gpt-4.1",
    "temperature":  0.7,
    "stream":       True,
    "secret_names": {"api_key": "GT_CLOUD_API_KEY"},  # name only, never the value
}
```

### New Helper
```python
build_prompt_driver_from_config(config, secrets_manager) -> BasePromptDriver
```

### Files to Change in Phase 2

| File | Notes |
|---|---|
| `utils/agent_utils.py` | Add `build_prompt_driver_from_config`; update `wrap_agent`/`unwrap_agent` |
| `config/prompt/griptape_cloud_prompt.py` | Output config dict |
| `config/prompt/anthropic_prompt.py` | Output config dict |
| `config/prompt/openai_prompt.py` | Output config dict |
| `config/prompt/groq_prompt.py` | Output config dict |
| `config/prompt/ollama_prompt_driver.py` | Output config dict (no secrets) |
| `config/prompt/amazon_bedrock_prompt.py` | Output config dict |
| `config/prompt/cohere_prompt.py` | Output config dict |
| `config/prompt/grok_prompt.py` | Output config dict |
| `config/prompt/nim_prompt.py` | Output config dict |
| `agents/agent.py` | Use `build_prompt_driver_from_config` when no explicit driver connected |
| `image/describe_image.py` | Already updated in Phase 1 for tools/rulesets; Phase 2 adds prompt driver config support |
