"""Shared helpers for the Agent wire format.

Tools, rulesets, and prompt drivers are serialized as config dicts on the
wire and rebuilt fresh at the point of use.  Every node that produces or
consumes an Agent value imports from here so the logic lives in one place.
"""

from typing import cast

from griptape.drivers.prompt.ollama import OllamaPromptDriver
from griptape.drivers.prompt.openai import OpenAiChatPromptDriver
from griptape.rules import Rule, Ruleset
from griptape.tasks import PromptTask
from griptape_nodes.drivers.cloud_models import ProviderID


# ---------------------------------------------------------------------------
# Temporary monkey-patch — remove when https://github.com/griptape-ai/griptape/pull/2200 is merged and released.
# exa-py v2 dropped use_autoprompt from search_and_contents(); ExaWebSearchDriver still passes it.
# ---------------------------------------------------------------------------
def _patch_exa_driver() -> None:
    try:
        from griptape.artifacts import JsonArtifact, ListArtifact
        from griptape.drivers.web_search.exa import ExaWebSearchDriver

        def _search(self: object, query: str, **kwargs: object) -> ListArtifact:
            response = self.client.search_and_contents(  # type: ignore[attr-defined]
                highlights=self.highlights,  # type: ignore[attr-defined]
                query=query,
                num_results=self.results_count,  # type: ignore[attr-defined]
                text=True,
                **self.params,  # type: ignore[attr-defined]
                **kwargs,
            )
            return ListArtifact(
                [
                    JsonArtifact({"title": r.title, "url": r.url, "highlights": r.highlights, "text": r.text})
                    for r in response.results
                ]
            )

        ExaWebSearchDriver.search = _search  # type: ignore[method-assign]
    except ImportError:
        pass  # exa extra not installed


_patch_exa_driver()


# ---------------------------------------------------------------------------
# Wrap / unwrap
# ---------------------------------------------------------------------------


def unwrap_agent(value: dict) -> tuple[dict, list, list]:
    """Return (agent_core_dict, tool_configs, ruleset_configs).

    Handles both the new wrapper format {"agent": {...}, "tools": [...], "rulesets": [...]}
    and the old raw griptape dict (backward compatibility — returns empty lists).
    Returns ({}, [], []) for non-dict input.
    """
    if not isinstance(value, dict):
        return {}, [], []
    if "agent" in value and "tools" in value:
        return value["agent"], value.get("tools", []), value.get("rulesets", [])
    return value, [], []


def ollama_host_from_base_url(base_url: str) -> str | None:
    """Convert an OpenAI-compat Ollama base_url to the host expected by OllamaPromptDriver.

    Provider configs store the OpenAI-compat URL (e.g. http://localhost:11434/v1).
    OllamaPromptDriver uses the native Ollama client, which takes just the host
    without any path suffix (e.g. http://localhost:11434). Passing None is valid
    and causes ollama.Client to default to http://localhost:11434.
    """
    host = base_url.rstrip("/")
    if host.endswith("/v1"):
        host = host[:-3]
    return host or None


def restore_provider_driver(agent: object, wrapper: dict) -> None:
    """Rebuild the prompt driver from provider config stored in the wrapper.

    When a non-GTC agent is serialized via to_dict(), griptape strips the api_key.
    Callers that deserialize via from_dict() must call this immediately after to
    restore the correct driver for the provider (Ollama native or OpenAI-compatible).

    Note: wrappers produced before the "type" key was added to the provider dict
    (i.e. saved workflows from older versions) will have no "type" entry and fall
    through to the OpenAI-compat driver. This is a known gap — those workflows
    will need to be re-run once to pick up the correct driver.
    """
    provider = wrapper.get("provider") if isinstance(wrapper, dict) else None
    if not provider:
        return

    tasks = getattr(agent, "tasks", None)
    if not tasks:
        return
    task = tasks[0]
    if not isinstance(task, PromptTask):
        return

    model = task.prompt_driver.model
    base_url = provider.get("base_url", "")

    if provider.get("type") == ProviderID.OLLAMA:
        # Native Ollama driver is required for tool calling; the OpenAI-compat path produces blank output.
        # Trade-off: ollama.Client accepts no api_key, so Ollama instances behind an auth reverse proxy
        # cannot have their credentials forwarded here. Use a non-ollama provider type for that setup.
        # TODO: remove once griptape exposes headers/api_key on OllamaPromptDriver
        #   https://github.com/griptape-ai/griptape/issues/2238
        rebuilt = OllamaPromptDriver(model=model, host=ollama_host_from_base_url(base_url), stream=True)
    else:
        rebuilt = OpenAiChatPromptDriver(
            model=model,
            base_url=base_url,
            api_key=provider.get("api_key") or "not-needed",
            stream=True,
        )
    cast(PromptTask, task).prompt_driver = rebuilt


def wrap_agent(agent_dict: dict, tool_configs: list, ruleset_configs: list, *, provider: dict | None = None) -> dict:
    """Strip non-serializable fields from the agent dict and return the wrapper.

    Tools, rulesets, and rules are cleared from the griptape dict — they live
    in the wrapper's tool_configs / ruleset_configs lists and are rebuilt fresh
    on the next node.
    """
    import json as _json

    for task in agent_dict.get("tasks", []):
        task["tools"] = []
        task.pop("rulesets", None)
        task.pop("rules", None)
    agent_dict.pop("rulesets", None)
    agent_dict.pop("rules", None)

    # Coerce any non-TextArtifact memory outputs to plain text.
    # ModelArtifact (schema output) stores a dict as `value`, which the Anthropic
    # API rejects when it's reconstructed as a message content block downstream.
    memory = agent_dict.get("conversation_memory", {})
    for run in memory.get("runs", []):
        output = run.get("output", {})
        if isinstance(output, dict) and output.get("type") != "TextArtifact":
            value = output.get("value", "")
            if not isinstance(value, str):
                value = _json.dumps(value)
            run["output"] = {
                "type": "TextArtifact",
                "value": value,
            }

    result: dict = {
        "agent": agent_dict,
        "tools": tool_configs,
        "rulesets": ruleset_configs,
    }
    if provider:
        result["provider"] = provider
    return result


# ---------------------------------------------------------------------------
# Ruleset helpers
# ---------------------------------------------------------------------------


def ruleset_to_config(ruleset: object) -> dict | None:
    """Convert a live Ruleset object to a serializable config dict.

    Already-serialized dicts pass through unchanged.
    """
    if isinstance(ruleset, dict):
        return ruleset
    try:
        return {"name": ruleset.name, "rules": [r.value for r in ruleset.rules]}  # type: ignore[union-attr]
    except AttributeError:
        return None


def build_rulesets_from_configs(configs: list) -> list:
    """Build live griptape Ruleset objects from serializable config dicts.

    Non-dict items (legacy live Ruleset objects) pass through unchanged.
    """
    result = []
    for config in configs:
        if isinstance(config, dict):
            rules = [Rule(r) for r in config.get("rules", [])]
            result.append(Ruleset(name=config["name"], rules=rules))
        else:
            result.append(config)
    return result


# ---------------------------------------------------------------------------
# Tool helpers
# ---------------------------------------------------------------------------


def build_tool_from_config(config: dict) -> object:
    """Build a live griptape tool from a serializable config dict.

    Dispatches on config["tool_type"].  New tool types must be added here.
    """
    tool_type = config.get("tool_type")

    if tool_type == "MCPTool":
        from griptape_nodes_library.utils.mcp_utils import create_mcp_tool

        return create_mcp_tool(config["mcp_server_name"], config["server_config"])

    if tool_type == "Calculator":
        from griptape.tools import CalculatorTool

        return CalculatorTool(off_prompt=config.get("off_prompt", False))

    if tool_type == "WebScraper":
        from griptape.tools import WebScraperTool

        return WebScraperTool(off_prompt=config.get("off_prompt", False))

    if tool_type == "DateTime":
        from griptape.tools import DateTimeTool

        return DateTimeTool(off_prompt=config.get("off_prompt", False))

    if tool_type == "FileManager":
        from griptape.drivers.file_manager.local import LocalFileManagerDriver
        from griptape.tools import FileManagerTool
        from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

        file_location = config.get("file_location", "Workspace Directory")
        if file_location == "GriptapeCloud":
            from griptape.drivers.file_manager.griptape_cloud import GriptapeCloudFileManagerDriver

            api_key = GriptapeNodes.SecretsManager().get_secret("GT_CLOUD_API_KEY")
            bucket_id = config.get("bucket_id", "")
            driver = GriptapeCloudFileManagerDriver(api_key=api_key, bucket_id=bucket_id)
        else:
            workdir = GriptapeNodes.ConfigManager().get_config_value("workspace_directory")
            driver = LocalFileManagerDriver(workdir=workdir)
        return FileManagerTool(file_manager_driver=driver, off_prompt=config.get("off_prompt", False))

    if tool_type == "AudioTranscription":
        from griptape.drivers.audio_transcription.openai import OpenAiAudioTranscriptionDriver
        from griptape.tools.audio_transcription.tool import AudioTranscriptionTool

        driver = OpenAiAudioTranscriptionDriver(model=config.get("model", "whisper-1"))
        return AudioTranscriptionTool(audio_transcription_driver=driver)

    if tool_type == "WebSearch":
        from griptape.tools import WebSearchTool
        from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

        engine = config.get("engine", "DuckDuckGo")
        off_prompt = config.get("off_prompt", False)
        if engine == "DuckDuckGo":
            from griptape.drivers.web_search.duck_duck_go import DuckDuckGoWebSearchDriver

            driver = DuckDuckGoWebSearchDriver()
        elif engine == "Google":
            from griptape.drivers.web_search.google import GoogleWebSearchDriver

            driver = GoogleWebSearchDriver(
                api_key=GriptapeNodes.SecretsManager().get_secret("GOOGLE_API_KEY"),
                search_id=GriptapeNodes.SecretsManager().get_secret("GOOGLE_API_SEARCH_ID"),
            )
        elif engine == "Exa":
            from griptape.drivers.web_search.exa import ExaWebSearchDriver

            driver = ExaWebSearchDriver(
                api_key=GriptapeNodes.SecretsManager().get_secret("EXA_API_KEY"),
            )
        else:
            msg = f"Unknown WebSearch engine: {engine}"
            raise ValueError(msg)
        return WebSearchTool(web_search_driver=driver, off_prompt=off_prompt)

    if tool_type == "AgentTool":
        from griptape.drivers.structure_run.local import LocalStructureRunDriver
        from griptape.tools import StructureRunTool

        from griptape_nodes_library.agents.griptape_nodes_agent import GriptapeNodesAgent
        from griptape_nodes_library.utils.utilities import to_pascal_case

        agent_wrapper = config["agent_dict"]
        agent_core_dict, incoming_tool_configs, incoming_ruleset_configs = unwrap_agent(agent_wrapper)
        agent = GriptapeNodesAgent.from_dict(agent_core_dict)
        restore_provider_driver(agent, agent_wrapper)
        if incoming_tool_configs:
            live_tools, _ = build_tools(incoming_tool_configs)
            if live_tools and agent.tasks:
                agent.tasks[0].tools = live_tools
        if incoming_ruleset_configs:
            agent._rulesets = build_rulesets_from_configs(incoming_ruleset_configs)
        driver = LocalStructureRunDriver(create_structure=lambda: agent)  # noqa: B023
        return StructureRunTool(
            name=to_pascal_case(config.get("name", "AgentTool")),
            description=(
                f"{config.get('description', 'An agent tool')}\n\n"
                "This tool requires an 'args' parameter as a list of strings. "
                'Example usage: { "values": { "args": ["your input here"] } }'
            ),
            structure_run_driver=driver,
            off_prompt=config.get("off_prompt", False),
        )

    msg = f"Unknown tool_type in config: {tool_type}"
    raise ValueError(msg)


def build_tools(tool_inputs: list) -> tuple[list, list]:
    """Split mixed tool inputs into (live_tools, tool_configs).

    Config dicts (dicts with a "tool_type" key) are rebuilt into live tools
    and also kept as configs for the output wrapper.  Live tool objects that
    are not config dicts pass through as-is but are NOT added to tool_configs
    (they cannot survive serialization).
    """
    live_tools: list = []
    tool_configs: list = []
    for item in tool_inputs:
        if isinstance(item, dict) and "tool_type" in item:
            tool_configs.append(item)
            live_tools.append(build_tool_from_config(item))
        else:
            live_tools.append(item)
    return live_tools, tool_configs
