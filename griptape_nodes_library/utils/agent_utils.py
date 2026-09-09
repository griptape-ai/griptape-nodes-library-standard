"""Shared helpers for the Agent wire format.

Tools, rulesets, and prompt drivers are serialized as config dicts on the
wire and rebuilt fresh at the point of use.  Every node that produces or
consumes an Agent value imports from here so the logic lives in one place.
"""

import copy
import logging
from typing import Any, cast

from griptape.drivers.prompt.base_prompt_driver import BasePromptDriver
from griptape.drivers.prompt.ollama import OllamaPromptDriver
from griptape.drivers.prompt.openai import OpenAiChatPromptDriver
from griptape.rules import Rule, Ruleset
from griptape.tasks import PromptTask
from griptape_nodes.drivers.cloud_models import ProviderID

from griptape_nodes_library.utils.cloud_credential_utils import missing_credential_message, resolve_cloud_api_key


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

logger = logging.getLogger("griptape_nodes")

# ---------------------------------------------------------------------------
# Wrap / unwrap
# ---------------------------------------------------------------------------


GRIPTAPE_CLOUD_DRIVER_PREFIX = "GriptapeCloud"
"""``type`` tag prefix griptape writes for its Griptape Cloud drivers in ``to_dict()``.

Matched as a prefix rather than against ``GriptapeCloudPromptDriver`` alone because
every ``GriptapeCloud*`` driver declares ``api_key`` the same unserializable way, and
a serialized agent can carry more than the prompt driver -- a conversation-memory
driver, or an image-generation driver on a swapped task.
"""


def _restored_cloud_credentials(agent_core_dict: dict, *, require_credential: bool) -> dict:
    """Return the agent dict with a freshly resolved ``api_key`` on every Cloud driver.

    ``api_key`` on a ``GriptapeCloud*`` driver is not marked serializable, so
    ``to_dict()`` drops it and ``from_dict()`` refills it from the attrs default -- a
    bare ``os.environ["GT_CLOUD_API_KEY"]`` read that never consults the License.
    Injecting the value *before* ``from_dict()`` is what fixes that: attrs takes the
    supplied value and never evaluates the environment-reading default, so the
    no-key-set ``KeyError`` is covered along with the wrong-key 401/402.

    Never mutates ``agent_core_dict``: a saved workflow pickles the upstream node's
    parameter value verbatim, so repairing in place would persist a License JWT to
    disk. The return is a copy when a Cloud driver was found and ``agent_core_dict``
    itself otherwise.

    Args:
        agent_core_dict: Serialized agent, as produced by ``Agent.to_dict()``.
        require_credential: Raise when a Cloud driver is present and no credential
            resolves, instead of injecting ``""``. True for callers that go on to send
            a request, so the missing credential is reported here rather than as a bare
            401 from Cloud -- see :func:`unwrap_agent`.

    Raises:
        KeyError: ``require_credential`` and no License or API key is set.
    """
    if not _iter_cloud_driver_dicts(agent_core_dict):
        return agent_core_dict
    api_key = resolve_cloud_api_key()
    if not api_key and require_credential:
        msg = missing_credential_message("use the incoming agent's Griptape Cloud driver")
        raise KeyError(msg)
    result = copy.deepcopy(agent_core_dict)
    for driver_dict in _iter_cloud_driver_dicts(result):
        driver_dict["api_key"] = api_key
    return result


def _iter_cloud_driver_dicts(node: Any) -> list[dict]:
    """Collect every serialized ``GriptapeCloud*`` driver dict nested under ``node``.

    Walks the structure instead of reaching for ``tasks[].prompt_driver`` so a Cloud
    driver in any position is repaired -- a conversation-memory driver hits the same
    stripped-``api_key`` problem. Non-GTC drivers are left alone: they are rebuilt from
    the wrapper's ``provider`` blob by :func:`restore_provider_driver`, and
    ``from_dict()`` rejects an ``api_key`` kwarg on a driver without one.
    """
    found: list[dict] = []
    if isinstance(node, dict):
        type_tag = node.get("type")
        if isinstance(type_tag, str) and type_tag.startswith(GRIPTAPE_CLOUD_DRIVER_PREFIX):
            found.append(node)
        for value in node.values():
            found.extend(_iter_cloud_driver_dicts(value))
    elif isinstance(node, list):
        for item in node:
            found.extend(_iter_cloud_driver_dicts(item))
    return found


def unwrap_agent(value: dict, *, require_credential: bool = True) -> tuple[dict, list, list]:
    """Return (agent_core_dict, tool_configs, ruleset_configs).

    Handles both the new wrapper format {"agent": {...}, "tools": [...], "rulesets": [...]}
    and the old raw griptape dict (backward compatibility — returns empty lists).
    Returns ({}, [], []) for non-dict input.

    Griptape Cloud drivers in the returned dict carry a freshly resolved ``api_key``
    (see :func:`_restored_cloud_credentials`), so a caller deserializing with
    ``from_dict()`` gets the License-first credential rather than a raw environment
    read. ``value`` is never modified.

    Args:
        value: The upstream node's ``agent`` parameter value.
        require_credential: Raise a user-facing ``KeyError`` when the agent carries a
            Griptape Cloud driver and no credential resolves. The default suits any
            caller that goes on to send a request: a connected agent bypasses the
            node's own ``validate_before_workflow_run`` credential check
            (``ProviderSelectionComponent.uses_griptape_cloud_driver`` returns ``False``
            once ``agent`` is connected), and several consumers define no such check at
            all, so the unwrap is where a missing credential is known and can still be
            named. Pass ``False`` from paths that only read or rewrite the wire dict --
            they should not fail for want of a credential they never use.

    Raises:
        KeyError: ``require_credential`` and no License or API key is set.
    """
    if not isinstance(value, dict):
        return {}, [], []
    if "agent" in value and "tools" in value:
        agent_core_dict = _restored_cloud_credentials(value["agent"], require_credential=require_credential)
        return agent_core_dict, value.get("tools", []), value.get("rulesets", [])
    return _restored_cloud_credentials(value, require_credential=require_credential), [], []


def _ollama_host_from_base_url(base_url: str) -> str | None:
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


def build_prompt_driver(
    *,
    provider_type: str | None,
    model: str,
    base_url: str,
    api_key: str | None = None,
    stream: bool = True,
) -> BasePromptDriver:
    """Build the correct prompt driver for a provider config.

    Uses the native OllamaPromptDriver for Ollama providers (required for tool
    calling — the OpenAI-compat path produces blank output). Falls through to
    OpenAiChatPromptDriver for all other provider types, including unknown/missing.

    Note: ollama.Client accepts no api_key, so api_key is silently ignored for
    Ollama providers. Ollama instances behind an auth reverse proxy should use a
    non-ollama provider type to have credentials forwarded.
    TODO: remove caveat once griptape exposes headers/api_key on OllamaPromptDriver
      https://github.com/griptape-ai/griptape/issues/2238
    """
    if provider_type == ProviderID.OLLAMA:
        # Ollama's OpenAI-compat /v1 endpoint drops tool_calls from streamed responses
        # (ollama/ollama#9084). The native /api/chat endpoint handles tool-call streaming
        # correctly, so we use OllamaPromptDriver instead of the OpenAI-compat driver.
        return OllamaPromptDriver(model=model, host=_ollama_host_from_base_url(base_url), stream=stream)
    return OpenAiChatPromptDriver(
        model=model,
        base_url=base_url,
        api_key=api_key or "not-needed",
        stream=stream,
    )


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

    provider_type = provider.get("type")
    if provider_type is None:
        logger.warning(
            "Saved agent wrapper has no provider 'type' — falling back to the OpenAI-compat driver. "
            "If this is an Ollama provider, tool calls may silently fail. "
            "Re-run the upstream Agent node to write the correct driver type into the wrapper."
        )
    cast(PromptTask, task).prompt_driver = build_prompt_driver(
        provider_type=provider_type,
        model=task.prompt_driver.model,
        base_url=provider.get("base_url", ""),
        api_key=provider.get("api_key"),
    )


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

            # A License is a valid Griptape Cloud credential; a license-only user has
            # no GT_CLOUD_API_KEY to read.
            api_key = resolve_cloud_api_key()
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
