"""Turn `Tool` config dicts (`{"tool_type": ..., ...}`) into pydantic-ai toolsets.

Tool nodes emit config dicts rather than live objects so an `Agent` value stays
serializable; toolsets are rebuilt fresh wherever an agent runs.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import dateparser
from asteval import Interpreter
from griptape_nodes.agents.pydantic_ai.mcp_servers import mcp_server_from_config
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from openai import OpenAI
from pydantic_ai import FunctionToolset, Tool

from griptape_nodes_library.llm.agent_state import AgentState
from griptape_nodes_library.llm.runner import build_agent, output_to_text, run_agent_async
from griptape_nodes_library.llm.web import format_search_results, scrape_url, search_web
from griptape_nodes_library.utils.utilities import to_pascal_case

if TYPE_CHECKING:
    from pydantic_ai import Agent
    from pydantic_ai.toolsets import AbstractToolset

TOOL_TYPE = "Tool"


class ToolType(StrEnum):
    MCP = "MCPTool"
    CALCULATOR = "Calculator"
    WEB_SCRAPER = "WebScraper"
    DATE_TIME = "DateTime"
    FILE_MANAGER = "FileManager"
    AUDIO_TRANSCRIPTION = "AudioTranscription"
    WEB_SEARCH = "WebSearch"
    AGENT_TOOL = "AgentTool"


def is_tool_config(value: Any) -> bool:
    return isinstance(value, dict) and "tool_type" in value


def tool_configs_from_inputs(values: list[Any]) -> list[dict]:
    """Flatten a node's `tools` list input to config dicts. Non-config values raise."""
    configs: list[dict] = []
    for value in values:
        items = value if isinstance(value, list) else [value]
        for item in items:
            if item is None:
                continue
            if not is_tool_config(item):
                msg = f"Unsupported tool value of type {type(item).__name__}; connect a Tool node."
                raise TypeError(msg)
            configs.append(item)
    return configs


def tool_display_name(config: dict) -> str:
    return str(config.get("mcp_server_name") or config.get("name") or config.get("tool_type", "unknown"))


def _error(e: Exception) -> str:
    return f"Error: {e}"


def _calculator() -> FunctionToolset:
    def calculate(expression: str) -> str:
        """Compute a simple numerical or algebraic calculation.

        Args:
            expression: Arithmetic expression parsable in pure Python. Single line only.
        """
        interpreter = Interpreter(minimal=True)
        result = interpreter(expression)
        if interpreter.error:
            return f"Error calculating: {interpreter.error[0].get_error()[1]}"
        return str(result)

    return FunctionToolset([calculate])


def _date_time() -> FunctionToolset:
    def get_current_datetime() -> str:
        """Return the current date and time."""
        return str(datetime.now())  # noqa: DTZ005

    def get_relative_datetime(relative_date_string: str) -> str:
        """Return a relative date and time.

        Args:
            relative_date_string: Relative date in English compatible with the dateparser library,
                e.g. "now EST", "20 minutes ago", "in 2 days", "3 months, 1 week and 1 day ago".
        """
        parsed = dateparser.parse(relative_date_string)
        return str(parsed) if parsed else "Error: invalid relative date string"

    def add_timedelta(timedelta_kwargs: dict[str, float], iso_datetime: str | None = None) -> str:
        """Add a timedelta to a datetime.

        Args:
            timedelta_kwargs: Keyword arguments for `datetime.timedelta`, e.g. {"days": -1, "hours": 2}.
            iso_datetime: ISO 8601 datetime, e.g. "2021-01-01T00:00:00". Defaults to now.
        """
        base = datetime.fromisoformat(iso_datetime) if iso_datetime else datetime.now()  # noqa: DTZ005
        return (base + timedelta(**timedelta_kwargs)).isoformat()

    def get_datetime_diff(start_datetime: str, end_datetime: str) -> str:
        """Calculate end_datetime - start_datetime.

        Args:
            start_datetime: ISO 8601 datetime, e.g. "2021-01-01T00:00:00".
            end_datetime: ISO 8601 datetime, e.g. "2021-01-02T00:00:00".
        """
        return str(datetime.fromisoformat(end_datetime) - datetime.fromisoformat(start_datetime))

    return FunctionToolset([get_current_datetime, get_relative_datetime, add_timedelta, get_datetime_diff])


def _web_scraper() -> FunctionToolset:
    def get_content(url: str) -> str:
        """Browse a web page and load its content.

        Args:
            url: Valid HTTP URL.
        """
        try:
            return scrape_url(url)
        except Exception as e:
            return _error(e)

    return FunctionToolset([get_content])


def _web_search(engine: str) -> FunctionToolset:
    def search(query: str) -> str:
        """Search the web. Returns a list of pages with titles, descriptions, and URLs.

        Args:
            query: Search engine request.
        """
        try:
            return format_search_results(search_web(query, engine))
        except Exception as e:
            return _error(e)

    return FunctionToolset([Tool(search, description=f"Search the web via {engine}.")])


def _resolve_in(workdir: Path, relative: str) -> Path:
    path = (workdir / relative).resolve()
    if not path.is_relative_to(workdir):
        msg = f"Path '{relative}' is outside the workspace."
        raise ValueError(msg)
    return path


def _file_manager(config: dict) -> FunctionToolset:
    location = config.get("file_location", "Workspace Directory")
    if location != "Workspace Directory":
        msg = f"FileManager location '{location}' is not supported."
        raise ValueError(msg)
    workdir = Path(GriptapeNodes.ConfigManager().get_config_value("workspace_directory")).resolve()

    def list_files_from_disk(path: str = ".") -> str:
        """List files on disk.

        Args:
            path: Relative path in POSIX format, e.g. 'foo/bar'.
        """
        try:
            return "\n".join(sorted(p.name for p in _resolve_in(workdir, path).iterdir()))
        except Exception as e:
            return _error(e)

    def load_files_from_disk(paths: list[str]) -> str:
        """Load text files from disk.

        Args:
            paths: Relative file paths in POSIX format, e.g. ['foo/bar/file.txt'].
        """
        try:
            return "\n\n".join(f"--- {p} ---\n{_resolve_in(workdir, p).read_text(errors='replace')}" for p in paths)
        except Exception as e:
            return _error(e)

    def save_content_to_file(path: str, content: str) -> str:
        """Save content to a file.

        Args:
            path: Destination file path in POSIX format, e.g. 'foo/bar/baz.txt'.
            content: Text to write.
        """
        try:
            target = _resolve_in(workdir, path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
        except Exception as e:
            return _error(e)
        return "Successfully saved file"

    return FunctionToolset([list_files_from_disk, load_files_from_disk, save_content_to_file])


def _audio_transcription(config: dict) -> FunctionToolset:
    model = config.get("model", "whisper-1")

    def transcribe_audio_from_disk(path: str) -> str:
        """Generate a transcription of an audio file on disk.

        Args:
            path: Path to an audio file on disk.
        """
        try:
            api_key = GriptapeNodes.SecretsManager().get_secret("OPENAI_API_KEY", should_error_on_not_found=False)
            with Path(path).open("rb") as audio:
                return OpenAI(api_key=api_key).audio.transcriptions.create(model=model, file=audio).text
        except Exception as e:
            return _error(e)

    return FunctionToolset([transcribe_audio_from_disk])


def _agent_tool(config: dict) -> FunctionToolset:
    state = AgentState.from_wire(config.get("agent_dict"))

    async def run_agent_tool(input: str) -> str:  # noqa: A002
        # Building resolves credentials and attribution synchronously; keep it off the loop.
        agent = await asyncio.to_thread(build_agent_from_state, state)
        result = await run_agent_async(agent, input, message_history=state.messages)
        return output_to_text(result.output)

    description = config.get("description") or "An agent tool"
    tool = Tool(run_agent_tool, name=to_pascal_case(config.get("name") or "AgentTool"), description=description)
    return FunctionToolset([tool])


def _mcp(config: dict) -> AbstractToolset[Any] | None:
    server_name = str(config.get("mcp_server_name", ""))
    clean_name = "".join(c for c in server_name if c.isalnum())
    built = mcp_server_from_config(f"mcp{clean_name.title()}", config.get("server_config") or {})
    return built.toolset if built else None


def build_toolset(config: dict) -> AbstractToolset[Any] | None:
    match config.get("tool_type"):
        case ToolType.MCP:
            return _mcp(config)
        case ToolType.CALCULATOR:
            return _calculator()
        case ToolType.WEB_SCRAPER:
            return _web_scraper()
        case ToolType.DATE_TIME:
            return _date_time()
        case ToolType.FILE_MANAGER:
            return _file_manager(config)
        case ToolType.AUDIO_TRANSCRIPTION:
            return _audio_transcription(config)
        case ToolType.WEB_SEARCH:
            return _web_search(config.get("engine", "DuckDuckGo"))
        case ToolType.AGENT_TOOL:
            return _agent_tool(config)
        case other:
            msg = f"Unknown tool_type in config: {other!r}"
            raise ValueError(msg)


def build_toolsets(configs: list[dict]) -> list[AbstractToolset[Any]]:
    """Build one toolset per distinct config. Duplicates (same JSON) are built once."""
    seen: set[str] = set()
    toolsets: list[AbstractToolset[Any]] = []
    for config in configs:
        key = json.dumps(config, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        toolset = build_toolset(config)
        if toolset is not None:
            toolsets.append(toolset)
    return toolsets


def build_agent_from_state(
    state: AgentState, *, output_type: Any = str, instructions: str | None = None
) -> Agent[None, Any]:
    if state.model is None:
        msg = "Agent has no model configured."
        raise ValueError(msg)
    return build_agent(
        state.model,
        instructions=instructions,
        rulesets=state.rulesets,
        toolsets=build_toolsets(state.tools),
        output_type=output_type,
    )
