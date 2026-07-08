from typing import Any

from griptape.artifacts import UrlArtifact
from griptape_nodes.common.macro_parser import ParsedMacro
from griptape_nodes.exe_types.node_types import SuccessFailureNode
from griptape_nodes.retained_mode.events.project_events import (
    GetPathForMacroRequest,
    GetPathForMacroResultFailure,
    GetPathForMacroResultSuccess,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes


class BaseResolveMacroPath(SuccessFailureNode):
    """Shared helpers for macro-path resolution nodes."""

    def _extract_path_string(self, value: Any) -> str | None:
        """Extract a plain string from a str, UrlArtifact, or artifact-like object."""
        if value is None:
            return None
        if isinstance(value, str):
            raw = value
        elif isinstance(value, UrlArtifact):
            raw = str(value.value)
        elif hasattr(value, "value"):
            raw = str(value.value)
        else:
            raw = str(value)
        cleaned = GriptapeNodes.OSManager().sanitize_path_string(raw)
        return cleaned if cleaned else None

    def _resolve_one(self, path_str: str) -> str:
        """Resolve a single macro path to an absolute path. Raises on malformed or unresolvable macros."""
        result = GriptapeNodes.handle_request(GetPathForMacroRequest(parsed_macro=ParsedMacro(path_str), variables={}))
        if isinstance(result, GetPathForMacroResultSuccess):
            return str(result.resolved_path)
        if isinstance(result, GetPathForMacroResultFailure):
            msg = f"Could not resolve macro path '{path_str}': {result.failure_reason.value}"
            raise ValueError(msg)
        msg = f"Unexpected result resolving macro path '{path_str}'"
        raise ValueError(msg)
