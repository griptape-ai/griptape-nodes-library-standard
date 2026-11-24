from typing import Any

from griptape.artifacts import UrlArtifact
from griptape_nodes.common.macro_parser import MacroSyntaxError, ParsedMacro
from griptape_nodes.exe_types.node_types import SuccessFailureNode
from griptape_nodes.files.file import File
from griptape_nodes.files.path_utils import sanitize_path_string

from griptape_nodes_library.utils.macro_path_utils import resolve_macro_path


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
        cleaned = sanitize_path_string(raw)
        return cleaned if cleaned else None

    def _resolve_one(self, path_str: str) -> str:
        """Resolve a macro or plain path to an absolute path.

        Plain paths are anchored to the workspace directory and remote URLs come back
        unchanged. A string with braces the project can't fill, such as "notes {draft}.txt",
        is a filename rather than a macro and also comes back unchanged.

        Raises:
            FileLoadError: If a path naming project variables fails to resolve.
        """
        try:
            has_variables = bool(ParsedMacro(path_str).get_variables())
        except MacroSyntaxError:
            has_variables = False
        if has_variables:
            return resolve_macro_path(path_str)
        return File(path_str).resolve()
