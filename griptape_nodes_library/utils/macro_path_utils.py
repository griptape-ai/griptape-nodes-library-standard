from dataclasses import dataclass
from pathlib import Path

from griptape_nodes.retained_mode.events.project_events import (
    AttemptMapAbsolutePathToProjectRequest,
    AttemptMapAbsolutePathToProjectResultSuccess,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes, logger


@dataclass
class MacroPathResult:
    resolved_path: str
    is_external: bool  # True if path is outside project or is a URL


def resolve_to_macro_path(path: str, node_name: str) -> MacroPathResult:
    """Attempt to resolve a path to a project macro path.

    If the path exists on disk and is inside the project, returns the macro form.
    If the path exists on disk but is outside the project, or if the path does not
    exist on disk (e.g. a remote URL), returns the original path with is_external=True
    so the caller can prompt the user to copy it into the project.
    """
    if Path(path).exists():
        try:
            result = GriptapeNodes.handle_request(
                AttemptMapAbsolutePathToProjectRequest(absolute_path=Path(path))
            )
            if isinstance(result, AttemptMapAbsolutePathToProjectResultSuccess) and result.mapped_path is not None:
                return MacroPathResult(resolved_path=result.mapped_path, is_external=False)
        except Exception as e:
            logger.debug(f"'{node_name}': failed to map path to macro: {e}")
        # Path exists on disk but is outside the project
        return MacroPathResult(resolved_path=path, is_external=True)

    # Path does not exist on disk — treat as external URL
    return MacroPathResult(resolved_path=path, is_external=True)
