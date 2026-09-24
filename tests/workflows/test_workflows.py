from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from griptape_nodes.bootstrap.workflow_executors.local_workflow_executor import LocalWorkflowExecutor

LIBRARY_ROOT = Path(__file__).parents[2]

# Templates that cannot run headlessly, mapped to the reason reported by the skip.
# These depend on a node whose input can only come from a live browser widget, so
# there is nothing for CI to feed them and the node fails closed.
INTERACTIVE_WORKFLOWS = {
    "flux_2_-_replace_a_face.py": "Webcam node requires a live capture from the browser widget",
}


def get_workflows() -> list[Any]:
    """Get all workflow templates for this library."""
    workflows_dir = LIBRARY_ROOT / "workflows" / "templates"
    paths = [
        path
        for path in sorted(workflows_dir.iterdir())
        if path.is_file() and path.suffix == ".py" and not path.name.startswith("__")
    ]

    # Fail collection rather than let the map drift out of sync with the directory. A key
    # matching nothing means a template was renamed or deleted and its skip has silently
    # stopped applying, so CI would report the node's own opaque failure instead of the
    # real cause.
    stale = sorted(INTERACTIVE_WORKFLOWS.keys() - {path.name for path in paths})
    if stale:
        msg = (
            f"INTERACTIVE_WORKFLOWS names templates that do not exist in {workflows_dir}: "
            f"{stale}. Update the map to match the directory."
        )
        raise RuntimeError(msg)

    workflows: list[Any] = []
    for path in paths:
        reason = INTERACTIVE_WORKFLOWS.get(path.name)
        marks = [pytest.mark.skip(reason=reason)] if reason else []
        workflows.append(pytest.param(str(path), marks=marks))
    return workflows


@pytest.mark.parametrize("workflow_path", get_workflows())
@pytest.mark.asyncio
async def test_workflow_runs(
    workflow_path: str,
    workflow_executor: LocalWorkflowExecutor,
    isolated_workflow_path: Callable[[str | Path], Path],
) -> None:
    """Simple test to check if the workflow runs without errors."""
    workflow_copy = isolated_workflow_path(workflow_path)
    await workflow_executor.arun(workflow_name="main", flow_input={}, workflow_path=str(workflow_copy))
