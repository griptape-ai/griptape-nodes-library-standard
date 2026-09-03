from collections.abc import Callable
from pathlib import Path

import pytest
from griptape_nodes.bootstrap.workflow_executors.local_workflow_executor import LocalWorkflowExecutor

LIBRARY_ROOT = Path(__file__).parents[2]


def get_workflows() -> list[str]:
    """Get all workflow templates for this library."""
    workflows_dir = LIBRARY_ROOT / "workflows" / "templates"
    return [
        str(f) for f in workflows_dir.iterdir() if f.is_file() and f.suffix == ".py" and not f.name.startswith("__")
    ]


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
