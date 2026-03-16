from pathlib import Path

import pytest

from griptape_nodes.bootstrap.workflow_executors.local_workflow_executor import LocalWorkflowExecutor

LIBRARY_ROOT = Path(__file__).parents[2]


def get_integration_workflows() -> list[str]:
    """Get all integration test workflows for this library."""
    workflows_dir = LIBRARY_ROOT / "workflows" / "integration_tests"
    return [
        str(f)
        for f in workflows_dir.iterdir()
        if f.is_file() and f.suffix == ".py" and not f.name.startswith("__")
    ]


@pytest.mark.parametrize("workflow_path", get_integration_workflows())
@pytest.mark.asyncio
async def test_workflow_runs(workflow_path: str, workflow_executor: LocalWorkflowExecutor) -> None:
    """Test that each integration workflow runs without errors."""
    await workflow_executor.arun(workflow_name="main", flow_input={}, workflow_path=workflow_path)
