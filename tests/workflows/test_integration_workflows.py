from collections.abc import Callable
from pathlib import Path

import pytest
from griptape_nodes.bootstrap.workflow_executors.local_workflow_executor import LocalWorkflowExecutor

from tests.integration.flow_inputs import FLOW_INPUTS


def get_integration_workflows() -> list[tuple[str, dict]]:
    """Get all integration test workflows for this library."""
    workflows_dir = Path(__file__).parents[1] / "integration"
    return [
        (str(f), FLOW_INPUTS.get(f.name, {}))
        for f in workflows_dir.iterdir()
        if f.is_file() and f.suffix == ".py" and f.name.startswith("test_")
    ]


@pytest.mark.parametrize("workflow_path,flow_input", get_integration_workflows())
@pytest.mark.asyncio
async def test_workflow_runs(
    workflow_path: str,
    flow_input: dict,
    workflow_executor: LocalWorkflowExecutor,
    isolated_workflow_path: Callable[[str | Path], Path],
) -> None:
    """Test that each integration workflow runs without errors."""
    workflow_copy = isolated_workflow_path(workflow_path)
    await workflow_executor.arun(workflow_name="main", flow_input=flow_input, workflow_path=str(workflow_copy))
