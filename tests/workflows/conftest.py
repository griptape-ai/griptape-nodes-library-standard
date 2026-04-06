from collections.abc import AsyncGenerator
from typing import Any

import pytest_asyncio
from griptape_nodes.bootstrap.workflow_executors.local_workflow_executor import LocalWorkflowExecutor


@pytest_asyncio.fixture(scope="session")
async def workflow_executor() -> AsyncGenerator[LocalWorkflowExecutor, Any]:
    """Create and manage a single LocalWorkflowExecutor for all tests."""
    async with LocalWorkflowExecutor() as executor:
        yield executor
