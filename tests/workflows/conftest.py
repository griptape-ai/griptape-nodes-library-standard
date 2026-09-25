import json
import os
import shutil
from collections.abc import AsyncGenerator, Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from dotenv.main import DotEnv
from griptape_nodes.bootstrap.workflow_executors.local_workflow_executor import LocalWorkflowExecutor
from griptape_nodes.drivers.cloud_credentials import API_KEY_SECRET_NAME, LICENSE_SECRET_NAME
from griptape_nodes.node_library.library_registry import LibraryRegistry
from griptape_nodes.node_library.workflow_registry import WorkflowRegistry
from griptape_nodes.retained_mode.engine import Engine, reset_root_engine
from griptape_nodes.retained_mode.events.object_events import ClearAllObjectStateRequest
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.retained_mode.managers import config_manager as config_manager_module
from griptape_nodes.retained_mode.managers import secrets_manager as secrets_manager_module
from griptape_nodes.retained_mode.managers.settings import LIBRARIES_TO_REGISTER_KEY

LIBRARY_ROOT = Path(__file__).parents[2]


@pytest.fixture(scope="session")
def cloud_credentials() -> tuple[str, str]:
    """Resolve a Cloud bearer credential without loading unrelated user secrets."""
    user_secrets = DotEnv(secrets_manager_module.ENV_VAR_PATH)
    for secret_name in (LICENSE_SECRET_NAME, API_KEY_SECRET_NAME):
        environment_value = os.environ.get(secret_name)
        if environment_value and environment_value.strip():
            return secret_name, environment_value.strip()

        user_secret_value = user_secrets.get(secret_name)
        if user_secret_value and user_secret_value.strip():
            return secret_name, user_secret_value.strip()

    pytest.fail(
        f"Workflow tests require {LICENSE_SECRET_NAME} or {API_KEY_SECRET_NAME}. Sign in through "
        f"Griptape Nodes Settings or export {API_KEY_SECRET_NAME} before running this suite."
    )


@pytest.fixture(scope="session")
def griptape_nodes(cloud_credentials: tuple[str, str], tmp_path_factory: pytest.TempPathFactory) -> Iterator[Engine]:
    """Initialize the engine against isolated user config and secrets files.

    `GriptapeNodes()` resolves to the current `Engine` rather than building a facade,
    so the fixture is typed as what it actually hands back.
    """
    reset_root_engine()
    LibraryRegistry._clear()
    WorkflowRegistry.clear_user_workflows()
    workspace_path = tmp_path_factory.mktemp("workflow_engine_workspace")
    config_directory = tmp_path_factory.mktemp("workflow_engine_config")
    config_path = config_directory / "griptape_nodes_config.json"
    config_path.write_text(json.dumps({"workspace_directory": str(workspace_path)}, indent=2), encoding="utf-8")
    env_path = config_directory / ".env"
    env_path.write_text("", encoding="utf-8")

    with pytest.MonkeyPatch.context() as monkeypatch:
        for key in list(os.environ):
            if key.startswith(("GT_CLOUD_", "GTN_CONFIG_")) or key == LICENSE_SECRET_NAME:
                monkeypatch.delenv(key)
        monkeypatch.setattr(config_manager_module, "USER_CONFIG_PATH", config_path)
        monkeypatch.setattr(secrets_manager_module, "ENV_VAR_PATH", env_path)
        secret_name, credential = cloud_credentials
        monkeypatch.setenv(secret_name, credential)
        # A configured bucket starts SyncManager's native watchfiles thread, but the engine has
        # no shutdown hook to stop it before interpreter teardown. Blank means "use the default
        # bucket" for public artifact uploads while preventing automatic workflow sync startup.
        monkeypatch.setenv("GT_CLOUD_BUCKET_ID", "")

        try:
            yield GriptapeNodes()
        finally:
            reset_root_engine()
            LibraryRegistry._clear()
            WorkflowRegistry.clear_user_workflows()


@pytest_asyncio.fixture(scope="session")
async def workflow_executor(griptape_nodes: Engine) -> AsyncGenerator[LocalWorkflowExecutor, Any]:
    """Create and manage a single LocalWorkflowExecutor for all tests."""
    async with LocalWorkflowExecutor() as executor:
        yield executor


@pytest.fixture
def isolated_workflow_path(tmp_path: Path) -> Callable[[str | Path], Path]:
    """Copy a workflow's source directory for isolated relative reads and writes."""

    def _copy(workflow_path: str | Path) -> Path:
        source = Path(workflow_path)
        workflow_directory = tmp_path / "workflow"
        shutil.copytree(source.parent, workflow_directory, copy_function=_hardlink_or_copy)
        return workflow_directory / source.name

    return _copy


@pytest_asyncio.fixture(scope="session", autouse=True)
async def setup_test_library(griptape_nodes: Engine) -> AsyncGenerator[None, Any]:
    """Set up this library for testing and restore original state afterwards."""
    config_manager = griptape_nodes.ConfigManager()

    # Save the original libraries state
    original_libraries = config_manager.get_config_value(key=LIBRARIES_TO_REGISTER_KEY, default=[])

    # Set this library for testing, plus the testing library for assertion nodes
    testing_library_root = LIBRARY_ROOT.parent / "griptape-nodes-library-testing"
    config_manager.set_config_value(
        key=LIBRARIES_TO_REGISTER_KEY,
        value=[
            str(LIBRARY_ROOT / "griptape_nodes_library.json"),
            str(testing_library_root / "griptape_nodes_library.json"),
        ],
    )

    try:
        yield  # Run all tests
    finally:
        # Restore original libraries state
        config_manager.set_config_value(key=LIBRARIES_TO_REGISTER_KEY, value=original_libraries)


@pytest_asyncio.fixture(autouse=True)
async def clear_state_before_each_test(griptape_nodes: Engine) -> AsyncGenerator[None, Any]:
    """Clear engine object and workflow registry state around each test."""
    WorkflowRegistry.clear_user_workflows()
    clear_request = ClearAllObjectStateRequest(i_know_what_im_doing=True)
    await griptape_nodes.ahandle_request(clear_request)

    griptape_nodes.ConfigManager()._set_log_level("DEBUG")

    try:
        yield  # Run the test
    finally:
        # Clean up after test
        clear_request = ClearAllObjectStateRequest(i_know_what_im_doing=True)
        await griptape_nodes.ahandle_request(clear_request)
        WorkflowRegistry.clear_user_workflows()


def _hardlink_or_copy(source: str, destination: str) -> str:
    """Hardlink a test file when possible, falling back to a portable copy."""
    try:
        os.link(source, destination)
    except OSError:
        return shutil.copy2(source, destination)
    return destination
