"""End-to-end test for bundles produced by the local publisher."""

import ast
import json
import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest
from griptape_nodes.bootstrap.workflow_executors.local_workflow_executor import LocalWorkflowExecutor
from griptape_nodes.drivers.cloud_credentials import API_KEY_SECRET_NAME, LICENSE_SECRET_NAME
from griptape_nodes.retained_mode.events.workflow_events import PublishWorkflowRequest, PublishWorkflowResultSuccess
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

from griptape_nodes_library.publish_workflow.local_publish_options import PUBLISH_OUTPUT_DIRECTORY_FIELD

LIBRARY_ROOT = Path(__file__).parents[2]
UV_SYNC_TIMEOUT_SECONDS = 300
WORKFLOW_RUN_TIMEOUT_SECONDS = 120


@pytest.mark.asyncio
async def test_published_workflow_can_resolve_workflow_metadata(
    workflow_executor: LocalWorkflowExecutor,
    publisher_name: str,
    isolated_workflow_path: Callable[[str | Path], Path],
    bundle_environment: dict[str, str],
    tmp_path: Path,
) -> None:
    """Publish then execute a workflow that asserts on {workflow_name} macro value.

    That is, asserts the workflow's metadata has been read and is available.
    """
    workflow_name = "assert_expected_workflow_name"
    run_command = ["uv", "run", "python", "run.py"]
    workflow_source_path = LIBRARY_ROOT / "tests" / "integration" / f"{workflow_name}.py"
    workflow_path = isolated_workflow_path(workflow_source_path)

    await workflow_executor.aprepare_workflow_for_run(flow_input={}, workflow_path=str(workflow_path))

    publish_output_directory = tmp_path / "published"
    result = await GriptapeNodes.ahandle_request(
        PublishWorkflowRequest(
            workflow_name=workflow_name,
            publisher_name=publisher_name,
            metadata={PUBLISH_OUTPUT_DIRECTORY_FIELD: str(publish_output_directory)},
        )
    )
    assert isinstance(result, PublishWorkflowResultSuccess), f"Publish failed: {result}"

    bundle_directory = publish_output_directory / workflow_name
    _run_in_bundle(
        bundle_directory,
        ["uv", "sync"],
        timeout=UV_SYNC_TIMEOUT_SECONDS,
        environment=bundle_environment,
    )
    completed = _run_in_bundle(
        bundle_directory,
        run_command,
        timeout=WORKFLOW_RUN_TIMEOUT_SECONDS,
        environment=bundle_environment,
    )
    workflow_output = _parse_workflow_output(run_command, completed)
    assertion_output = workflow_output.get("End Flow")
    if not isinstance(assertion_output, dict):
        pytest.fail(_describe(run_command, "produced no End Flow output", completed.stdout, completed.stderr))

    result_details = assertion_output.get("result_details_1")
    assert assertion_output.get("was_successful_1") is True, _describe(
        run_command, f"reported assertion failure: {result_details!r}", completed.stdout, completed.stderr
    )
    assert isinstance(result_details, str), _describe(
        run_command, "produced no assertion result details", completed.stdout, completed.stderr
    )
    assert "Assertion passed:" in result_details
    assert "ends_with 'expected_workflow_name'" in result_details


def _run_in_bundle(
    bundle_directory: Path, command: list[str], timeout: int, environment: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    """Run a command inside a published bundle, failing the test with its output."""
    try:
        completed = subprocess.run(
            command,
            cwd=bundle_directory,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=environment,
        )
    except subprocess.TimeoutExpired as error:
        pytest.fail(_describe(command, f"timed out after {timeout}s", error.stdout, error.stderr))

    assert completed.returncode == 0, _describe(
        command, f"exited with {completed.returncode}", completed.stdout, completed.stderr
    )
    return completed


def _parse_workflow_output(command: list[str], completed: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    """Parse the workflow dictionary from the final non-empty stdout line."""
    output_lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not output_lines:
        pytest.fail(_describe(command, "produced no workflow output", completed.stdout, completed.stderr))

    try:
        workflow_output = ast.literal_eval(output_lines[-1])
    except (SyntaxError, ValueError) as error:
        pytest.fail(
            _describe(command, f"produced invalid workflow output: {error}", completed.stdout, completed.stderr)
        )

    if not isinstance(workflow_output, dict):
        pytest.fail(
            _describe(command, "produced workflow output that was not a dictionary", completed.stdout, completed.stderr)
        )
    return cast("dict[str, Any]", workflow_output)


def _describe(command: list[str], outcome: str, stdout: str | bytes | None, stderr: str | bytes | None) -> str:
    """Format a failed bundle command and whatever output it managed to produce."""
    return (
        f"'{' '.join(command)}' {outcome}\n"
        f"--- stdout ---\n{stdout if stdout is not None else '<none>'}\n"
        f"--- stderr ---\n{stderr if stderr is not None else '<none>'}"
    )


@pytest.fixture(scope="session")
def publisher_name() -> str:
    """Read the publish handler's library name from the manifest."""
    manifest = json.loads((LIBRARY_ROOT / "griptape_nodes_library.json").read_text(encoding="utf-8"))
    return manifest["name"]


@pytest.fixture
def isolated_user_config_home(tmp_path: Path) -> Path:
    """Provide an empty XDG config home for the published bundle subprocess."""
    config_home = tmp_path / "xdg_config"
    config_home.mkdir()
    return config_home


@pytest.fixture
def bundle_environment(isolated_user_config_home: Path) -> dict[str, str]:
    """Return an isolated environment for commands executed in the published bundle.

    The suite itself runs under `uv run`, so an inherited `VIRTUAL_ENV` or
    `UV_PROJECT_ENVIRONMENT` would point the bundle's own `uv sync` at this library's
    venv. The engine resolves its user config path from `XDG_CONFIG_HOME` at import
    time, so the override prevents the fresh Python process from reading or writing the
    developer's config.
    """
    environment = dict(os.environ)
    for key in list(environment):
        if key.startswith(("GT_CLOUD_", "GTN_CONFIG_")):
            environment.pop(key)
    environment.pop("VIRTUAL_ENV", None)
    environment.pop("UV_PROJECT_ENVIRONMENT", None)
    environment[LICENSE_SECRET_NAME] = ""
    environment[API_KEY_SECRET_NAME] = "fake-test-key-for-bootstrap"
    environment["GT_CLOUD_BUCKET_ID"] = ""
    environment["XDG_CONFIG_HOME"] = str(isolated_user_config_home)
    return environment
