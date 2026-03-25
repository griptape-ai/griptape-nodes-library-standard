from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from griptape_nodes.node_library.workflow_registry import WorkflowRegistry
from griptape_nodes.retained_mode.events.flow_events import GetTopLevelFlowRequest, GetTopLevelFlowResultSuccess
from griptape_nodes.retained_mode.events.os_events import WriteFileRequest, WriteFileResultSuccess
from griptape_nodes.retained_mode.events.workflow_events import (
    PublishWorkflowResultFailure,
    PublishWorkflowResultSuccess,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.retained_mode.publishing import WorkflowPackager

from griptape_nodes_library.publish_workflow.local_publish_options import PUBLISH_OUTPUT_DIRECTORY_FIELD

if TYPE_CHECKING:
    from griptape_nodes.retained_mode.events.base_events import ResultPayload

logger = logging.getLogger("local_publisher")


class LocalPublisher:
    def __init__(
        self,
        workflow_name: str,
        metadata: dict | None = None,
        *,
        pickle_control_flow_result: bool = False,
    ) -> None:
        self._workflow_name = workflow_name
        self._metadata: dict = metadata or {}
        self.pickle_control_flow_result = pickle_control_flow_result
        self._packager = WorkflowPackager(workflow_name)

    def publish_workflow(self) -> ResultPayload:
        try:
            self._packager.emit_progress(5.0, "Validating workflow before publish...")
            GriptapeNodes.WorkflowManager().extract_workflow_shape(self._workflow_name)

            destination_path = self._metadata.get(PUBLISH_OUTPUT_DIRECTORY_FIELD)
            if not destination_path or not str(destination_path).strip():
                return PublishWorkflowResultFailure(
                    result_details=(
                        f"Attempted to publish workflow '{self._workflow_name}'. "
                        f"Failed because '{PUBLISH_OUTPUT_DIRECTORY_FIELD}' was not provided."
                    ),
                    exception=ValueError(f"Missing '{PUBLISH_OUTPUT_DIRECTORY_FIELD}' in publish options."),
                )

            workflow = WorkflowRegistry.get_workflow_by_name(self._workflow_name)
            destination = Path(str(destination_path).strip()) / self._workflow_name

            self._packager.emit_progress(5.0, "Preparing output directory...")
            destination.mkdir(parents=True, exist_ok=True)

            # Bundle workflow file, libraries, config, .env, project template, static files,
            # HuggingFace download script, and pyproject.toml.
            self._packager.package_to_folder(destination, workflow)

            workflow_file_name = Path(WorkflowRegistry.get_complete_file_path(workflow.file_path)).name
            dest_workflow_path = destination / workflow_file_name

            self._packager.emit_progress(3.0, "Writing entrypoint script...")
            self._write_entrypoint(destination, workflow_file_name)

            self._packager.emit_progress(4.0, "Writing README...")
            self._write_readme(destination)

            self._save_publish_config(destination_path)

            self._packager.emit_progress(8.0, "Successfully published workflow!")

            return PublishWorkflowResultSuccess(
                published_workflow_file_path=str(dest_workflow_path),
                skip_published_workflow_registration=True,
                result_details=f"Workflow '{self._workflow_name}' published successfully to '{destination}'.",
            )
        except Exception as e:
            details = f"Failed to publish workflow '{self._workflow_name}'. Error: {e}"
            logger.error(details)
            return PublishWorkflowResultFailure(result_details=details, exception=e)

    def _save_publish_config(self, destination_path: str) -> None:
        """Persist the chosen publishing config.

        Persist the chosen output directory on the StartFlow node so the dialog
        pre-populates with the same value on the next publish.
        """
        result = GriptapeNodes.handle_request(GetTopLevelFlowRequest())
        if not isinstance(result, GetTopLevelFlowResultSuccess) or result.flow_name is None:
            return
        control_flow = GriptapeNodes.FlowManager().get_flow_by_name(result.flow_name)
        for node in control_flow.nodes.values():
            if node.__class__.__name__ == "StartFlow":
                node.metadata["publish_config"] = {
                    PUBLISH_OUTPUT_DIRECTORY_FIELD: destination_path,
                }
                return

    def _write_entrypoint(self, destination: Path, workflow_file_name: str) -> None:
        content = f"""\
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Resolve paths relative to this script's location
script_dir = Path(__file__).parent

# Load .env with python-dotenv (handles quoted values correctly)
load_dotenv(script_dir / ".env")

# Set workspace directory to this script's directory
os.environ["GTN_CONFIG_WORKSPACE_DIRECTORY"] = str(script_dir)
os.environ["GTN_ENABLE_WORKSPACE_FILE_WATCHING"] = "false"

# Supply the project file path if not provided by the user
if "--project-file-path" not in sys.argv:
    sys.argv.extend(["--project-file-path", str(script_dir / "project.yml")])

# Forward to the workflow script
sys.argv[0] = str(script_dir / "{workflow_file_name}")
exec(open(sys.argv[0]).read())
"""
        result = GriptapeNodes.handle_request(
            WriteFileRequest(file_path=str(destination / "run.py"), content=content, encoding="utf-8")
        )
        if not isinstance(result, WriteFileResultSuccess):
            msg = f"Failed to write run.py entrypoint to '{destination}'."
            logger.error(msg)
            raise TypeError(msg)

    def _write_readme(self, destination: Path) -> None:
        content = f"""\
# {self._workflow_name}

A standalone [Griptape Nodes](https://github.com/griptape-ai/griptape-nodes) workflow packaged for headless execution.

## Prerequisites

- Python 3.12
- [uv](https://docs.astral.sh/uv/)

## Setup

Install dependencies:

```bash
uv sync
```

## Running

To see available options:

```bash
uv run python run.py --help
```

## Generating requirements.txt

If you need a `requirements.txt` for environments that don't use `uv`:

```bash
uv export --no-hashes --no-dev -o requirements.txt
```
"""
        result = GriptapeNodes.handle_request(
            WriteFileRequest(file_path=str(destination / "README.md"), content=content, encoding="utf-8")
        )
        if not isinstance(result, WriteFileResultSuccess):
            msg = f"Failed to write README.md to '{destination}'."
            logger.error(msg)
            raise TypeError(msg)
