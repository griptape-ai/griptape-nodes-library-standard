from __future__ import annotations

import importlib.metadata
import json
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from dotenv import set_key
from dotenv.main import DotEnv
from griptape_nodes.node_library.library_registry import LibraryNameAndVersion, LibraryRegistry
from griptape_nodes.node_library.workflow_registry import Workflow, WorkflowRegistry
from griptape_nodes.retained_mode.events.app_events import (
    GetEngineVersionRequest,
    GetEngineVersionResultSuccess,
)
from griptape_nodes.retained_mode.events.base_events import (
    ExecutionEvent,
    ExecutionGriptapeNodeEvent,
)
from griptape_nodes.retained_mode.events.flow_events import GetTopLevelFlowRequest, GetTopLevelFlowResultSuccess
from griptape_nodes.retained_mode.events.os_events import (
    CopyFileRequest,
    CopyFileResultSuccess,
    CopyTreeRequest,
    CopyTreeResultSuccess,
)
from griptape_nodes.retained_mode.events.project_events import (
    GetCurrentProjectRequest,
    GetCurrentProjectResultSuccess,
)
from griptape_nodes.retained_mode.events.secrets_events import (
    GetAllSecretValuesRequest,
    GetAllSecretValuesResultSuccess,
)
from griptape_nodes.retained_mode.events.workflow_events import (
    PublishWorkflowProgressEvent,
    PublishWorkflowResultFailure,
    PublishWorkflowResultSuccess,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

if TYPE_CHECKING:
    from griptape_nodes.retained_mode.events.base_events import ResultPayload

logger = logging.getLogger("local_publisher")

PUBLISH_OUTPUT_DIRECTORY_PARAM = "publish_output_directory"


class LocalPublisher:
    def __init__(
        self,
        workflow_name: str,
        *,
        pickle_control_flow_result: bool = False,
    ) -> None:
        self._workflow_name = workflow_name
        self.pickle_control_flow_result = pickle_control_flow_result
        self._progress: float = 0.0

    def publish_workflow(self) -> ResultPayload:
        try:
            self._emit_progress_event(additional_progress=10.0, message="Validating workflow before publish...")

            # Find the StartFlow node and read destination path
            destination_path = self._get_destination_path()
            if not destination_path:
                return PublishWorkflowResultFailure(
                    result_details=(
                        f"Attempted to publish workflow '{self._workflow_name}'. "
                        "Failed because 'publish_output_directory' is not set on the Start Flow node."
                    ),
                )

            workflow = WorkflowRegistry.get_workflow_by_name(self._workflow_name)

            # Create a subfolder named after the workflow inside the specified directory
            destination = Path(destination_path) / self._workflow_name

            self._emit_progress_event(additional_progress=10.0, message="Preparing output directory...")
            destination.mkdir(parents=True, exist_ok=True)

            # Copy the workflow file
            self._emit_progress_event(additional_progress=15.0, message="Copying workflow file...")
            full_workflow_file_path = WorkflowRegistry.get_complete_file_path(workflow.file_path)
            workflow_file_name = Path(full_workflow_file_path).name
            dest_workflow_path = destination / workflow_file_name
            self._copy_file(full_workflow_file_path, dest_workflow_path)

            # Copy libraries
            self._emit_progress_event(additional_progress=20.0, message="Copying libraries...")
            library_paths = self._copy_libraries(
                node_libraries=workflow.metadata.node_libraries_referenced,
                destination_path=destination / "libraries",
                workflow=workflow,
            )

            # Write config
            self._emit_progress_event(additional_progress=10.0, message="Writing configuration...")
            self._write_config(destination, library_paths)

            # Write project template
            self._emit_progress_event(additional_progress=5.0, message="Writing project template...")
            self._write_project_template(destination)

            # Write .env
            self._emit_progress_event(additional_progress=10.0, message="Writing environment file...")
            self._write_env(destination)

            # Write pyproject.toml
            self._emit_progress_event(additional_progress=5.0, message="Writing pyproject.toml...")
            self._write_pyproject_toml(destination, workflow)

            # Write entrypoint script
            self._emit_progress_event(additional_progress=3.0, message="Writing entrypoint script...")
            self._write_entrypoint(destination, workflow_file_name)

            # Write README
            self._emit_progress_event(additional_progress=4.0, message="Writing README...")
            self._write_readme(destination)

            self._emit_progress_event(additional_progress=8.0, message="Successfully published workflow!")

            return PublishWorkflowResultSuccess(
                published_workflow_file_path=str(dest_workflow_path),
                skip_published_workflow_registration=True,
                result_details=f"Workflow '{self._workflow_name}' published successfully to '{destination}'.",
            )
        except Exception as e:
            details = f"Failed to publish workflow '{self._workflow_name}'. Error: {e}"
            logger.error(details)
            return PublishWorkflowResultFailure(result_details=details)

    # -- Destination path --

    def _get_destination_path(self) -> str | None:
        """Read the publish_output_directory from the StartFlow node."""
        get_top_level_flow_result = GriptapeNodes.handle_request(GetTopLevelFlowRequest())
        if (
            not isinstance(get_top_level_flow_result, GetTopLevelFlowResultSuccess)
            or get_top_level_flow_result.flow_name is None
        ):
            return None

        flow_manager = GriptapeNodes.FlowManager()
        control_flow = flow_manager.get_flow_by_name(get_top_level_flow_result.flow_name)

        for node in control_flow.nodes.values():
            if node.__class__.__name__ == "StartFlow":
                value = node.get_parameter_value(PUBLISH_OUTPUT_DIRECTORY_PARAM)
                if value is not None and str(value).strip():
                    return str(value).strip()
        return None

    # -- File copy utilities --

    @classmethod
    def _copy_file(cls, source_path: str | Path, destination_path: str | Path) -> None:
        copy_file_result = GriptapeNodes.handle_request(
            CopyFileRequest(
                source_path=str(source_path),
                destination_path=str(destination_path),
            )
        )
        if not isinstance(copy_file_result, CopyFileResultSuccess):
            msg = f"Failed to copy file from '{source_path}' to '{destination_path}'."
            logger.error(msg)
            raise TypeError(msg)

    # -- Library copying --

    def _copy_libraries(
        self,
        node_libraries: list[LibraryNameAndVersion],
        destination_path: Path,
        workflow: Workflow,
    ) -> list[str]:
        """Copy library source trees to the destination, returning relative library paths."""
        library_paths: list[str] = []

        for library_ref in node_libraries:
            library = GriptapeNodes.LibraryManager().get_library_info_by_library_name(library_ref.library_name)
            if library is None:
                msg = (
                    f"Attempted to publish workflow '{workflow.metadata.name}'. "
                    f"Failed gathering library info for library '{library_ref.library_name}'."
                )
                logger.error(msg)
                raise ValueError(msg)

            library_data = LibraryRegistry.get_library(library_ref.library_name).get_library_data()

            if library.library_path.endswith(".json"):
                library_path = Path(library.library_path)
                absolute_library_path = library_path.resolve()
                abs_paths = [absolute_library_path]
                for node in library_data.nodes:
                    p = (library_path.parent / Path(node.file_path)).resolve()
                    abs_paths.append(p)
                common_root = Path(os.path.commonpath([str(p) for p in abs_paths]))
                dest = destination_path / common_root.name
                copy_tree_request = CopyTreeRequest(
                    source_path=str(common_root),
                    destination_path=str(dest),
                    ignore_patterns=[".venv", "__pycache__"],
                    dirs_exist_ok=True,
                )
                copy_tree_result = GriptapeNodes.handle_request(copy_tree_request)
                if not isinstance(copy_tree_result, CopyTreeResultSuccess):
                    msg = (
                        f"Failed to copy library files from '{common_root}' to '{dest}' "
                        f"for library '{library_ref.library_name}'."
                    )
                    logger.error(msg)
                    raise TypeError(msg)

                library_path_relative_to_common_root = absolute_library_path.relative_to(common_root)
                relative_path = (Path("libraries") / common_root.name / library_path_relative_to_common_root).as_posix()
                library_paths.append(relative_path)
            else:
                library_paths.append(library.library_path)

        return library_paths

    # -- Config writing --

    def _write_config(self, destination: Path, library_paths: list[str]) -> None:
        config: dict[str, Any] = {
            "workspace_directory": ".",
            "enable_workspace_file_watching": False,
            "app_events": {
                "on_app_initialization_complete": {
                    "workflows_to_register": [],
                    "libraries_to_register": library_paths,
                }
            },
        }
        config_path = destination / "griptape_nodes_config.json"
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)

    # -- Project template writing --

    def _write_project_template(self, destination: Path) -> None:
        """Write the current project template to the bundle directory."""
        current_project_result = GriptapeNodes.handle_request(GetCurrentProjectRequest())
        if not isinstance(current_project_result, GetCurrentProjectResultSuccess):
            logger.warning(
                "Could not retrieve current project template: %s. No project.yml will be written.",
                current_project_result,
            )
            return

        template = current_project_result.project_info.template
        project_yaml = template.to_yaml()
        project_yaml_path = destination / "project.yml"
        project_yaml_path.write_text(project_yaml, encoding="utf-8")

    # -- .env writing --

    def _write_env(self, destination: Path) -> None:
        secrets_manager = GriptapeNodes.SecretsManager()
        env_file_mapping = self._get_merged_env_file_mapping(secrets_manager.workspace_env_path)
        env_file_mapping["GTN_CONFIG_WORKSPACE_DIRECTORY"] = "."
        env_file_mapping["GTN_ENABLE_WORKSPACE_FILE_WATCHING"] = "false"

        env_file_path = destination / ".env"
        self._write_env_file(env_file_path, env_file_mapping)

    @classmethod
    def _get_merged_env_file_mapping(cls, workspace_env_file_path: Path) -> dict[str, Any]:
        env_file_dict: dict[str, Any] = {}
        if workspace_env_file_path.exists():
            env_file = DotEnv(workspace_env_file_path)
            env_file_dict = env_file.dict()

        get_all_secrets_request = GetAllSecretValuesRequest()
        get_all_secrets_result = GriptapeNodes.handle_request(request=get_all_secrets_request)
        if not isinstance(get_all_secrets_result, GetAllSecretValuesResultSuccess):
            msg = "Failed to get all secret values."
            logger.error(msg)
            raise TypeError(msg)

        secret_values = get_all_secrets_result.values
        for secret_name, secret_value in secret_values.items():
            if secret_name not in env_file_dict:
                env_file_dict[secret_name] = secret_value

        return env_file_dict

    @classmethod
    def _write_env_file(cls, env_file_path: Path, env_file_dict: dict[str, Any]) -> None:
        env_file_path.touch(exist_ok=True)
        for key, val in env_file_dict.items():
            set_key(env_file_path, key, str(val))

    # -- pyproject.toml writing --

    @staticmethod
    def _slugify(name: str) -> str:
        """Convert a workflow name to a valid Python package/project name."""
        slug = name.lower()
        slug = re.sub(r"[^a-z0-9]+", "-", slug)
        return slug.strip("-")

    def _collect_dependencies(self, workflow: Workflow) -> list[str]:
        """Collect all pip dependencies for the workflow."""
        engine_version = self._get_engine_version()
        source, commit_id = self._get_install_source()
        if source == "git" and commit_id is not None:
            engine_version = commit_id

        dependencies: list[str] = [
            f"griptape-nodes @ git+https://github.com/griptape-ai/griptape-nodes.git@{engine_version}",
        ]

        for library_ref in workflow.metadata.node_libraries_referenced:
            library_data = LibraryRegistry.get_library(library_ref.library_name).get_library_data()
            if library_data.metadata and library_data.metadata.dependencies:
                pip_deps = library_data.metadata.dependencies.pip_dependencies
                if pip_deps:
                    for dep in pip_deps:
                        if dep not in dependencies:
                            dependencies.append(dep)

        return dependencies

    def _write_pyproject_toml(self, destination: Path, workflow: Workflow) -> None:
        project_name = self._slugify(self._workflow_name)
        dependencies = self._collect_dependencies(workflow)

        deps_toml = ",\n".join(f'    "{dep}"' for dep in dependencies)

        content = f"""\
[project]
name = "{project_name}"
description = "A published Griptape Nodes workflow packaged for headless execution."
readme = "README.md"
version = "0.1.0"
requires-python = ">=3.12.0, <3.13"
dependencies = [
{deps_toml},
]
"""

        pyproject_path = destination / "pyproject.toml"
        with pyproject_path.open("w", encoding="utf-8") as f:
            f.write(content)

    # -- Entrypoint writing --

    def _write_entrypoint(self, destination: Path, workflow_file_name: str) -> None:
        """Write a run.py entrypoint that loads .env via python-dotenv and runs the workflow."""
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

        entrypoint_path = destination / "run.py"
        with entrypoint_path.open("w", encoding="utf-8") as f:
            f.write(content)

    # -- README writing --

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

        readme_path = destination / "README.md"
        with readme_path.open("w", encoding="utf-8") as f:
            f.write(content)

    def _get_engine_version(self) -> str:
        engine_version_request = GetEngineVersionRequest()
        engine_version_result = GriptapeNodes.handle_request(request=engine_version_request)
        if not isinstance(engine_version_result, GetEngineVersionResultSuccess):
            msg = f"Attempted to publish workflow '{self._workflow_name}'. Failed getting the engine version."
            logger.error(msg)
            raise TypeError(msg)
        return f"v{engine_version_result.major}.{engine_version_result.minor}.{engine_version_result.patch}"

    def _find_griptape_nodes_distribution(self) -> importlib.metadata.Distribution | None:
        """Find the griptape_nodes distribution from the current executable's venv.

        Uses sys.executable to derive the venv site-packages path, scoping the
        search to avoid picking up distributions from other venvs that may have
        leaked onto sys.path via dynamic library loading.

        Returns:
            The Distribution object for griptape_nodes, or None if not found.
        """
        import sys

        # Derive venv site-packages from sys.executable (without resolving symlinks,
        # which would follow through to the system Python).
        exe_path = Path(sys.executable)
        venv_root = exe_path.parent.parent
        site_packages = venv_root / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"

        if not site_packages.exists():
            logger.info("Venv site-packages not found at %s, falling back to default lookup", site_packages)
            try:
                return importlib.metadata.distribution("griptape_nodes")
            except importlib.metadata.PackageNotFoundError:
                return None

        logger.info("Searching for griptape_nodes in venv site-packages: %s", site_packages)
        for dist in importlib.metadata.distributions(path=[str(site_packages)]):
            if dist.metadata["Name"] == "griptape-nodes":
                logger.info("Found griptape_nodes at %s", dist.locate_file(""))
                return dist

        logger.info("griptape_nodes not found in venv site-packages, falling back to default lookup")
        try:
            return importlib.metadata.distribution("griptape_nodes")
        except importlib.metadata.PackageNotFoundError:
            return None

    def _get_install_source(self) -> tuple[Literal["git", "file", "pypi"], str | None]:
        """Determines the install source of the Griptape Nodes package.

        Returns:
            tuple: A tuple containing the install source and commit ID (if applicable).
        """
        dist = self._find_griptape_nodes_distribution()
        if dist is None:
            logger.info("Could not find griptape_nodes distribution, assuming pypi install")
            return "pypi", None
        direct_url_text = dist.read_text("direct_url.json")
        logger.info("griptape_nodes direct_url.json: %s", direct_url_text)
        # installing from pypi doesn't have a direct_url.json file
        if direct_url_text is None:
            logger.info("No direct_url.json file found, assuming pypi install")
            return "pypi", None

        direct_url_info = json.loads(direct_url_text)
        url = direct_url_info.get("url")
        logger.info("griptape_nodes install URL: %s", url)
        if url.startswith("file://"):
            try:
                pkg_dir = Path(str(dist.locate_file(""))).resolve()
                logger.info("Package directory: %s", pkg_dir)
                git_root = next(p for p in (pkg_dir, *pkg_dir.parents) if (p / ".git").is_dir())
                logger.info("Git root found: %s", git_root)
                commit = (
                    subprocess.check_output(
                        ["git", "rev-parse", "--short", "HEAD"],  # noqa: S607
                        cwd=git_root,
                        stderr=subprocess.DEVNULL,
                    )
                    .decode()
                    .strip()
                )
            except (StopIteration, subprocess.CalledProcessError) as e:
                logger.info("File URL but no git repo or git command failed: %s", e)
                return "file", None
            else:
                logger.info("Detected git install source at %s (commit %s)", git_root, commit)
                return "git", commit
        if "vcs_info" in direct_url_info:
            commit_id = direct_url_info["vcs_info"].get("commit_id", "")[:7]
            logger.info("Detected vcs_info git install source at %s (commit %s)", url, commit_id)
            return "git", commit_id
        # Fall back to pypi if no other source is found
        logger.info("No install source detected, assuming pypi")
        return "pypi", None

    # -- Progress events --

    def _emit_progress_event(self, additional_progress: float, message: str) -> None:
        self._progress += additional_progress
        self._progress = min(self._progress, 100.0)
        event = ExecutionGriptapeNodeEvent(
            wrapped_event=ExecutionEvent(
                payload=PublishWorkflowProgressEvent(
                    progress=self._progress,
                    message=message,
                )
            )
        )
        GriptapeNodes.EventManager().put_event(event)
