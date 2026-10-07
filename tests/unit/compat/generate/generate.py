"""Generate `fixtures/main/` by building, running, and saving workflows with the griptape-era library.

Run with main's library and venv (the merge-base, d19c834) checked out at MAIN:

    uv run --project MAIN python tests/unit/compat/generate/generate.py MAIN [spec ...]

Only `GT_CLOUD_API_KEY` is read from ~/.config/griptape_nodes/.env. Third-party
providers run against `provider_stub` with a fake key, so saved values hold no secrets.
Paths are replaced with the tokens `harness.materialize` substitutes back.
"""

import asyncio
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
COMPAT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(COMPAT))

from dotenv import dotenv_values  # noqa: E402
from fixture_tools import rewrite_pickled_strings  # noqa: E402
from griptape_nodes.bootstrap.workflow_executors.local_workflow_executor import LocalWorkflowExecutor  # noqa: E402
from griptape_nodes.node_library.library_registry import LibraryRegistry  # noqa: E402
from griptape_nodes.retained_mode.engine import reset_root_engine  # noqa: E402
from griptape_nodes.retained_mode.events.workflow_events import (  # noqa: E402
    SaveWorkflowRequest,
    SaveWorkflowResultSuccess,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes  # noqa: E402
from griptape_nodes.retained_mode.managers import config_manager as config_manager_module  # noqa: E402
from griptape_nodes.retained_mode.managers import secrets_manager as secrets_manager_module  # noqa: E402
from provider_stub import serve  # noqa: E402
from specs import MEDIA, SPECS  # noqa: E402

OUT = COMPAT / "fixtures" / "main"
MCP_SERVER = COMPAT / "mcp_compat_server.py"
STUB_PORT = 18999
OLLAMA_PORT = 11434
FAKE_KEY = "sk-compat-fake"

BUILDER = """# /// script
# dependencies = []
#
# [tool.griptape-nodes]
# name = "{name}"
# schema_version = "0.20.0"
# engine_version_created_with = "0.103.0"
# node_libraries_referenced = [["Griptape Nodes Library", "0.86.0"]]
# is_griptape_provided = false
# is_template = false
#
# ///

import sys

sys.path.insert(0, {here!r})
from builder import build
from specs import SPECS


async def build_workflow() -> None:
    await build(SPECS[{name!r}], __file__)
"""


def _configure(library_root: Path) -> Path:
    """Point the engine at an isolated config, secrets file, and workspace."""
    stub = f"http://127.0.0.1:{STUB_PORT}/v1"
    workdir = Path(tempfile.mkdtemp(prefix="compat_gen_"))
    workspace = workdir / "workspace"
    workspace.mkdir()
    for key in list(os.environ):
        if key.startswith(("GT_CLOUD_", "GTN_CONFIG_")) or key == "GRIPTAPE_NODES_LICENSE":
            del os.environ[key]
    cloud_key = dotenv_values(Path.home() / ".config/griptape_nodes/.env").get("GT_CLOUD_API_KEY") or ""
    os.environ.update({"GT_CLOUD_BUCKET_ID": "", "OPENAI_BASE_URL": stub, "OPENAI_API_KEY": FAKE_KEY})
    (workdir / ".env").write_text(f"GT_CLOUD_API_KEY={cloud_key}\nOPENAI_API_KEY={FAKE_KEY}\n")
    config = {
        "workspace_directory": str(workspace),
        "app_events": {
            "on_app_initialization_complete": {
                "libraries_to_register": [str(library_root / "griptape_nodes_library.json")]
            }
        },
        "mcp_servers": [{"name": "compat", "transport": "stdio", "command": sys.executable, "args": [str(MCP_SERVER)]}],
        "agent": {
            "providers": [
                {
                    "name": "compat-openai",
                    "type": "openai",
                    "model": "gpt-4.1-mini",
                    "base_url": stub,
                    "api_key_secret_name": "OPENAI_API_KEY",
                }
            ]
        },
    }
    (workdir / "griptape_nodes_config.json").write_text(json.dumps(config, indent=2))
    config_manager_module.USER_CONFIG_PATH = workdir / "griptape_nodes_config.json"
    secrets_manager_module.ENV_VAR_PATH = workdir / ".env"
    return workspace


async def _generate(workspace: Path, names: list[str]) -> dict[str, str]:
    tokens = {
        sys.executable: "__COMPAT_PYTHON__",
        str(MCP_SERVER): "__COMPAT_MCP_SERVER__",
        MEDIA: "__COMPAT_MEDIA__",
        # Logs only, e.g. ffmpeg command lines.
        str(Path(tempfile.gettempdir()).resolve()): "/tmp",
        tempfile.gettempdir(): "/tmp",
        str(Path.home()): "~",
    }
    status: dict[str, str] = {}
    reset_root_engine()
    LibraryRegistry._clear()
    async with LocalWorkflowExecutor() as executor:
        for name in names:
            builder = workspace / f"build_{name}.py"
            builder.write_text(BUILDER.format(name=name, here=str(HERE)))
            try:
                await asyncio.wait_for(executor.arun(flow_input={}, workflow_path=str(builder)), timeout=600)
                status[name] = "ran"
            except Exception as e:  # noqa: BLE001
                # Saved anyway: an unrun workflow still records main's parameter values.
                status[name] = f"run failed: {type(e).__name__}: {e}"
            result = await GriptapeNodes.ahandle_request(SaveWorkflowRequest(file_name=f"compat_{name}"))
            if not isinstance(result, SaveWorkflowResultSuccess):
                status[name] += f" | save failed: {result.result_details}"
            else:
                source = rewrite_pickled_strings(Path(result.file_path).read_text(), tokens)
                (OUT / f"{name}.py").write_text(source)
            print(name, status[name], flush=True)
    return status


def main() -> None:
    library_root = Path(sys.argv[1]).resolve()
    names = sys.argv[2:] or list(SPECS)
    workspace = _configure(library_root)
    logging.basicConfig(filename=workspace.parent / "generate.log", level=logging.INFO)
    serve(STUB_PORT)
    serve(OLLAMA_PORT)
    status_path = OUT / "generation_status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    status.update(asyncio.run(_generate(workspace, names)))
    status_path.write_text(json.dumps(dict(sorted(status.items())), indent=1) + "\n")
    sys.stdout.flush()
    # The engine leaves non-daemon threads running.
    os._exit(0)


if __name__ == "__main__":
    main()
