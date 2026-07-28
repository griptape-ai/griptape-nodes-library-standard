"""Shared fixtures and setup for library unit tests."""

import contextlib
import json
import tempfile
from collections.abc import Generator
from pathlib import Path
from unittest.mock import patch

import griptape_nodes.retained_mode.managers.config_manager as config_manager_module
import griptape_nodes.retained_mode.managers.secrets_manager as secrets_manager_module
import pytest
from griptape_nodes.exe_types.param_components.artifact_url.public_artifact_url_parameter import (
    PublicArtifactUrlParameter,
)
from griptape_nodes.retained_mode.events.library_events import RegisterLibraryFromFileRequest
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.utils.metaclasses import SingletonMeta


@pytest.fixture(autouse=True, scope="session")
def load_library_in_isolated_env() -> Generator[None, None, None]:
    """Register library in an isolated env.

    Mirrors what the engine does at runtime: installs the library's pip dependencies
    into the library's own .venv and adds that venv's site-packages to sys.path.
    Must run before collection so that library dependency imports succeed.

    Use an isolated config environment so that ``SecretsManager.__init__``
    does not load real secrets from ``~/.config/griptape_nodes/.env`` into
    ``os.environ`` during library loading.
    """

    with _isolated_env():
        library_json = str(Path(__file__).parents[2] / "griptape_nodes_library.json")
        GriptapeNodes.handle_request(RegisterLibraryFromFileRequest(file_path=library_json))

        yield


@pytest.fixture(autouse=True, scope="function")
def run_test_in_isolated_env() -> Generator[None, None, None]:
    """Run each test in a fresh config/secrets environment so that tests do
    not pollute each other.
    """

    with _isolated_env():
        yield


@pytest.fixture(autouse=True)
def stub_public_artifact_bucket_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent PublicArtifactUrlParameter from making HTTP calls to list buckets."""
    monkeypatch.setattr(
        PublicArtifactUrlParameter, "_get_bucket_id", staticmethod(lambda *_args, **_kwargs: "test-bucket")
    )


@pytest.fixture
def griptape_nodes() -> GriptapeNodes:
    """Provide a properly initialized GriptapeNodes instance for testing."""
    return GriptapeNodes()


@contextlib.contextmanager
def _isolated_env():
    """Patch config and secrets paths to temp files and clear singletons."""
    SingletonMeta._instances.clear()

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_config_path = Path(temp_dir) / "griptape_nodes_config.json"
            temp_config_path.write_text(json.dumps({}, indent=2))
            temp_env_path = Path(temp_dir) / ".env"
            temp_env_path.write_text("")

            with (
                patch.object(config_manager_module, "USER_CONFIG_PATH", temp_config_path),
                patch.object(secrets_manager_module, "ENV_VAR_PATH", temp_env_path),
            ):
                yield

    finally:
        SingletonMeta._instances.clear()
