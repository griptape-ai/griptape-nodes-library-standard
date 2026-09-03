"""Tests for the entrypoint script ``LocalPublisher`` writes into a published bundle."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest
from griptape_nodes.retained_mode.events.os_events import (
    WriteFileRequest,
    WriteFileResultFailure,
    WriteFileResultSuccess,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

from griptape_nodes_library.publish_workflow import local_publisher as local_publisher_module
from griptape_nodes_library.publish_workflow.local_publisher import LocalPublisher

if TYPE_CHECKING:
    from griptape_nodes.retained_mode.events.base_events import ResultPayload

WORKFLOW_NAME = "my_workflow"
WORKFLOW_FILE_NAME = "my_workflow.py"

RECORD_FILE_NAME = "recorded.json"
ENTRYPOINT_TIMEOUT_SECONDS = 60

# Stands in for an engine-generated workflow file, recording the namespace it was given.
STUB_WORKFLOW = f"""\
import json
from pathlib import Path

Path({RECORD_FILE_NAME!r}).write_text(json.dumps({{"file": __file__, "name": __name__}}), encoding="utf-8")
"""


@pytest.fixture
def handle_request(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Mock ``GriptapeNodes.handle_request`` so the entrypoint is captured, not written."""

    def _succeed(request: WriteFileRequest) -> ResultPayload:
        return WriteFileResultSuccess(final_file_path=str(request.file_path), bytes_written=0, result_details="stubbed")

    mock = Mock(spec=GriptapeNodes.handle_request, side_effect=_succeed)
    monkeypatch.setattr(GriptapeNodes, "handle_request", mock)
    return mock


@pytest.fixture
def logger(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Mock the module logger so failure-path logging is asserted, not emitted."""
    mock = Mock(spec=local_publisher_module.logger)
    monkeypatch.setattr(local_publisher_module, "logger", mock)
    return mock


class TestWriteEntrypoint:
    """Exercises ``LocalPublisher._write_entrypoint``."""

    def test_generated_entrypoint_is_valid_python(self, tmp_path: Path, handle_request: Mock, logger: Mock) -> None:
        """The templated entrypoint compiles, so a broken template fails here and not on publish."""
        content = self._write_entrypoint(tmp_path, handle_request)

        compile(content, "run.py", "exec")
        logger.error.assert_not_called()

    def test_runs_the_workflow_in_its_own_namespace(self, tmp_path: Path, handle_request: Mock, logger: Mock) -> None:
        """The workflow gets its own ``__file__`` and runs as ``__main__``.

        Its generated ``__main__`` block hands ``__file__`` to ``workflows_to_register``, so
        leaking the entrypoint's own path there leaves the workflow unregistered.
        """
        workflow_path = tmp_path / WORKFLOW_FILE_NAME
        workflow_path.write_text(STUB_WORKFLOW, encoding="utf-8")
        entrypoint_path = tmp_path / "run.py"
        entrypoint_path.write_text(self._write_entrypoint(tmp_path, handle_request), encoding="utf-8")

        completed = subprocess.run(
            [sys.executable, entrypoint_path.name],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=ENTRYPOINT_TIMEOUT_SECONDS,
            check=False,
        )

        assert completed.returncode == 0, f"--- stdout ---\n{completed.stdout}\n--- stderr ---\n{completed.stderr}"
        recorded = json.loads((tmp_path / RECORD_FILE_NAME).read_text(encoding="utf-8"))
        assert Path(recorded["file"]) == workflow_path
        assert recorded["name"] == "__main__"
        logger.error.assert_not_called()

    def test_writes_run_py_into_the_destination(self, tmp_path: Path, handle_request: Mock, logger: Mock) -> None:
        """The entrypoint is written to ``run.py`` at the bundle root as UTF-8."""
        self._write_entrypoint(tmp_path, handle_request)

        request = self._sole_request(handle_request)
        assert request.file_path == str(tmp_path / "run.py")
        assert request.encoding == "utf-8"
        logger.error.assert_not_called()

    def test_raises_when_the_write_fails(self, tmp_path: Path, handle_request: Mock, logger: Mock) -> None:
        """A non-success result from the write request is surfaced as a ``TypeError``."""
        handle_request.side_effect = None
        handle_request.return_value = Mock(spec=WriteFileResultFailure)
        publisher = LocalPublisher(workflow_name=WORKFLOW_NAME)

        with pytest.raises(TypeError, match="Failed to write run.py entrypoint"):
            publisher._write_entrypoint(tmp_path, WORKFLOW_FILE_NAME)  # noqa: SLF001

        logger.error.assert_called_once()

    def _write_entrypoint(self, destination: Path, handle_request: Mock) -> str:
        """Run ``_write_entrypoint`` and return the generated entrypoint source."""
        publisher = LocalPublisher(workflow_name=WORKFLOW_NAME)
        publisher._write_entrypoint(destination, WORKFLOW_FILE_NAME)  # noqa: SLF001
        content = self._sole_request(handle_request).content
        assert isinstance(content, str)
        return content

    def _sole_request(self, handle_request: Mock) -> WriteFileRequest:
        """Return the single ``WriteFileRequest`` the publisher issued."""
        handle_request.assert_called_once()
        (request,) = handle_request.call_args.args
        return request
