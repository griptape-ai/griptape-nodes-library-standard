"""Regression tests for ExtractFrames re-running on every downstream run.

Before each workflow run, the engine unresolves any node with a parameter value that looks like
a workflow variable (``\\{[A-Za-z_]``). The default directory ``{outputs}/frames_v{###}`` matched,
so ExtractFrames re-ran and wrote a new version folder every time a downstream node ran.
"""

from __future__ import annotations

import pytest
from griptape_nodes.exe_types.node_types import NodeResolutionState

from griptape_nodes_library.video.extract_frames import DEFAULT_DIRECTORY, ExtractFrames


def _resolved_node() -> ExtractFrames:
    node = ExtractFrames(name="test_extract_frames")
    node.state = NodeResolutionState.RESOLVED
    return node


@pytest.mark.parametrize(
    "directory",
    [
        DEFAULT_DIRECTORY,
        "{outputs}/shots/frames_v{###}",
        "{outputs}/frames",
    ],
)
def test_macro_directory_stays_resolved_before_workflow_run(directory: str) -> None:
    node = _resolved_node()
    node.set_parameter_value("directory", directory)

    node.validate_before_workflow_run()

    assert node.state == NodeResolutionState.RESOLVED


def test_macro_in_substitutable_parameter_still_unresolves() -> None:
    """Confirms the engine check is active here, so the test above is meaningful."""
    node = _resolved_node()
    node.set_parameter_value("file_prefix", "{outputs}")

    node.validate_before_workflow_run()

    assert node.state == NodeResolutionState.UNRESOLVED
