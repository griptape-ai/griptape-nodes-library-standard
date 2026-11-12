"""Publish to Folder remembers its output directory on the workflow's Folder Configuration node.

The engine hands the node's values to the dialog and the publisher as ``configuration`` and writes
back the ``configuration`` the publisher returns. A workflow published before the node existed kept
its directory on the Start Flow node instead; the engine resolves either shape into the same
``configuration`` field (see the engine's own tests for that), so this module only ever has to read
one field.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock

import pytest
from griptape_nodes.exe_types.core_types import ParameterMode
from griptape_nodes.exe_types.node_types import ConfigurationNode
from griptape_nodes.retained_mode.events.config_events import GetConfigValueResultSuccess
from griptape_nodes.retained_mode.events.workflow_events import (
    GetPublishOptionsRequest,
    PublishWorkflowResultSuccess,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

from griptape_nodes_library.publish_workflow import local_publisher as local_publisher_module
from griptape_nodes_library.publish_workflow.folder_configuration import FolderConfiguration
from griptape_nodes_library.publish_workflow.local_publish_options import (
    PUBLISH_OUTPUT_DIRECTORY_FIELD,
    get_local_publish_options,
)
from griptape_nodes_library.publish_workflow.local_publisher import LocalPublisher

WORKSPACE = "/home/user/workspace"


@pytest.fixture(autouse=True)
def workspace_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    """The workspace directory the dialog falls back to when nothing was remembered yet."""
    monkeypatch.setattr(
        GriptapeNodes,
        "handle_request",
        Mock(return_value=GetConfigValueResultSuccess(value=WORKSPACE, result_details="ok")),
    )


def _directory_offered(request: GetPublishOptionsRequest) -> str | None:
    fields = get_local_publish_options(request).fields
    return next(field.default_value for field in fields if field.name == PUBLISH_OUTPUT_DIRECTORY_FIELD)


class TestThePublishDialog:
    def test_opens_on_the_folder_the_engine_remembered(self) -> None:
        offered = _directory_offered(
            GetPublishOptionsRequest(
                workflow_name="wf",
                publisher_name="lib",
                configuration={PUBLISH_OUTPUT_DIRECTORY_FIELD: "/opt/remembered"},
            )
        )

        assert offered == "/opt/remembered"

    def test_a_workflow_never_published_opens_on_the_workspace(self) -> None:
        offered = _directory_offered(GetPublishOptionsRequest(workflow_name="wf", publisher_name="lib"))

        assert offered == WORKSPACE

    def test_the_users_current_choice_wins(self) -> None:
        offered = _directory_offered(
            GetPublishOptionsRequest(
                workflow_name="wf",
                publisher_name="lib",
                current_selections={PUBLISH_OUTPUT_DIRECTORY_FIELD: "/opt/chosen"},
                configuration={PUBLISH_OUTPUT_DIRECTORY_FIELD: "/opt/remembered"},
            )
        )

        assert offered == "/opt/chosen"


class TestThePublisher:
    def test_returns_the_folder_for_the_engine_to_remember(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        workflow = MagicMock(file_path="wf.py")
        monkeypatch.setattr(GriptapeNodes, "WorkflowManager", Mock(return_value=MagicMock()))
        monkeypatch.setattr(
            local_publisher_module.WorkflowRegistry, "get_workflow_by_name", Mock(return_value=workflow)
        )
        monkeypatch.setattr(
            local_publisher_module.WorkflowRegistry, "get_complete_file_path", Mock(return_value="/opt/wf.py")
        )
        publisher = LocalPublisher("wf", metadata={PUBLISH_OUTPUT_DIRECTORY_FIELD: str(tmp_path)})
        publisher._packager = MagicMock()
        monkeypatch.setattr(publisher, "_write_entrypoint", Mock())
        monkeypatch.setattr(publisher, "_write_readme", Mock())

        result = publisher.publish_workflow()

        assert isinstance(result, PublishWorkflowResultSuccess), result.result_details
        assert result.configuration == {PUBLISH_OUTPUT_DIRECTORY_FIELD: str(tmp_path)}
        assert Path(result.published_workflow_file_path) == tmp_path / "wf" / "wf.py"


class TestTheConfigurationNode:
    def test_is_a_configuration_node_holding_the_output_directory_as_a_setting(self) -> None:
        node = FolderConfiguration(name="Folder Configuration")

        parameter: Any = node.get_parameter_by_name(PUBLISH_OUTPUT_DIRECTORY_FIELD)

        assert isinstance(node, ConfigurationNode)
        assert parameter is not None
        assert parameter.allowed_modes == {ParameterMode.PROPERTY}
