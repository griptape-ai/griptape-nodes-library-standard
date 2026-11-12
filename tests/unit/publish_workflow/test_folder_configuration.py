"""Publish to Folder remembers its output directory on the workflow's Folder Configuration node.

The engine hands the node's values to the dialog and the publisher as ``configuration`` and writes
back the ``configuration`` the publisher returns. A workflow published before the node existed kept
its directory in the Start Flow node's metadata; the dialog reads that until the next publish moves
it onto the node.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, Mock

import pytest
from griptape_nodes.exe_types.core_types import ParameterMode
from griptape_nodes.exe_types.node_types import ConfigurationNode, StartNode
from griptape_nodes.retained_mode.events.config_events import GetConfigValueRequest, GetConfigValueResultSuccess
from griptape_nodes.retained_mode.events.flow_events import GetTopLevelFlowRequest, GetTopLevelFlowResultSuccess
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

if TYPE_CHECKING:
    from griptape_nodes.retained_mode.events.base_events import RequestPayload, ResultPayload

WORKSPACE = "/home/user/workspace"


class _LegacyStart(StartNode):
    def process(self) -> None:
        pass


@pytest.fixture
def start_node(monkeypatch: pytest.MonkeyPatch) -> StartNode:
    """A top-level flow holding one Start node, and a workspace directory to fall back to."""
    node = _LegacyStart(name="Start Flow")

    def handle_request(request: RequestPayload) -> ResultPayload:
        if isinstance(request, GetTopLevelFlowRequest):
            return GetTopLevelFlowResultSuccess(flow_name="canvas", result_details="ok")
        if isinstance(request, GetConfigValueRequest):
            return GetConfigValueResultSuccess(value=WORKSPACE, result_details="ok")
        msg = f"Unexpected request in test: {type(request).__name__}"
        raise AssertionError(msg)

    flow = MagicMock()
    flow.nodes = {node.name: node}
    flow_manager = MagicMock()
    flow_manager.get_flow_by_name.return_value = flow
    monkeypatch.setattr(GriptapeNodes, "handle_request", Mock(side_effect=handle_request))
    monkeypatch.setattr(GriptapeNodes, "FlowManager", Mock(return_value=flow_manager))
    return node


def _directory_offered(request: GetPublishOptionsRequest) -> str | None:
    fields = get_local_publish_options(request).fields
    return next(field.default_value for field in fields if field.name == PUBLISH_OUTPUT_DIRECTORY_FIELD)


class TestThePublishDialog:
    def test_opens_on_the_folder_the_configuration_node_remembers(self, start_node: StartNode) -> None:
        start_node.metadata["publish_config"] = {PUBLISH_OUTPUT_DIRECTORY_FIELD: "/opt/older"}

        offered = _directory_offered(
            GetPublishOptionsRequest(
                workflow_name="wf",
                publisher_name="lib",
                configuration={PUBLISH_OUTPUT_DIRECTORY_FIELD: "/opt/remembered"},
            )
        )

        assert offered == "/opt/remembered"

    def test_a_workflow_published_before_the_node_existed_opens_on_its_old_folder(self, start_node: StartNode) -> None:
        start_node.metadata["publish_config"] = {PUBLISH_OUTPUT_DIRECTORY_FIELD: "/opt/older"}

        offered = _directory_offered(GetPublishOptionsRequest(workflow_name="wf", publisher_name="lib"))

        assert offered == "/opt/older"

    @pytest.mark.usefixtures("start_node")
    def test_a_workflow_never_published_opens_on_the_workspace(self) -> None:
        offered = _directory_offered(GetPublishOptionsRequest(workflow_name="wf", publisher_name="lib"))

        assert offered == WORKSPACE

    @pytest.mark.usefixtures("start_node")
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
