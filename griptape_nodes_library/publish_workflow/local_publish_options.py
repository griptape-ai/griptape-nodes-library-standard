"""Publish-time options provider for the local folder publisher.

Called by the engine when the frontend opens the publish dialog. Returns a
single field for the destination directory. If a previous publish saved a
config under the StartFlow node's metadata, that value pre-populates the
field.
"""

from __future__ import annotations

from pathlib import Path

from griptape_nodes.retained_mode.events.config_events import (
    GetConfigValueRequest,
    GetConfigValueResultSuccess,
)
from griptape_nodes.retained_mode.events.flow_events import GetTopLevelFlowRequest, GetTopLevelFlowResultSuccess
from griptape_nodes.retained_mode.events.workflow_events import (
    GetPublishOptionsRequest,
    GetPublishOptionsResultSuccess,
    PublishFieldType,
    PublishOptionField,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

PUBLISH_OUTPUT_DIRECTORY_FIELD = "publish_output_directory"


def _find_start_flow_node():  # noqa: ANN202
    result = GriptapeNodes.handle_request(GetTopLevelFlowRequest())
    if not isinstance(result, GetTopLevelFlowResultSuccess) or result.flow_name is None:
        return None
    control_flow = GriptapeNodes.FlowManager().get_flow_by_name(result.flow_name)
    for node in control_flow.nodes.values():
        if node.__class__.__name__ == "StartFlow":
            return node
    return None


def _get_workspace_directory() -> str:
    result = GriptapeNodes.handle_request(GetConfigValueRequest(category_and_key="workspace_directory"))
    if isinstance(result, GetConfigValueResultSuccess) and result.value:
        return str(result.value)
    return str(Path.home())


def get_local_publish_options(request: GetPublishOptionsRequest) -> GetPublishOptionsResultSuccess:
    """Build the list of fields for the local folder publish dialog."""
    current: dict = request.current_selections or {}

    if not current:
        start_flow = _find_start_flow_node()
        if start_flow is not None:
            saved = start_flow.metadata.get("publish_config")
            if isinstance(saved, dict):
                current = saved

    default_directory = current.get(PUBLISH_OUTPUT_DIRECTORY_FIELD) or _get_workspace_directory()

    fields = [
        PublishOptionField(
            name=PUBLISH_OUTPUT_DIRECTORY_FIELD,
            label="Output Directory",
            field_type=PublishFieldType.FILE_PICKER,
            tooltip="Directory where the workflow will be published as a self-contained project.",
            default_value=default_directory,
        ),
    ]

    return GetPublishOptionsResultSuccess(
        fields=fields,
        result_details="Local publish options resolved successfully.",
    )
