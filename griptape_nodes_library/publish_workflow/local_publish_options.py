"""Publish-time options provider for the local folder publisher.

Called by the engine when the frontend opens the publish dialog. Returns a
single field for the destination directory, pre-populated from the workflow's
Folder Configuration node. A workflow published before that node existed kept
its directory on the Start Flow node instead; the engine reads either shape
and hands it over as ``request.configuration``, so this module does not have
to tell them apart.
"""

from __future__ import annotations

from pathlib import Path

from griptape_nodes.retained_mode.events.config_events import (
    GetConfigValueRequest,
    GetConfigValueResultSuccess,
)
from griptape_nodes.retained_mode.events.workflow_events import (
    GetPublishOptionsRequest,
    GetPublishOptionsResultSuccess,
    PublishFieldType,
    PublishOptionField,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

PUBLISH_OUTPUT_DIRECTORY_FIELD = "publish_output_directory"


def _get_workspace_directory() -> str:
    result = GriptapeNodes.handle_request(GetConfigValueRequest(category_and_key="workspace_directory"))
    if isinstance(result, GetConfigValueResultSuccess) and result.value:
        return str(result.value)
    return str(Path.home())


def get_local_publish_options(request: GetPublishOptionsRequest) -> GetPublishOptionsResultSuccess:
    """Build the list of fields for the local folder publish dialog."""
    current: dict = request.current_selections or request.configuration or {}

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
