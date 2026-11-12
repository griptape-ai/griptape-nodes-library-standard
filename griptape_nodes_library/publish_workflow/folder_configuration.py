"""The Folder publish target's Configuration node.

The engine adds it to a workflow after its first publish to a folder and keeps the chosen output
directory on it, so the next publish dialog opens on the same folder.
"""

from __future__ import annotations

from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import ConfigurationNode

from griptape_nodes_library.publish_workflow.local_publish_options import PUBLISH_OUTPUT_DIRECTORY_FIELD


class FolderConfiguration(ConfigurationNode):
    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)
        self.add_parameter(
            Parameter(
                name=PUBLISH_OUTPUT_DIRECTORY_FIELD,
                tooltip="Folder the workflow was last published to. The next publish starts here.",
                type="str",
                default_value="",
                allowed_modes={ParameterMode.PROPERTY},
                ui_options={"display_name": "Output Directory"},
            )
        )
