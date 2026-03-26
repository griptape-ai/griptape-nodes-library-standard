from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterGroup, ParameterMode
from griptape_nodes.exe_types.node_types import StartNode
from griptape_nodes.traits.file_system_picker import FileSystemPicker


class StartFlow(StartNode):
    PUBLISH_OUTPUT_DIRECTORY_PARAM = "publish_output_directory"

    def __init__(
        self,
        name: str,
        metadata: dict[Any, Any] | None = None,
    ) -> None:
        if metadata is None:
            metadata = {}
        metadata["showaddparameter"] = True
        super().__init__(name, metadata)

        with ParameterGroup(name="Publishing Config", collapsed=True) as publishing_config_group:
            Parameter(
                name=self.PUBLISH_OUTPUT_DIRECTORY_PARAM,
                input_types=["str"],
                type="str",
                output_type="str",
                default_value=None,
                tooltip="Directory path where the workflow will be published as a self-contained project.",
                allowed_modes={ParameterMode.PROPERTY},
                traits={FileSystemPicker(allow_directories=True, allow_files=False)},
            )

        self.add_node_element(publishing_config_group)

    def process(self) -> None:
        pass
