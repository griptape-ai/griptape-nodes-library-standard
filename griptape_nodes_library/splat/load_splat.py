from typing import Any, ClassVar

from griptape_nodes.exe_types.core_types import Parameter
from griptape_nodes.exe_types.node_types import BaseNode, ControlNode

from griptape_nodes_library.splat.parameter_splat import ParameterSplat
from griptape_nodes_library.splat.splat_artifact import SplatUrlArtifact
from griptape_nodes_library.utils.artifact_path_tethering import (
    ArtifactPathTethering,
    ArtifactTetheringConfig,
    default_extract_url_from_artifact_value,
)
from griptape_nodes_library.utils.splat_utils import dict_to_splat_url_artifact


class LoadSplat(ControlNode):
    # Supported splat file extensions
    SUPPORTED_EXTENSIONS: ClassVar[set[str]] = {".spz", ".splat", ".ply"}

    @staticmethod
    def _extract_url_from_splat_value(splat_value: Any) -> str | None:
        """Extract URL from splat parameter value."""
        return default_extract_url_from_artifact_value(artifact_value=splat_value, artifact_classes=SplatUrlArtifact)

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self._tethering_config = ArtifactTetheringConfig(
            dict_to_artifact_func=dict_to_splat_url_artifact,
            extract_url_func=self._extract_url_from_splat_value,
            supported_extensions=self.SUPPORTED_EXTENSIONS,
            default_extension="spz",
            url_content_type_prefix="application/",
        )

        self.splat_parameter = ParameterSplat(
            name="splat",
            default_value=None,
            ui_options={
                "clickable_file_browser": True,
                "expander": True,
                "display_name": "Splat File or path to Splat file",
            },
            tooltip="The loaded Gaussian splat file.",
        )
        self.add_parameter(self.splat_parameter)

        # Use the tethering utility to create the properly configured path parameter
        self.path_parameter = ArtifactPathTethering.create_path_parameter(
            name="path",
            config=self._tethering_config,
            display_name="File Path or URL",
            tooltip="Path to a local splat file or URL to a splat file",
        )
        self.add_parameter(self.path_parameter)

        self._tethering = ArtifactPathTethering(
            node=self,
            artifact_parameter=self.splat_parameter,
            path_parameter=self.path_parameter,
            config=self._tethering_config,
        )

    def after_incoming_connection(
        self,
        source_node: BaseNode,
        source_parameter: Parameter,
        target_parameter: Parameter,
    ) -> None:
        # Delegate to tethering helper - only artifact parameter can receive connections
        self._tethering.on_incoming_connection(target_parameter)
        return super().after_incoming_connection(source_node, source_parameter, target_parameter)

    def after_incoming_connection_removed(
        self,
        source_node: BaseNode,
        source_parameter: Parameter,
        target_parameter: Parameter,
    ) -> None:
        # Delegate to tethering helper - only artifact parameter can have connections removed
        self._tethering.on_incoming_connection_removed(target_parameter)
        return super().after_incoming_connection_removed(source_node, source_parameter, target_parameter)

    def before_value_set(self, parameter: Parameter, value: Any) -> Any:
        # Delegate to tethering helper for dynamic settable control
        return self._tethering.on_before_value_set(parameter, value)

    def after_value_set(
        self,
        parameter: Parameter,
        value: Any,
    ) -> None:
        self._tethering.on_after_value_set(parameter, value)
        return super().after_value_set(parameter, value)

    def process(self) -> None:
        splat = self.get_parameter_value("splat")
        self.parameter_output_values["splat"] = splat
