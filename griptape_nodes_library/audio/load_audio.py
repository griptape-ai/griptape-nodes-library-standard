from typing import Any

from griptape.artifacts.audio_url_artifact import AudioUrlArtifact
from griptape_nodes.exe_types.core_types import NodeMessageResult, Parameter
from griptape_nodes.exe_types.node_types import BaseNode, DataNode, NodeDependencies
from griptape_nodes.exe_types.param_types.parameter_audio import ParameterAudio
from griptape_nodes.retained_mode.griptape_nodes import logger
from griptape_nodes.traits.button import Button, ButtonDetailsMessagePayload

from griptape_nodes_library.utils.artifact_path_tethering import (
    ArtifactPathTethering,
    ArtifactTetheringConfig,
    default_extract_url_from_artifact_value,
)
from griptape_nodes_library.utils.audio_utils import SUPPORTED_AUDIO_EXTENSIONS, dict_to_audio_url_artifact
from griptape_nodes_library.utils.macro_path_utils import (
    copy_external_file_to_project,
    create_external_file_controls,
    resolve_to_macro_path,
    update_external_file_controls,
)


class LoadAudio(DataNode):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # Configuration for artifact tethering
        self._tethering_config = ArtifactTetheringConfig(
            dict_to_artifact_func=dict_to_audio_url_artifact,
            extract_url_func=lambda value: default_extract_url_from_artifact_value(
                artifact_value=value, artifact_classes=AudioUrlArtifact
            ),
            supported_extensions=SUPPORTED_AUDIO_EXTENSIONS,
            default_extension="mp3",
            url_content_type_prefix="audio/",
        )

        self.audio_parameter = ParameterAudio(
            name="audio",
            default_value=None,
            ui_options={
                "clickable_file_browser": True,
                "expander": True,
                "display_name": "Audio or Path to Audio",
            },
            tooltip="The loaded audio.",
        )
        self.add_parameter(self.audio_parameter)

        # Use the tethering utility to create the properly configured path parameter
        self.path_parameter = ArtifactPathTethering.create_path_parameter(
            name="path",
            config=self._tethering_config,
            display_name="File Path or URL",
            tooltip="Path to a local audio file or URL to an audio",
        )
        self.add_parameter(self.path_parameter)

        # Tethering helper: keeps audio and path parameters in sync bidirectionally
        # When user uploads a file -> path shows the URL, when user enters path -> audio loads that file
        self._tethering = ArtifactPathTethering(
            node=self,
            artifact_parameter=self.audio_parameter,
            path_parameter=self.path_parameter,
            config=self._tethering_config,
        )

        self.set_initial_node_size(height=320)

        # Warning and button shown when file is outside the project (initially hidden)
        self._external_warning, self._copy_button = create_external_file_controls(self._on_copy_to_project_clicked)
        self.add_node_element(self._external_warning)
        self.add_parameter(self._copy_button)

    def get_node_dependencies(self) -> NodeDependencies | None:
        deps = super().get_node_dependencies()
        if deps is None:
            deps = NodeDependencies()
        value = self.get_parameter_value("path")
        if not value or not isinstance(value, str):
            return deps

        deps.static_files.add(value)
        return deps

    def after_incoming_connection(
        self,
        source_node: BaseNode,
        source_parameter: Parameter,
        target_parameter: Parameter,
    ) -> None:
        # Delegate to tethering helper - only artifact parameter can receive connections
        self._tethering.on_incoming_connection(target_parameter)
        if target_parameter == self.audio_parameter:
            self._update_audio_controls(source_node.get_parameter_value(source_parameter.name))
        return super().after_incoming_connection(source_node, source_parameter, target_parameter)

    def after_incoming_connection_removed(
        self,
        source_node: BaseNode,
        source_parameter: Parameter,
        target_parameter: Parameter,
    ) -> None:
        # Delegate to tethering helper - only artifact parameter can have connections removed
        self._tethering.on_incoming_connection_removed(target_parameter)
        if target_parameter == self.audio_parameter:
            self._update_audio_controls(self.get_parameter_value("audio"))
        return super().after_incoming_connection_removed(source_node, source_parameter, target_parameter)

    def before_value_set(self, parameter: Parameter, value: Any) -> Any:
        # Delegate to tethering helper for dynamic settable control
        return self._tethering.on_before_value_set(parameter, value)

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        # Delegate tethering logic to helper for value synchronization and settable restoration
        self._tethering.on_after_value_set(parameter, value)

        if parameter == self.audio_parameter:
            self._update_audio_controls(value)
        elif parameter == self.path_parameter and value:
            result = resolve_to_macro_path(value)
            if not result.is_external and result.resolved_path != value:
                self.set_parameter_value("path", result.resolved_path)
            else:
                self._update_audio_controls(AudioUrlArtifact(value))

        return super().after_value_set(parameter, value)

    def _update_audio_controls(self, value: Any) -> None:
        if isinstance(value, AudioUrlArtifact) and value.value:
            result = resolve_to_macro_path(value.value)  # pyright: ignore[reportAttributeAccessIssue]
            update_external_file_controls(result, self._external_warning, self._copy_button, self.name, "audio")
            if not result.is_external and result.resolved_path != value.value:
                resolved = AudioUrlArtifact(result.resolved_path)
                self.parameter_output_values["audio"] = resolved
                self.parameter_output_values["path"] = result.resolved_path
                self.publish_update_to_parameter("audio", resolved)
                self.publish_update_to_parameter("path", result.resolved_path)
        else:
            self._external_warning.hide = True
            self._copy_button.hide = True

    def process(self) -> None:
        # Get parameter values and assign to outputs
        audio_artifact = self.get_parameter_value("audio")
        path_value = self.get_parameter_value("path")

        if isinstance(audio_artifact, AudioUrlArtifact):
            result = resolve_to_macro_path(audio_artifact.value)  # pyright: ignore[reportAttributeAccessIssue]
            update_external_file_controls(result, self._external_warning, self._copy_button, self.name, "audio")
            if not result.is_external:
                audio_artifact = AudioUrlArtifact(result.resolved_path)
                path_value = result.resolved_path

        self.parameter_output_values["audio"] = audio_artifact
        self.parameter_output_values["path"] = path_value

    def _on_copy_to_project_clicked(
        self,
        button: Button,  # noqa: ARG002
        button_details: ButtonDetailsMessagePayload,
    ) -> NodeMessageResult:
        """Copy the external file into the project's inputs folder."""
        path = self.parameter_output_values.get("path")
        if not path:
            return NodeMessageResult(success=False, details="No external path to copy", response=button_details)

        try:
            new_artifact, macro_path = copy_external_file_to_project(
                path=path,
                artifact_class=AudioUrlArtifact,
                default_filename="audio.mp3",
                node_name=self.name,
                parameter_name="audio",
            )
            self.set_parameter_value("audio", new_artifact)
            self.publish_update_to_parameter("audio", new_artifact)
            self.publish_update_to_parameter("path", macro_path)
            return NodeMessageResult(
                success=True,
                details=f"Copied '{path}' to project: {macro_path}",
                response=button_details,
                altered_workflow_state=True,
            )
        except Exception as e:
            logger.error(f"LoadAudio '{self.name}': failed to copy to project: {e}")
            return NodeMessageResult(
                success=False, details=f"Failed to copy '{path}' to project: {e}", response=button_details
            )
