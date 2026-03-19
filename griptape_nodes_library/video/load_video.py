from typing import Any

from griptape.artifacts.video_url_artifact import VideoUrlArtifact
from griptape_nodes.exe_types.core_types import NodeMessageResult, Parameter
from griptape_nodes.exe_types.node_types import BaseNode, DataNode
from griptape_nodes.retained_mode.griptape_nodes import logger
from griptape_nodes.traits.button import Button, ButtonDetailsMessagePayload

from griptape_nodes_library.utils.artifact_path_tethering import (
    ArtifactPathTethering,
    ArtifactTetheringConfig,
    default_extract_url_from_artifact_value,
)
from griptape_nodes_library.utils.macro_path_utils import (
    copy_external_file_to_project,
    create_external_file_controls,
    resolve_to_macro_path,
    update_external_file_controls,
)
from griptape_nodes_library.utils.video_utils import SUPPORTED_VIDEO_EXTENSIONS, dict_to_video_url_artifact


class LoadVideo(DataNode):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # Configuration for artifact tethering
        self._tethering_config = ArtifactTetheringConfig(
            dict_to_artifact_func=dict_to_video_url_artifact,
            extract_url_func=lambda value: default_extract_url_from_artifact_value(
                artifact_value=value, artifact_classes=VideoUrlArtifact
            ),
            supported_extensions=SUPPORTED_VIDEO_EXTENSIONS,
            default_extension="mp4",
            url_content_type_prefix="video/",
        )

        self.video_parameter = Parameter(
            name="video",
            input_types=["VideoUrlArtifact", "VideoArtifact", "str"],
            type="VideoUrlArtifact",
            output_type="VideoUrlArtifact",
            default_value=None,
            ui_options={
                "clickable_file_browser": True,
                "expander": True,
                "display_name": "Video or Path to Video",
            },
            tooltip="The loaded video.",
        )
        self.add_parameter(self.video_parameter)

        # Use the tethering utility to create the properly configured path parameter
        self.path_parameter = ArtifactPathTethering.create_path_parameter(
            name="path",
            config=self._tethering_config,
            display_name="File Path or URL",
            tooltip="Path to a local video file or URL to a video",
        )
        self.add_parameter(self.path_parameter)

        # Tethering helper: keeps video and path parameters in sync bidirectionally
        # When user uploads a file -> path shows the URL, when user enters path -> video loads that file
        self._tethering = ArtifactPathTethering(
            node=self,
            artifact_parameter=self.video_parameter,
            path_parameter=self.path_parameter,
            config=self._tethering_config,
        )

        # Warning and button shown when file is outside the project (initially hidden)
        self._external_warning, self._copy_button = create_external_file_controls(self._on_copy_to_project_clicked)
        self.add_node_element(self._external_warning)
        self.add_parameter(self._copy_button)

    def after_incoming_connection(
        self,
        source_node: BaseNode,
        source_parameter: Parameter,
        target_parameter: Parameter,
    ) -> None:
        # Delegate to tethering helper - only artifact parameter can receive connections
        self._tethering.on_incoming_connection(target_parameter)
        if target_parameter == self.video_parameter:
            self._update_video_controls(source_node.get_parameter_value(source_parameter.name))
        return super().after_incoming_connection(source_node, source_parameter, target_parameter)

    def after_incoming_connection_removed(
        self,
        source_node: BaseNode,
        source_parameter: Parameter,
        target_parameter: Parameter,
    ) -> None:
        # Delegate to tethering helper - only artifact parameter can have connections removed
        self._tethering.on_incoming_connection_removed(target_parameter)
        if target_parameter == self.video_parameter:
            self._update_video_controls(self.get_parameter_value("video"))
        return super().after_incoming_connection_removed(source_node, source_parameter, target_parameter)

    def before_value_set(self, parameter: Parameter, value: Any) -> Any:
        # Delegate to tethering helper for dynamic settable control
        return self._tethering.on_before_value_set(parameter, value)

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        # Delegate tethering logic to helper for value synchronization and settable restoration
        self._tethering.on_after_value_set(parameter, value)

        if parameter == self.video_parameter:
            self._update_video_controls(value)

        return super().after_value_set(parameter, value)

    def _update_video_controls(self, value: Any) -> None:
        if isinstance(value, VideoUrlArtifact) and value.value:
            result = resolve_to_macro_path(value.value)  # pyright: ignore[reportAttributeAccessIssue]
            update_external_file_controls(result, self._external_warning, self._copy_button, self.name, "video")
            if not result.is_external and result.resolved_path != value.value:
                resolved = VideoUrlArtifact(result.resolved_path)
                self.parameter_output_values["video"] = resolved
                self.parameter_output_values["path"] = result.resolved_path
                self.publish_update_to_parameter("video", resolved)
                self.publish_update_to_parameter("path", result.resolved_path)
        else:
            self._external_warning.hide = True
            self._copy_button.hide = True

    def process(self) -> None:
        # Get parameter values and assign to outputs
        video_artifact = self.get_parameter_value("video")
        path_value = self.get_parameter_value("path")

        if isinstance(video_artifact, VideoUrlArtifact):
            result = resolve_to_macro_path(video_artifact.value)  # pyright: ignore[reportAttributeAccessIssue]
            update_external_file_controls(result, self._external_warning, self._copy_button, self.name, "video")
            if not result.is_external:
                video_artifact = VideoUrlArtifact(result.resolved_path)
                path_value = result.resolved_path

        self.parameter_output_values["video"] = video_artifact
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
                artifact_class=VideoUrlArtifact,
                default_filename="video.mp4",
                node_name=self.name,
                parameter_name="video",
            )
            self.set_parameter_value("video", new_artifact)
            self.publish_update_to_parameter("video", new_artifact)
            self.publish_update_to_parameter("path", macro_path)
            return NodeMessageResult(
                success=True,
                details=f"Copied '{path}' to project: {macro_path}",
                response=button_details,
                altered_workflow_state=True,
            )
        except Exception as e:
            logger.error(f"LoadVideo '{self.name}': failed to copy to project: {e}")
            return NodeMessageResult(
                success=False, details=f"Failed to copy '{path}' to project: {e}", response=button_details
            )
