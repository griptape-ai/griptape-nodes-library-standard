from typing import Any

from griptape.artifacts import ImageUrlArtifact
from griptape_nodes.exe_types.core_types import NodeMessageResult, Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import BaseNode, SuccessFailureNode
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.retained_mode.griptape_nodes import logger
from griptape_nodes.traits.button import Button, ButtonDetailsMessagePayload
from griptape_nodes.traits.options import Options

from griptape_nodes_library.utils.artifact_path_tethering import (
    ArtifactPathTethering,
    ArtifactTetheringConfig,
    default_extract_url_from_artifact_value,
)
from griptape_nodes_library.utils.image_utils import (
    SUPPORTED_IMAGE_EXTENSIONS,
    dict_to_image_url_artifact,
    extract_channel_from_image,
    image_to_bytes,
    load_pil_from_url,
)
from griptape_nodes_library.utils.macro_path_utils import (
    copy_external_file_to_project,
    create_external_file_controls,
    resolve_to_macro_path,
    update_external_file_controls,
)

CHANNEL_NONE = "none"
CHANNEL_RED = "red"
CHANNEL_GREEN = "green"
CHANNEL_BLUE = "blue"
CHANNEL_ALPHA = "alpha"
CHANNEL_OPTIONS = [CHANNEL_NONE, CHANNEL_RED, CHANNEL_GREEN, CHANNEL_BLUE, CHANNEL_ALPHA]


class LoadImage(SuccessFailureNode):
    @staticmethod
    def _extract_url_from_image_value(image_value: Any) -> str | None:
        """Extract URL from image parameter value and strip query parameters."""
        return default_extract_url_from_artifact_value(artifact_value=image_value, artifact_classes=ImageUrlArtifact)

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self._output_file = ProjectFileParameter(node=self, name="output_file", default_filename="mask.png")
        self._output_file.add_parameter()

        # Configuration for artifact tethering
        self._tethering_config = ArtifactTetheringConfig(
            dict_to_artifact_func=dict_to_image_url_artifact,
            extract_url_func=self._extract_url_from_image_value,
            supported_extensions=SUPPORTED_IMAGE_EXTENSIONS,
            default_extension="png",
            url_content_type_prefix="image/",
        )
        self.image_parameter = Parameter(
            name="image",
            input_types=["ImageUrlArtifact", "ImageArtifact", "str"],
            type="ImageUrlArtifact",
            output_type="ImageUrlArtifact",
            default_value=None,
            ui_options={
                "clickable_file_browser": True,
                "expander": True,
                "edit_mask": True,
                "display_name": "Image or Path to Image",
            },
            tooltip="The loaded image.",
        )
        self.add_parameter(self.image_parameter)

        # Use the tethering utility to create the properly configured path parameter
        self.path_parameter = ArtifactPathTethering.create_path_parameter(
            name="path",
            config=self._tethering_config,
            display_name="File Path or URL",
            tooltip="Path to a local image file or URL to an image",
        )
        self.add_parameter(self.path_parameter)

        # Tethering helper: keeps image and path parameters in sync bidirectionally
        # When user uploads a file -> path shows the URL, when user enters path -> image loads that file
        self._tethering = ArtifactPathTethering(
            node=self,
            artifact_parameter=self.image_parameter,
            path_parameter=self.path_parameter,
            config=self._tethering_config,
        )

        channel_param = Parameter(
            name="mask_channel",
            type="str",
            tooltip=f"Channel to extract as mask ({', '.join(CHANNEL_OPTIONS)}).",
            default_value=CHANNEL_NONE,
            ui_options={"hide": True},
            traits={Options(choices=CHANNEL_OPTIONS)},
        )
        self.add_parameter(channel_param)

        self.add_parameter(
            Parameter(
                name="output_mask",
                type="ImageUrlArtifact",
                tooltip="The Mask for the image",
                ui_options={"expander": True, "hide": True},
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

        # Warning and button shown when file is outside the project (initially hidden)
        self._external_warning, self._copy_button = create_external_file_controls(self._on_copy_to_project_clicked)
        self.add_node_element(self._external_warning)
        self.add_parameter(self._copy_button)

        # Add status parameters using the helper method
        self._create_status_parameters(
            result_details_tooltip="Details about the image loading operation result",
            result_details_placeholder="Details on the load attempt will be presented here.",
        )

    def after_incoming_connection(
        self,
        source_node: BaseNode,
        source_parameter: Parameter,
        target_parameter: Parameter,
    ) -> None:
        # Delegate to tethering helper - only artifact parameter can receive connections
        self._tethering.on_incoming_connection(target_parameter)
        if target_parameter == self.image_parameter:
            self._update_image_controls(source_node.get_parameter_value(source_parameter.name))
        return super().after_incoming_connection(source_node, source_parameter, target_parameter)

    def after_incoming_connection_removed(
        self,
        source_node: BaseNode,
        source_parameter: Parameter,
        target_parameter: Parameter,
    ) -> None:
        # Delegate to tethering helper - only artifact parameter can have connections removed
        self._tethering.on_incoming_connection_removed(target_parameter)
        if target_parameter == self.image_parameter:
            self._update_image_controls(self.get_parameter_value("image"))
        return super().after_incoming_connection_removed(source_node, source_parameter, target_parameter)

    def before_value_set(self, parameter: Parameter, value: Any) -> Any:
        # Delegate to tethering helper for dynamic settable control
        return self._tethering.on_before_value_set(parameter, value)

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        # Delegate tethering logic to helper for value synchronization and settable restoration
        self._tethering.on_after_value_set(parameter, value)

        # Handle mask extraction when image or mask_channel changes
        if parameter.name in ["image", "mask_channel"] and value is not None:
            self._extract_mask_if_possible()

        if parameter == self.image_parameter:
            self._update_image_controls(value)
        elif parameter == self.path_parameter and value:
            result = resolve_to_macro_path(value)
            if not result.is_external and result.resolved_path != value:
                self.set_parameter_value("path", result.resolved_path)
            else:
                self._update_image_controls(ImageUrlArtifact(value))

        return super().after_value_set(parameter, value)

    def _update_image_controls(self, value: Any) -> None:
        if isinstance(value, ImageUrlArtifact) and value.value:
            result = resolve_to_macro_path(value.value)  # pyright: ignore[reportAttributeAccessIssue]
            update_external_file_controls(result, self._external_warning, self._copy_button, self.name, "image")
            if not result.is_external and result.resolved_path != value.value:
                resolved = ImageUrlArtifact(result.resolved_path)
                self.parameter_output_values["image"] = resolved
                self.parameter_output_values["path"] = result.resolved_path
                self.publish_update_to_parameter("image", resolved)
                self.publish_update_to_parameter("path", result.resolved_path)
        else:
            self._external_warning.hide = True
            self._copy_button.hide = True

    def process(self) -> None:
        # Reset execution state and result details at the start of each run
        self._clear_execution_status()

        # Clear output values to prevent downstream nodes from getting stale data on errors
        self.parameter_output_values["image"] = None
        self.parameter_output_values["path"] = None
        self.parameter_output_values["output_mask"] = None

        # Get parameter values
        image_artifact = self.get_parameter_value("image")
        path_value = self.get_parameter_value("path")

        # Determine which input source to use
        input_source = None
        if image_artifact is not None:
            input_source = "image parameter"
        elif path_value is not None:
            input_source = "path parameter"

        if input_source is None:
            error_details = "No image or path provided"
            self._set_status_results(was_successful=False, result_details=f"FAILURE: {error_details}")
            self._handle_failure_exception(RuntimeError(error_details))
            return

        try:
            # If we have a path but no image, try to load from path
            if image_artifact is None and path_value is not None:
                image_artifact = self._load_image_from_path(path_value)

            # Validate the image artifact
            if image_artifact is None:
                error_details = f"Failed to load image from {input_source}"
                raise RuntimeError(error_details)  # noqa: TRY301 - Direct raise is clearer than helper function

            # Normalize input to ImageUrlArtifact if needed
            if isinstance(image_artifact, dict):
                image_artifact = dict_to_image_url_artifact(image_artifact)

            # Verify image can be loaded (we know it's not None at this point)
            if isinstance(image_artifact, ImageUrlArtifact):
                self._verify_image_loadable(image_artifact)
                macro_result = resolve_to_macro_path(image_artifact.value)  # pyright: ignore[reportAttributeAccessIssue]
                update_external_file_controls(
                    macro_result, self._external_warning, self._copy_button, self.name, "image"
                )
                if not macro_result.is_external:
                    image_artifact = ImageUrlArtifact(macro_result.resolved_path)
                    path_value = image_artifact.value

            # Set output values on success
            self.parameter_output_values["image"] = image_artifact
            self.parameter_output_values["path"] = path_value

            # Extract mask if image is available
            self._extract_mask_if_possible()

            # Success case
            source_info = f"from {input_source}"
            if hasattr(image_artifact, "value") and image_artifact is not None:
                source_info += f" ({image_artifact.value})"

            success_details = f"Image loaded successfully {source_info}"
            self._set_status_results(was_successful=True, result_details=f"SUCCESS: {success_details}")
            logger.info(f"LoadImage '{self.name}': {success_details}")

        except Exception as e:
            error_details = f"Failed to load image from {input_source}: {e}"
            self._set_status_results(was_successful=False, result_details=f"FAILURE: {error_details}")
            logger.error(f"LoadImage '{self.name}': {error_details}")
            self._handle_failure_exception(e)

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
                artifact_class=ImageUrlArtifact,
                default_filename="image.png",
                node_name=self.name,
                parameter_name="image",
            )
            self.set_parameter_value("image", new_artifact)
            self.publish_update_to_parameter("image", new_artifact)
            self.publish_update_to_parameter("path", macro_path)
            return NodeMessageResult(
                success=True,
                details=f"Copied '{path}' to project: {macro_path}",
                response=button_details,
                altered_workflow_state=True,
            )
        except Exception as e:
            logger.error(f"LoadImage '{self.name}': failed to copy to project: {e}")
            return NodeMessageResult(
                success=False, details=f"Failed to copy '{path}' to project: {e}", response=button_details
            )

    def _load_image_from_path(self, path_value: str | None) -> ImageUrlArtifact | None:
        """Load image artifact from a path value."""
        if not path_value:
            return None

        return ImageUrlArtifact(value=path_value)

    def _verify_image_loadable(self, image_artifact: ImageUrlArtifact) -> None:
        """Verify that the image can actually be loaded."""
        try:
            # Attempt to load the image to verify it's valid
            load_pil_from_url(image_artifact.value)
        except Exception as e:
            msg = f"Image verification failed - cannot load image: {e}"
            raise RuntimeError(msg) from e

    def _extract_mask_if_possible(self) -> None:
        """Extract mask from the loaded image if both image and channel are available."""
        image_artifact = self.get_parameter_value("image")
        mask_channel = self.get_parameter_value("mask_channel")

        if image_artifact is None or mask_channel is None or mask_channel == CHANNEL_NONE:
            self.parameter_output_values["output_mask"] = None
            return

        # Normalize input to ImageUrlArtifact
        if isinstance(image_artifact, dict):
            image_artifact = dict_to_image_url_artifact(image_artifact)

        self._extract_channel_as_mask(image_artifact, mask_channel)

    def _extract_channel_as_mask(self, image_artifact: ImageUrlArtifact, channel: str) -> None:
        """Extract a channel from the input image and set as mask output."""
        try:
            # Load image
            image_pil = load_pil_from_url(image_artifact.value)

            # Extract the specified channel as mask
            mask = extract_channel_from_image(image_pil, channel, "image")

            # Save output mask and create URL artifact
            image_bytes = image_to_bytes(mask, "PNG")
            dest = self._output_file.build_file()
            saved = dest.write_bytes(image_bytes)
            output_artifact = ImageUrlArtifact(saved.location)
            self.set_parameter_value("output_mask", output_artifact)
            self.publish_update_to_parameter("output_mask", output_artifact)

        except Exception as e:
            logger.warning(f"{self.name}: Error extracting mask: {e}")
