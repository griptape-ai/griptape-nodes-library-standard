from typing import Any, ClassVar

from griptape.artifacts import ImageUrlArtifact
from griptape_nodes.exe_types.core_types import ParameterMode
from griptape_nodes.exe_types.node_types import DataNode
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_types.parameter_image import ParameterImage
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.traits.options import Options

from griptape_nodes_library.image.image_preview import ImagePreviewMixin
from griptape_nodes_library.utils.image_utils import (
    dict_to_image_url_artifact,
    extract_channel_from_image,
    image_to_bytes,
    load_pil_from_url,
)


class DisplayMask(ImagePreviewMixin, DataNode):
    PREVIEW_OPERATION = "display_channel"
    PREVIEW_PARAMETERS: ClassVar[list[str]] = ["channel"]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # Default output parameter name
        self._output_param_name = "output_mask"

        self.add_parameter(
            ParameterImage(
                name="input_image",
                default_value=None,
                tooltip="The image to create a mask from",
                hide_property=True,
                allowed_modes={ParameterMode.INPUT, ParameterMode.OUTPUT},
            )
        )

        self._add_preview_parameter()

        channel_param = ParameterString(
            name="channel",
            tooltip="Channel to extract as mask (red, green, blue, or alpha).",
            default_value="alpha",
            ui_options={"expander": True, "edit_mask": True, "edit_mask_paint_mask": True},
        )
        channel_param.add_trait(Options(choices=["red", "green", "blue", "alpha"]))
        self.add_parameter(channel_param)

        self.add_parameter(
            ParameterImage(
                name=self._output_param_name,
                tooltip="Generated mask image.",
                ui_options={"expander": True},
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

        self._output_file = ProjectFileParameter(node=self, name="output_file", default_filename="display_mask.png")
        self._output_file.add_parameter()

    def _preview_extras(self) -> dict[str, Any]:
        """The image's mode, which decides how the node reads it and which the browser can't tell."""
        return {"image_mode": self._image_mode("input_image")}

    def _set_output_parameter_name(self, name: str) -> None:
        """Set the name of the output parameter. Must be called before adding the output parameter."""
        self._output_param_name = name

    def _get_output_parameter_name(self) -> str:
        """Get the name of the output parameter."""
        return self._output_param_name

    def process(self) -> None:
        """Process the node during execution."""
        input_image = self.get_parameter_value("input_image")
        channel = self.get_parameter_value("channel")

        if input_image is None:
            return

        # Normalize input to ImageUrlArtifact
        if isinstance(input_image, dict):
            input_image = dict_to_image_url_artifact(input_image)

        # Create mask from image
        self._extract_channel(input_image, channel)

    def _extract_channel(self, image_artifact: ImageUrlArtifact, channel: str) -> None:
        """Extract a channel from the input image and set as output."""
        # Load image
        image_pil = load_pil_from_url(image_artifact.value)

        # Extract the specified channel as mask
        mask = extract_channel_from_image(image_pil, channel, "image")

        # Save output mask and create URL artifact
        image_bytes = image_to_bytes(mask, "PNG")
        dest = self._output_file.build_file()
        saved = dest.write_bytes(image_bytes)
        output_artifact = ImageUrlArtifact(saved.location)
        output_param_name = self._get_output_parameter_name()
        self.set_parameter_value(output_param_name, output_artifact)
        self.publish_update_to_parameter(output_param_name, output_artifact)
