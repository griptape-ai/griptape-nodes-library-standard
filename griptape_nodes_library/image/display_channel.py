from griptape.artifacts import ImageUrlArtifact
from griptape_nodes.exe_types.core_types import ParameterMode
from griptape_nodes.exe_types.param_types.parameter_image import ParameterImage

from griptape_nodes_library.image.display_mask import DisplayMask
from griptape_nodes_library.utils.image_utils import (
    extract_channel_from_image,
    image_to_bytes,
    load_pil_from_url,
)


class DisplayChannel(DisplayMask):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # Remove the old output_mask parameter and add a new "output" parameter
        self.remove_parameter_element_by_name("output_mask")

        self.add_parameter(
            ParameterImage(
                name="output",
                tooltip="Generated channel image.",
                ui_options={"expander": True},
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

        # Change default channel to red
        channel_param = self.get_parameter_by_name("channel")
        if channel_param is not None:
            channel_param.set_default_value("red")

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
        self.set_parameter_value("output", output_artifact)
        self.publish_update_to_parameter("output", output_artifact)
