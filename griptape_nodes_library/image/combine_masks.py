from __future__ import annotations

from typing import Any

import numpy as np
from griptape.artifacts import ImageArtifact, ImageUrlArtifact
from griptape_nodes.exe_types.core_types import ParameterList, ParameterMode
from griptape_nodes.exe_types.node_types import DataNode
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_types.parameter_image import ParameterImage
from griptape_nodes.image import GriptapeImage
from griptape_nodes.retained_mode.griptape_nodes import logger

from griptape_nodes_library.utils.image_utils import (
    ImageMode,
    get_image_color_info,
    image_to_bytes,
    load_image_from_url,
    open_image_bytes,
    pixels_of,
)

_GRAY_ALPHA_CHANNELS = 2
_RGBA_CHANNELS = 4


class CombineMasks(DataNode):
    """Combine a list of masks into a single consolidated mask (pixel-wise max)."""

    _ALPHA_NEAR_OPAQUE_MIN = 200
    _ALPHA_NEAR_OPAQUE_RANGE = 32

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.add_parameter(
            ParameterList(
                name="masks",
                input_types=[
                    "ImageUrlArtifact",
                    "ImageArtifact",
                    "list",
                    "list[ImageUrlArtifact]",
                    "list[ImageArtifact]",
                ],
                default_value=None,
                tooltip="List of mask images to combine into a single mask (union/max).",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"display_name": "Masks"},
            )
        )

        self.add_parameter(
            ParameterImage(
                name="output_mask",
                default_value=None,
                tooltip="Combined mask image.",
                allowed_modes={ParameterMode.OUTPUT},
                ui_options={"expander": True, "edit_mask": True, "edit_mask_paint_mask": True},
            )
        )

        self._output_file = ProjectFileParameter(node=self, name="output_file", default_filename="combined_mask.png")
        self._output_file.add_parameter()

    def validate_before_node_run(self) -> list[Exception] | None:
        exceptions: list[Exception] = []

        masks = self.get_parameter_list_value("masks")
        if not masks:
            msg = "Connect at least one mask to 'Masks'."
            exceptions.append(ValueError(msg))
            return exceptions

        expected_size: tuple[int, int] | None = None
        for idx, mask_value in enumerate(masks):
            if not isinstance(mask_value, (ImageArtifact, ImageUrlArtifact)):
                msg = f"Mask at index {idx} in 'Masks' must be an image, got {type(mask_value).__name__}."
                exceptions.append(ValueError(msg))
                continue
            try:
                mask = self._load_mask(mask_value)
                size = (mask.width, mask.height)
            except Exception as e:
                msg = f"Could not load the mask at index {idx} in 'Masks': {e}"
                exceptions.append(ValueError(msg))
                continue

            if expected_size is None:
                expected_size = size
                continue

            if size != expected_size:
                msg = f"All masks must be the same size. Expected {expected_size}, got {size} at index {idx}."
                exceptions.append(ValueError(msg))

        return exceptions or None

    def process(self) -> None:
        self.parameter_output_values["output_mask"] = None

        masks = self.get_parameter_list_value("masks")
        if not masks:
            return

        combined: np.ndarray | None = None
        first_mask: GriptapeImage | None = None
        for idx, mask_value in enumerate(masks):
            mask = self._load_mask(mask_value)
            mask_gray = self._mask_to_gray(mask)
            logger.debug(
                f"{self.name}: Loaded mask {idx} mode={get_image_color_info(mask).color_space} "
                f"size={(mask.width, mask.height)}"
            )

            if combined is None or first_mask is None:
                combined = mask_gray
                first_mask = mask
                continue

            # validate sizes again defensively (validate_before_node_run should catch this)
            if mask_gray.shape != combined.shape:
                expected_size = (combined.shape[1], combined.shape[0])
                msg = f"All masks must be the same size. Expected {expected_size}, got {(mask.width, mask.height)} at index {idx}."
                raise ValueError(msg)

            combined = np.maximum(combined, mask_gray)

        if combined is None or first_mask is None:
            return

        # The combined mask keeps the first mask's metadata
        combined_mask = GriptapeImage.from_array(combined, metadata=first_mask.metadata)
        image_bytes = image_to_bytes(combined_mask, "PNG")
        dest = self._output_file.build_file()
        saved = dest.write_bytes(image_bytes)
        output_artifact = ImageUrlArtifact(saved.location)
        self.set_parameter_value("output_mask", output_artifact)
        self.publish_update_to_parameter("output_mask", output_artifact)
        self.parameter_output_values["output_mask"] = output_artifact

    def _load_mask(self, value: Any) -> GriptapeImage:
        """Load a mask image from supported artifact-like inputs."""
        if isinstance(value, ImageUrlArtifact):
            return load_image_from_url(value.value)

        if isinstance(value, ImageArtifact):
            return open_image_bytes(value.to_bytes())

        if hasattr(value, "to_bytes"):
            return open_image_bytes(value.to_bytes())

        msg = f"Unsupported mask type: {type(value)}"
        raise ValueError(msg)

    def _mask_to_gray(self, mask: GriptapeImage) -> np.ndarray:
        """Convert a mask image to a height x width uint8 grayscale array.

        A mask with alpha uses its alpha, unless the alpha is near-uniformly opaque, in which case
        it uses the red (or gray) channel instead. Color masks without alpha use their luma.
        """
        if mask.channels in (_GRAY_ALPHA_CHANNELS, _RGBA_CHANNELS):
            mode = ImageMode.GRAY_ALPHA if mask.channels == _GRAY_ALPHA_CHANNELS else ImageMode.RGBA
            pixels = pixels_of(mask, mode)
            alpha = pixels[:, :, -1]
            alpha_min = int(alpha.min())
            alpha_range = int(alpha.max()) - alpha_min
            if alpha_min >= self._ALPHA_NEAR_OPAQUE_MIN and alpha_range <= self._ALPHA_NEAR_OPAQUE_RANGE:
                return pixels[:, :, 0]
            return alpha
        # Gray stays as it is; color gets its luma
        return pixels_of(mask, ImageMode.GRAY)[:, :, 0]
