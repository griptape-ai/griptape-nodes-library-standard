from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterGroup
from griptape_nodes.exe_types.param_types.parameter_int import ParameterInt
from griptape_nodes.retained_mode.griptape_nodes import logger
from griptape_nodes.traits.slider import Slider
from PIL import Image

from griptape_nodes_library.image.base_image_processor import BaseImageProcessor
from griptape_nodes_library.utils.image_utils import (
    apply_grow_shrink_to_mask,
    dict_to_image_url_artifact,
    load_pil_from_url,
)


class DilateErodeMaskImage(BaseImageProcessor):
    """Grow or shrink bright (white) mask regions: dilate adds pixels at edges, erode removes them."""

    MIN_AMOUNT = -100
    MAX_AMOUNT = 100
    DEFAULT_AMOUNT = 0

    def _setup_custom_parameters(self) -> None:
        """Setup dilate/erode iteration count (3x3 kernel passes, ~1px edge movement per step)."""
        with ParameterGroup(name="mask_morph_settings", ui_options={"collapsed": False}) as morph_group:
            amount_param = ParameterInt(
                name="amount",
                default_value=self.DEFAULT_AMOUNT,
                tooltip=(
                    "Iterations of dilate (positive) or erode (negative) on the mask. "
                    f"Each step is one 3×3 morphological pass (~one pixel at the boundary per step). "
                    f"Range {self.MIN_AMOUNT} to {self.MAX_AMOUNT}. "
                    "Uses alpha when it varies; if alpha is uniform, uses grayscale from RGB (white-on-black masks)."
                ),
            )
            amount_param.add_trait(Slider(min_val=self.MIN_AMOUNT, max_val=self.MAX_AMOUNT))
            self.add_parameter(amount_param)

        self.add_node_element(morph_group)

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        """Process image when input connects or when amount changes."""
        if parameter.name == "input_image" and value is not None:
            self._process_image_immediately(value)
        elif parameter.name == "amount":
            image_value = self.get_parameter_value("input_image")
            if image_value is not None:
                self._process_image_immediately(image_value)
        return super().after_value_set(parameter, value)

    def _process_image_immediately(self, image_value: Any) -> None:
        """Process image immediately for live preview."""
        try:
            if isinstance(image_value, dict):
                image_artifact = dict_to_image_url_artifact(image_value)
            else:
                image_artifact = image_value

            pil_image = load_pil_from_url(image_artifact.value)
            processed_image = self._process_image(pil_image, **self._get_custom_parameters())
            output_artifact = self._save_image_artifact(processed_image, "png")

            self.set_parameter_value("output", output_artifact)
            self.publish_update_to_parameter("output", output_artifact)

        except Exception as e:
            logger.warning(f"{self.name}: Live preview failed: {e}")

    def _get_processing_description(self) -> str:
        """Get description of what this processor does."""
        return "Dilate/erode mask"

    def _process_image(self, pil_image: Image.Image, **kwargs) -> Image.Image:
        """Apply dilate/erode to mask channels (alpha or grayscale)."""
        amount = kwargs.get("amount", self.DEFAULT_AMOUNT)
        if amount is None:
            amount = self.DEFAULT_AMOUNT

        amount_i = int(amount)
        logger.debug(f"{self.name}: Processing image with amount={amount_i}")

        if amount_i == 0:
            return pil_image

        # apply_grow_shrink_to_mask: int(abs(grow_shrink)) = iteration count (3×3 kernel).
        iterations = abs(amount_i)
        if amount_i > 0:
            grow_shrink = -float(iterations)
        else:
            grow_shrink = float(iterations)

        if pil_image.mode == "RGBA":
            return self._morph_rgba_mask(pil_image, grow_shrink)

        if pil_image.mode == "L":
            return apply_grow_shrink_to_mask(pil_image, grow_shrink, self.name)

        if pil_image.mode == "RGB":
            gray = pil_image.convert("L")
            morphed = apply_grow_shrink_to_mask(gray, grow_shrink, self.name)
            return Image.merge("RGB", (morphed, morphed, morphed))

        gray = pil_image.convert("L")
        return apply_grow_shrink_to_mask(gray, grow_shrink, self.name)

    @staticmethod
    def _channel_is_uniform(channel: Image.Image) -> bool:
        """True when the channel has a single value everywhere (e.g. alpha all-opaque)."""
        lo, hi = channel.getextrema()
        return lo == hi

    def _morph_rgba_mask(self, pil_image: Image.Image, grow_shrink: float) -> Image.Image:
        """Morph mask content: alpha when it carries the mask; else luminance from RGB."""
        r, g, b, a = pil_image.split()
        if not self._channel_is_uniform(a):
            a_morphed = apply_grow_shrink_to_mask(a, grow_shrink, self.name)
            return Image.merge("RGBA", (r, g, b, a_morphed))

        # Uniform alpha (e.g. fully opaque): the visible mask is almost always in RGB (white on black).
        # Dilate/erode only the alpha channel would not change the image.
        mask_plane = Image.merge("RGB", (r, g, b)).convert("L")
        morphed = apply_grow_shrink_to_mask(mask_plane, grow_shrink, self.name)
        return Image.merge("RGBA", (morphed, morphed, morphed, a))

    def _validate_custom_parameters(self) -> list[Exception] | None:
        """Validate amount parameter."""
        exceptions = []

        amount = self.get_parameter_value("amount")
        if amount is not None and (int(amount) < self.MIN_AMOUNT or int(amount) > self.MAX_AMOUNT):
            msg = f"{self.name} - Amount must be between {self.MIN_AMOUNT} and {self.MAX_AMOUNT}, got {amount}"
            exceptions.append(ValueError(msg))

        return exceptions if exceptions else None

    def _get_custom_parameters(self) -> dict[str, Any]:
        """Get morph parameters."""
        return {
            "amount": self.get_parameter_value("amount"),
        }

    def _get_output_suffix(self, **kwargs) -> str:
        """Get output filename suffix."""
        amount = kwargs.get("amount", self.DEFAULT_AMOUNT)
        if amount is None:
            amount = self.DEFAULT_AMOUNT
        return f"_morph_{int(amount)}"
