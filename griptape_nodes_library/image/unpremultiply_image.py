from typing import ClassVar

from PIL import Image

from griptape_nodes_library.image.base_alpha_conversion import BaseAlphaConversion
from griptape_nodes_library.utils.image_utils import unpremultiply_rgba


class UnpremultiplyImage(BaseAlphaConversion):
    """Recover straight (unweighted) RGB values from a premultiplied image.

    Renders from 3D software or EXR files typically deliver premultiplied output:
    RGB has already been scaled by alpha, so semi-transparent areas appear darker
    than their true colour. Run this node first before colour-grading, keying, or
    compositing with other nodes in this library, which expect straight alpha.

    Equivalent to Nuke's Unpremult node: C_out = C_in / A.
    """

    INPUT_TOOLTIP: ClassVar[str] = "The premultiplied RGBA image to unpremultiply."
    INVERT_TOOLTIP: ClassVar[str] = "Divide by (1 - alpha) instead of alpha."
    OUTPUT_TOOLTIP: ClassVar[str] = "Unpremultiplied (straight alpha) RGBA image."
    DEFAULT_FILENAME: ClassVar[str] = "unpremultiplied.png"
    ACTION: ClassVar[str] = "unpremultiply"

    def _convert(self, image: Image.Image, *, invert: bool) -> Image.Image:
        return unpremultiply_rgba(image, invert=invert)
