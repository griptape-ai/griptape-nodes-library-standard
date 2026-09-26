from typing import ClassVar

from PIL import Image

from griptape_nodes_library.image.base_alpha_conversion import BaseAlphaConversion
from griptape_nodes_library.utils.image_utils import premultiply_rgba


class PremultiplyImage(BaseAlphaConversion):
    """Scale an image's RGB channels by its alpha, converting straight alpha to premultiplied.

    Use this when handing images to tools or file pipelines that expect premultiplied
    input, such as a compositing package reading EXRs. Nodes in this library work in
    straight alpha by default, so don't premultiply before them. If you do feed the
    result into Merge Images, set its input_alpha to premultiplied or soft edges will
    come out with a dark fringe.

    Equivalent to Nuke's Premult node: C_out = C_in * A.
    """

    INPUT_TOOLTIP: ClassVar[str] = "The straight-alpha RGBA image to premultiply."
    INVERT_TOOLTIP: ClassVar[str] = "Use (1 - alpha) as the multiplier instead of alpha."
    OUTPUT_TOOLTIP: ClassVar[str] = (
        "Premultiplied RGBA image. Library nodes expect straight alpha; set input_alpha to "
        "premultiplied on Merge Images if you connect it there."
    )
    DEFAULT_FILENAME: ClassVar[str] = "premultiplied.png"
    ACTION: ClassVar[str] = "premultiply"

    def _convert(self, image: Image.Image, *, invert: bool) -> Image.Image:
        return premultiply_rgba(image, invert=invert)
