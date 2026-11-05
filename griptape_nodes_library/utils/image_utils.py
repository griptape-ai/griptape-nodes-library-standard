"""Shared image helpers, built on the engine's ``GriptapeImage`` and ``GriptapeImagePipeline``.

Images are ``GriptapeImage`` objects. Pixel work NumPy does better than a pipeline step goes
through ``edited_pixels``, which keeps the image's metadata.
"""

import base64
import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from math import gcd
from typing import Any, NamedTuple
from urllib.error import URLError
from urllib.parse import urlparse

import numpy as np
from griptape.artifacts import ImageArtifact, ImageUrlArtifact
from griptape_nodes.files.file import File, FileLoadError
from griptape_nodes.files.project_file import ProjectFileDestination
from griptape_nodes.image import GriptapeImage, GriptapeImagePipeline, ImageFormat, PixelType, ResizeFilter
from requests.exceptions import RequestException

from griptape_nodes_library.utils.color_utils import NAMED_COLORS

logger = logging.getLogger("griptape_nodes")

# Constants for placeholder images
DEFAULT_PLACEHOLDER_WIDTH = 400
DEFAULT_PLACEHOLDER_HEIGHT = 300
DEFAULT_TIMEOUT = 30

# Supported image file extensions
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff", ".tif"}

# Formats GriptapeImage can write. MPO is only read, as JPEG.
WRITABLE_IMAGE_FORMATS = frozenset(set(ImageFormat) - {ImageFormat.MPO})

MAX_UINT8 = 255

# Bytes from the start of a file checked for an SVG tag
_SVG_SNIFF_BYTES = 100

_GRAY_CHANNELS = 1
_GRAY_ALPHA_CHANNELS = 2
_RGB_CHANNELS = 3
_RGBA_CHANNELS = 4

# Pillow's ITU-R 601 luma weights, in 16-bit fixed point
_LUMA_WEIGHTS = (19595, 38470, 7471)


class ImageMode(StrEnum):
    """An 8-bit channel layout, named as Pillow names its modes."""

    GRAY = "L"
    GRAY_ALPHA = "LA"
    RGB = "RGB"
    RGBA = "RGBA"


def validate_image_format(format_str: str, param_name: str = "format") -> ImageFormat:
    """Return the format ``format_str`` names ("png", "JPG", ...), if images can be written in it.

    Raises:
        ValueError: If the format is unknown or can't be written
    """
    image_format = ImageFormat.from_name(format_str)
    if image_format is None or image_format not in WRITABLE_IMAGE_FORMATS:
        supported = ", ".join(sorted(WRITABLE_IMAGE_FORMATS))
        msg = f"Unsupported {param_name}: '{format_str}'. Supported formats: {supported}"
        raise ValueError(msg)
    return image_format


@dataclass
class ResizedImageResult:
    """Result of resizing an image for a cell."""

    image: GriptapeImage | None
    x_offset: int
    y_offset: int


def parse_hex_color(color: str) -> tuple[int, int, int]:
    """Parse hex color string to RGB tuple."""
    color = color.removeprefix("#")
    return (int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16))


def luma(rgb: np.ndarray) -> np.ndarray:
    """Return the uint8 gray of uint8 RGB pixels, as Pillow's ``convert("L")`` computes it."""
    red, green, blue = (rgb[..., index].astype(np.uint32) for index in range(_RGB_CHANNELS))
    weighted = red * _LUMA_WEIGHTS[0] + green * _LUMA_WEIGHTS[1] + blue * _LUMA_WEIGHTS[2]
    return ((weighted + 0x8000) >> 16).astype(np.uint8)


def converted(image: GriptapeImage, mode: ImageMode) -> GriptapeImage:
    """Return ``image`` as 8-bit ``mode``, keeping its metadata.

    16-bit and float pixels are scaled to 8 bits rather than clipped. One-channel float pixels are
    data such as masks and depth, scaled linearly from 0-1; float color is scene-linear and gets the
    sRGB curve. Gray gets the luma of color, as Pillow computes it.
    """
    pipeline = GriptapeImagePipeline(image)
    if image.channels == _GRAY_CHANNELS and image.pixel_type.is_float:
        pipeline.apply(lambda pixels: np.round(np.clip(pixels, 0.0, 1.0) * MAX_UINT8).astype(np.uint8))
    pipeline.to_uint8()
    match mode:
        case ImageMode.GRAY:
            pipeline.to_rgb().apply(luma)
        case ImageMode.GRAY_ALPHA:
            pipeline.to_rgba().apply(lambda pixels: np.dstack([luma(pixels), pixels[:, :, 3]]))
        case ImageMode.RGB:
            pipeline.to_rgb()
        case ImageMode.RGBA:
            pipeline.to_rgba()
        case _:
            msg = f"Unknown image mode: {mode!r}"
            raise ValueError(msg)
    return pipeline.end()


def pixels_of(image: GriptapeImage, mode: ImageMode) -> np.ndarray:
    """Return ``image``'s pixels as a height x width x channels uint8 array in ``mode``."""
    with converted(image, mode) as result:
        return result.to_array()


def edited_pixels(image: GriptapeImage, mode: ImageMode, function: Callable[[np.ndarray], np.ndarray]) -> GriptapeImage:
    """Convert ``image`` to 8-bit ``mode``, then replace its pixels with ``function``'s result, keeping the metadata.

    ``function`` gets the uint8 array ``pixels_of`` would return and may return any array an image
    can hold: 1-4 channels, any size.
    """
    with converted(image, mode) as source:
        return GriptapeImagePipeline(source).apply(function).end()


def scale_alpha(image: GriptapeImage, opacity: float) -> GriptapeImage:
    """Scale an image's alpha channel by `opacity`, adding one if it has none."""
    if opacity >= 1.0:
        return converted(image, ImageMode.RGBA)

    def scaled(pixels: np.ndarray) -> np.ndarray:
        result = pixels.copy()
        result[:, :, 3] = np.floor(pixels[:, :, 3] * opacity + 0.5).astype(np.uint8)
        return result

    return edited_pixels(image, ImageMode.RGBA, scaled)


def composite_over(base: GriptapeImage, overlay: GriptapeImage, position: tuple[int, int]) -> GriptapeImage:
    """Composite `overlay` over `base` at `position` using Porter-Duff "over", as RGBA.

    Over is associative, so stacking many layers stays correct.
    """
    x, y = position
    with converted(base, ImageMode.RGBA) as rgba:
        return GriptapeImagePipeline(rgba).composite(overlay, x, y).end()


def premultiply_rgba(image: GriptapeImage, *, invert: bool = False) -> GriptapeImage:
    """Multiply an RGBA image's RGB channels by its alpha (straight → premultiplied).

    With invert=True the factor is (1 - alpha) instead, which premultiplies by the
    complement — useful for edge matting where opaque areas should be darkened rather
    than transparent ones.  The alpha channel is left unchanged in both cases.
    """

    def premultiplied(pixels: np.ndarray) -> np.ndarray:
        arr = pixels.astype(np.float32)
        alpha = arr[..., 3:4] / 255.0
        factor = (1.0 - alpha) if invert else alpha
        rgb = np.clip(np.round(arr[..., :3] * factor), 0.0, 255.0)
        return np.concatenate([rgb, arr[..., 3:4]], axis=-1).astype(np.uint8)

    return edited_pixels(image, ImageMode.RGBA, premultiplied)


def unpremultiply_rgba(image: GriptapeImage, *, invert: bool = False) -> GriptapeImage:
    """Divide an RGBA image's RGB channels by its alpha (premultiplied → straight).

    With invert=True the divisor is (1 - alpha) instead.  Pixels where the divisor is
    zero are left black.  The alpha channel is left unchanged in both cases.
    """

    def unpremultiplied(pixels: np.ndarray) -> np.ndarray:
        arr = pixels.astype(np.float32)
        alpha = arr[..., 3:4] / 255.0
        factor = (1.0 - alpha) if invert else alpha
        safe = np.where(factor > 0, factor, 1.0)
        rgb = np.where(factor > 0, arr[..., :3] / safe, 0.0)
        rgb = np.clip(np.round(rgb), 0.0, 255.0)
        return np.concatenate([rgb, arr[..., 3:4]], axis=-1).astype(np.uint8)

    return edited_pixels(image, ImageMode.RGBA, unpremultiplied)


def create_background_image(width: int, height: int, background_color: str, *, transparent_bg: bool) -> GriptapeImage:
    """Create background image with specified color and transparency."""
    if transparent_bg:
        return GriptapeImage.new(width, height, NAMED_COLORS["transparent"])
    return GriptapeImage.new(width, height, parse_hex_color(background_color))


def resize_image_for_cell(
    img: GriptapeImage, cell_width: int, cell_height: int, *, crop_to_fit: bool
) -> ResizedImageResult:
    """Resize image to fit cell and return image with positioning offsets."""
    if crop_to_fit:
        # Validate image dimensions to prevent division by zero
        if img.width <= 0 or img.height <= 0:
            msg = f"Skipping invalid image: {img.width}x{img.height}"
            logger.warning(msg)
            return ResizedImageResult(None, 0, 0)  # Skip invalid images
        # Scale to cover the cell, then crop its center
        scale = max(cell_width / img.width, cell_height / img.height)
        new_width = int(img.width * scale)
        new_height = int(img.height * scale)
        left = (new_width - cell_width) // 2
        top = (new_height - cell_height) // 2
        resized = (
            GriptapeImagePipeline(img)
            .resize(new_width, new_height, ResizeFilter.LANCZOS)
            .crop(left, top, left + cell_width, top + cell_height)
            .end()
        )
        return ResizedImageResult(resized, 0, 0)

    # Scale to fit - maintain aspect ratio within cell bounds, centered
    resized = GriptapeImagePipeline(img).thumbnail(cell_width, cell_height).end()
    x_offset = (cell_width - resized.width) // 2
    y_offset = (cell_height - resized.height) // 2
    return ResizedImageResult(resized, x_offset, y_offset)


def dict_to_image_url_artifact(image_dict: dict, image_format: str | None = None) -> ImageUrlArtifact:
    """Convert a dictionary representation of an image to an ImageUrlArtifact."""
    value = image_dict["value"]
    if image_dict["type"] == "ImageUrlArtifact":
        return ImageUrlArtifact(value)

    # Strip base64 prefix if needed
    if "base64," in value:
        value = value.split("base64,")[1]

    image_bytes = base64.b64decode(value)

    # Infer format from MIME type if not specified
    if image_format is None:
        if "type" in image_dict:
            mime_format = image_dict["type"].split("/")[1] if "/" in image_dict["type"] else None
            image_format = mime_format
        else:
            image_format = "png"

    dest = ProjectFileDestination.from_situation(filename=f"input.{image_format}", situation="copy_external_file")
    saved = dest.write_bytes(image_bytes)
    return ImageUrlArtifact(saved.location)


def resolve_localhost_url_to_path(url: str) -> str:
    """Resolve localhost static file URLs to workspace file paths.

    Converts URLs like http://localhost:8124/workspace/static_files/file.jpg
    to actual workspace file paths like static_files/file.jpg

    Args:
        url: URL string that may be a localhost URL

    Returns:
        Resolved file path relative to workspace, or original string if not a localhost URL
    """
    if not isinstance(url, str):
        return url

    # Check if it's a localhost URL (any port) FIRST, before stripping query params.
    # External URLs may have required query parameters (e.g., Azure SAS tokens,
    # AWS presigned URLs) that must be preserved for authentication.
    if not url.startswith(("http://localhost:", "https://localhost:")):
        return url

    # Only strip query parameters from localhost URLs (cachebuster ?t=...)
    url_without_params = url.split("?")[0] if "?" in url else url

    parsed = urlparse(url_without_params)
    # Extract path after /workspace/
    if "/workspace/" in parsed.path:
        workspace_relative_path = parsed.path.split("/workspace/", 1)[1]
        return workspace_relative_path

    # Localhost URL but not a workspace path, return without query params
    return url_without_params


def _is_svg(image_bytes: bytes) -> bool:
    return "<svg" in image_bytes[:_SVG_SNIFF_BYTES].decode("utf-8", errors="ignore").lower()


def open_image_bytes(image_bytes: bytes, source: str = "image bytes") -> GriptapeImage:
    """Open encoded image bytes, reading the header so a bad image fails here.

    Raises:
        ValueError: If the bytes are SVG or not an image that can be read
    """
    if _is_svg(image_bytes):
        msg = f"SVG files are not supported. Cannot load vector graphics: {source}"
        raise ValueError(msg)
    try:
        image = GriptapeImage.open(image_bytes)
        # The header is read lazily; read it now so an unreadable image fails here
        _ = image.width
    except (OSError, ValueError) as e:
        msg = f"Failed to load image from: {source}\nError: {e}"
        raise ValueError(msg) from e
    return image


def load_image_from_url(url: str) -> GriptapeImage:
    """Load image from URL or local file path via File.

    Note: SVG files are not supported, as vector graphics can't be decoded to pixels.
    TODO: Add SVG support by rasterizing SVG files: https://github.com/griptape-ai/griptape-nodes/issues/3721
    """
    if url.lower().endswith(".svg"):
        msg = f"SVG files are not supported. Cannot load vector graphics: {url}"
        raise ValueError(msg)

    try:
        image_bytes = File(url).read_bytes()
    except FileLoadError as e:
        msg = f"Failed to load image from: {url}\nError: {e}"
        raise ValueError(msg) from e

    return open_image_bytes(image_bytes, url)


def load_image_from_artifact(image: ImageUrlArtifact | ImageArtifact | str, context_name: str) -> GriptapeImage:
    """Load an image from various artifact types.

    Args:
        image: ImageUrlArtifact, ImageArtifact, or str URL
        context_name: Name for log lines (e.g., node name)

    Returns:
        The image

    Raises:
        TypeError: If image type is unsupported
        ValueError: If image loading fails
    """
    if not isinstance(image, ImageUrlArtifact | ImageArtifact | str):
        error_msg = f"Unsupported image type: {type(image).__name__}"
        raise TypeError(error_msg)

    if isinstance(image, ImageUrlArtifact):
        return load_image_from_url(image.value)
    if isinstance(image, ImageArtifact):
        return open_image_bytes(image.value, context_name)
    return load_image_from_url(image)


def image_artifact_from_bytes(image_bytes: bytes, source: str = "image bytes") -> ImageArtifact:
    """Make an ImageArtifact of encoded image bytes, with the format and size from their header.

    Raises:
        ValueError: If the bytes are not an image that can be read
    """
    with open_image_bytes(image_bytes, source) as image:
        image_format = image.format or ImageFormat.PNG
        return ImageArtifact(image_bytes, format=image_format.lower(), width=image.width, height=image.height)


def load_image_from_url_artifact(image_url_artifact: ImageUrlArtifact) -> ImageArtifact:
    """Load an ImageArtifact from an ImageUrlArtifact with proper error handling.

    Args:
        image_url_artifact: The ImageUrlArtifact to load

    Returns:
        ImageArtifact: The loaded image artifact

    Raises:
        ValueError: If image download fails with descriptive error message
    """
    try:
        image_bytes = File(image_url_artifact.value).read_bytes()
    except (URLError, RequestException, ConnectionError, TimeoutError, OSError) as err:
        details = (
            f"Failed to download image at '{image_url_artifact.value}'.\n"
            f"If this workflow was shared from another engine installation, "
            f"that image file will need to be regenerated.\n"
            f"Error: {err}"
        )
        raise ValueError(details) from err

    return image_artifact_from_bytes(image_bytes, image_url_artifact.value)


def extract_channel_from_image(image: GriptapeImage, channel: str, context_name: str = "image") -> GriptapeImage:
    """Extract one channel of an image as an 8-bit gray image.

    Gray images give their gray for every color channel. An image without alpha gives opaque
    alpha.

    Args:
        image: Image to extract channel from
        channel: Channel to extract ("red", "green", "blue", "alpha")
        context_name: Name for error messages (e.g., "mask", "image")

    Returns:
        A one-channel image holding the extracted channel

    Raises:
        ValueError: If the channel is unknown
    """
    has_color = image.channels >= _RGB_CHANNELS
    match channel:
        case "red":
            index = 0
        case "green":
            index = 1 if has_color else 0
        case "blue":
            index = 2 if has_color else 0
        case "alpha":
            index = -1
        case _:
            msg = f"Unknown {context_name} channel: {channel!r}"
            raise ValueError(msg)
    mode = ImageMode.RGBA if has_color else ImageMode.GRAY_ALPHA
    return edited_pixels(image, mode, lambda pixels: pixels[:, :, index].copy())


def create_placeholder_image(width: int, height: int, background_color: str, *, transparent_bg: bool) -> GriptapeImage:
    """Create a placeholder image with specified dimensions and background."""
    return create_background_image(width, height, background_color, transparent_bg=transparent_bg)


def create_default_placeholder(
    background_color: str,
    width: int = DEFAULT_PLACEHOLDER_WIDTH,
    height: int = DEFAULT_PLACEHOLDER_HEIGHT,
    *,
    transparent_bg: bool,
) -> GriptapeImage:
    """Create a default placeholder image with specified or standard dimensions."""
    return create_placeholder_image(width, height, background_color, transparent_bg=transparent_bg)


def image_to_bytes(
    image: GriptapeImage,
    output_format: str,
    *,
    quality: int | None = None,
    lossless: bool = False,
    optimize: bool = False,
) -> bytes:
    """Encode an image in the named format ("png", "JPEG", ...), keeping the metadata that format can hold.

    Raises:
        ValueError: If the format can't be written or the image fails to encode
    """
    image_format = validate_image_format(output_format, "output_format")
    return image.encode(image_format, quality=quality, lossless=lossless, optimize=optimize)


def extract_image_url(image_item: Any) -> str:
    """Extract URL from various image input types.

    Returns ``""`` when there is no image, so callers can treat a falsy result as
    "no image was set". ``None`` in particular must not reach ``str(image_item)``
    below: that produces the truthy string ``"None"``, which slips past an
    ``if not url`` guard and only fails much later with an opaque AttributeError.
    """
    if image_item is None:
        return ""
    if isinstance(image_item, ImageUrlArtifact):
        return image_item.value
    if isinstance(image_item, dict) and "value" in image_item:
        return image_item["value"] or ""
    if isinstance(image_item, str):
        return image_item
    # Try to load from URL if it's a string
    return str(image_item)


def load_images_from_list(images: list) -> list[GriptapeImage]:
    """Load images from a list of image items, skipping invalid ones."""
    loaded = []
    for img_item in images:
        try:
            url = extract_image_url(img_item)
            image = load_image_from_url(url)
            # Validate image dimensions
            if image.width <= 0 or image.height <= 0:
                msg = f"Skipping image with invalid dimensions: {image.width}x{image.height}"
                logger.warning(msg)
                continue
            loaded.append(image)
        except (URLError, RequestException, ConnectionError, TimeoutError, OSError) as e:
            # Skip invalid images
            msg = f"Skipping invalid image: {e}"
            logger.warning(msg)
            continue
    return loaded


def create_grid_layout(  # noqa: PLR0913
    images: list[str],
    columns: int,
    output_image_width: int,
    spacing: int,
    background_color: str,
    border_radius: int,
    *,
    crop_to_fit: bool,
    transparent_bg: bool,
    justification: str = "left",
) -> GriptapeImage:
    """Create a uniform grid layout of images."""
    if not images:
        return create_default_placeholder(
            background_color, transparent_bg=transparent_bg, width=output_image_width, height=output_image_width
        )

    # Load and process images
    loaded = load_images_from_list(images)

    if not loaded:
        return create_default_placeholder(background_color, transparent_bg=transparent_bg)

    # Calculate grid dimensions
    if columns <= 0:
        return create_default_placeholder(background_color, transparent_bg=transparent_bg)

    rows = (len(loaded) + columns - 1) // columns
    cell_width = (output_image_width - spacing * (columns + 1)) // columns
    cell_height = cell_width  # Square cells for grid layout

    # Create background - use output_image_width to ensure consistent sizing
    total_width = output_image_width
    total_height = cell_height * rows + spacing * (rows + 1)
    background = create_background_image(total_width, total_height, background_color, transparent_bg=transparent_bg)
    grid = GriptapeImagePipeline(background)

    # Place images in grid
    for idx, img in enumerate(loaded):
        row = idx // columns
        col = idx % columns

        # Resize image to fit cell
        resized_result = resize_image_for_cell(img, cell_width, cell_height, crop_to_fit=crop_to_fit)
        if resized_result.image is None:
            continue  # Skip invalid images

        cell_image = resized_result.image
        if border_radius > 0:
            cell_image = apply_border_radius(cell_image, border_radius)

        # Base x position without justification
        base_x = col * (cell_width + spacing) + spacing + resized_result.x_offset

        # Shift a short last row to honor the justification
        images_in_row = min(columns, len(loaded) - row * columns)
        row_width = images_in_row * (cell_width + spacing) + spacing
        match justification:
            case "center" if images_in_row < columns:
                final_x = base_x + (total_width - row_width) // 2
            case "right" if images_in_row < columns:
                final_x = base_x + total_width - row_width
            case _:
                final_x = base_x

        final_y = row * (cell_height + spacing) + spacing + resized_result.y_offset
        grid.composite(cell_image, final_x, final_y)

    return grid.end()


def create_masonry_layout(  # noqa: PLR0913
    images: list,
    columns: int,
    output_image_width: int,
    spacing: int,
    background_color: str,
    border_radius: int,
    *,
    transparent_bg: bool,
) -> GriptapeImage:
    """Create a masonry layout with variable height columns."""
    if not images:
        return create_default_placeholder(background_color, transparent_bg=transparent_bg)

    # Load and process images
    loaded = load_images_from_list(images)

    if not loaded:
        return create_default_placeholder(background_color, transparent_bg=transparent_bg)

    # Calculate column width
    if columns <= 0:
        return create_default_placeholder(background_color, transparent_bg=transparent_bg)

    column_width = (output_image_width - spacing * (columns + 1)) // columns

    # Distribute images across columns
    columns_content: list[list[GriptapeImage]] = [[] for _ in range(columns)]
    column_heights = [0] * columns

    for img in loaded:
        # Find shortest column
        shortest_col = column_heights.index(min(column_heights))
        columns_content[shortest_col].append(img)

        # Calculate height for this image
        if img.width <= 0 or img.height <= 0:
            continue  # Skip invalid images
        aspect_ratio = img.width / img.height
        img_height = int(column_width / aspect_ratio)
        column_heights[shortest_col] += img_height + spacing

    # Create background
    total_height = max(column_heights) + spacing
    background = create_background_image(
        output_image_width, total_height, background_color, transparent_bg=transparent_bg
    )
    grid = GriptapeImagePipeline(background)

    # Place images in columns
    for col_idx, column_images in enumerate(columns_content):
        x_offset = col_idx * (column_width + spacing) + spacing
        y_offset = spacing

        for img in column_images:
            if img.width <= 0 or img.height <= 0:
                msg = f"Skipping image with invalid dimensions: {img.width}x{img.height}"
                logger.warning(msg)
                continue  # Skip invalid images
            aspect_ratio = img.width / img.height
            img_height = int(column_width / aspect_ratio)
            img_resized = GriptapeImagePipeline(img).resize(column_width, img_height, ResizeFilter.LANCZOS).end()

            # Apply border radius if specified
            if border_radius > 0:
                img_resized = apply_border_radius(img_resized, border_radius)

            grid.composite(img_resized, x_offset, y_offset)
            y_offset += img_height + spacing

    return grid.end()


def _rounded_rectangle_coverage(width: int, height: int, radius: int) -> np.ndarray:
    """Return per-pixel coverage (0-1) of a ``width`` x ``height`` rectangle with corners of ``radius``."""
    radius = min(radius, min(width, height) // 2)
    # Pixel centers, in a space where the rectangle spans 0 to width and 0 to height
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float64) + 0.5
    # Distance from each pixel center to the rectangle inset by radius; inside it is 0
    nearest_x = np.clip(xs, radius, width - radius)
    nearest_y = np.clip(ys, radius, height - radius)
    distance = np.hypot(xs - nearest_x, ys - nearest_y)
    return np.clip(radius + 0.5 - distance, 0.0, 1.0)


def apply_border_radius(image: GriptapeImage, radius: int) -> GriptapeImage:
    """Make an image transparent outside a rectangle with corners of ``radius`` pixels, as RGBA.

    Alpha the image already has is kept inside the corners.
    """
    if radius <= 0:
        return image

    def rounded(pixels: np.ndarray) -> np.ndarray:
        height, width = pixels.shape[:2]
        coverage = _rounded_rectangle_coverage(width, height, radius)
        result = pixels.copy()
        result[:, :, 3] = np.round(pixels[:, :, 3] * coverage).astype(np.uint8)
        return result

    return edited_pixels(image, ImageMode.RGBA, rounded)


def cleanup_temp_files() -> None:
    """Clean up temporary files (placeholder for compatibility)."""
    # This is a placeholder function for compatibility
    # In this implementation, we don't need to clean up temp files
    # as we're using the static file manager


class ImageColorInfo(NamedTuple):
    """Color information for an image."""

    color_space: str
    channels: int


def calculate_aspect_ratio(width: int, height: int) -> tuple[int, int] | None:
    """Calculate GCD-reduced aspect ratio from width and height.

    Args:
        width: Image width in pixels
        height: Image height in pixels

    Returns:
        Tuple of (ratio_width, ratio_height) or None if invalid dimensions
    """

    if width == 0 or height == 0:
        return (0, 0)

    if width < 0 or height < 0:
        return None

    divisor = gcd(width, height)
    ratio_width = width // divisor
    ratio_height = height // divisor

    return (ratio_width, ratio_height)


_MODE_NAMES = {
    _GRAY_CHANNELS: ImageMode.GRAY,
    _GRAY_ALPHA_CHANNELS: ImageMode.GRAY_ALPHA,
    _RGB_CHANNELS: ImageMode.RGB,
    _RGBA_CHANNELS: ImageMode.RGBA,
}


def get_image_color_info(image: GriptapeImage) -> ImageColorInfo:
    """Get color space and channel count from an image.

    The color space is named as Pillow names modes ("L", "LA", "RGB", "RGBA"), with the pixel type
    after a semicolon when it isn't 8-bit: "RGB;uint16", "RGBA;half". 16-bit gray is "I;16", as in
    Pillow.

    Args:
        image: The image

    Returns:
        ImageColorInfo with color_space and channels
    """
    channels = image.channels
    pixel_type = image.pixel_type
    color_space = str(_MODE_NAMES.get(channels, f"{channels}-channel"))
    if pixel_type == PixelType.UINT16 and channels == _GRAY_CHANNELS:
        color_space = "I;16"
    elif pixel_type != PixelType.UINT8:
        color_space = f"{color_space};{pixel_type}"
    return ImageColorInfo(color_space=color_space, channels=channels)


def get_image_format_from_artifact(image_artifact: ImageUrlArtifact | ImageArtifact) -> str:
    """Determine image format from an image artifact.

    Args:
        image_artifact: ImageUrlArtifact or ImageArtifact

    Returns:
        Image format string (e.g., "JPEG", "PNG", "WEBP")
    """
    if isinstance(image_artifact, ImageArtifact):
        return image_artifact.format or "UNKNOWN"

    if isinstance(image_artifact, ImageUrlArtifact):
        try:
            image = load_image_from_url(image_artifact.value)
        except Exception:
            return "UNKNOWN"
        else:
            return str(image.format) if image.format else "UNKNOWN"

    return "UNKNOWN"


def get_image_dimensions_from_artifact(
    image_artifact: ImageUrlArtifact | ImageArtifact | None,
) -> tuple[int, int]:
    """Get image dimensions from an image artifact, reading only the header of a URL's image.

    Args:
        image_artifact: ImageUrlArtifact or ImageArtifact

    Returns:
        Tuple of (width, height) or (0, 0) if unable to determine
    """
    if image_artifact is None:
        return (0, 0)

    if isinstance(image_artifact, ImageArtifact):
        return (image_artifact.width, image_artifact.height)

    if isinstance(image_artifact, ImageUrlArtifact):
        try:
            image = load_image_from_url(image_artifact.value)
        except Exception:
            logger.warning("Could not determine image dimensions from ImageUrlArtifact")
            return (0, 0)
        else:
            return (image.width, image.height)

    return (0, 0)


def apply_mask_transformations(
    alpha: GriptapeImage,
    *,
    grow_shrink: float = 0,
    invert: bool = False,
    blur_radius: float = 0,
    context_name: str = "mask",
) -> GriptapeImage:
    """Apply all mask transformations in the correct order.

    Args:
        alpha: Gray image representing the alpha channel/mask
        grow_shrink: Positive values shrink (erode), negative values grow (dilate), by that many pixels
        invert: Whether to invert the mask
        blur_radius: Blur radius (0 = no blur)
        context_name: Name for debug logging (e.g., "mask", "Paint Mask")

    Returns:
        The transformed mask

    Order of operations: grow/shrink first (modify mask shape), then invert, then blur.
    """
    pipeline = GriptapeImagePipeline(alpha)
    # Growing or shrinking by n pixels is one pass of a (2n + 1)-pixel square, the same as n 3x3 passes
    pixels = int(abs(grow_shrink))
    if pixels > 0:
        logger.debug("%s: grow/shrink by %d pixels", context_name, pixels)
        if grow_shrink > 0:
            pipeline.erode(2 * pixels + 1)
        else:
            pipeline.dilate(2 * pixels + 1)

    if invert:
        pipeline.invert()

    if blur_radius != 0:
        logger.debug("%s: blur with radius %s", context_name, blur_radius)
        pipeline.gaussian_blur(blur_radius)

    return pipeline.end()


def shrink_image_to_size(image_bytes: bytes, max_size_bytes: int, context_name: str = "image") -> bytes:
    """Best-effort shrink image to ensure it fits under the byte limit while maximizing quality.

    Uses a strategy that progressively reduces quality and resolution to find the largest
    file size (best quality) that still fits under the limit.

    Args:
        image_bytes: Raw image bytes
        max_size_bytes: Maximum allowed size in bytes
        context_name: Name for logging context (e.g., node name)

    Returns:
        Possibly converted/compressed bytes, or original if shrinking fails or unnecessary
    """
    if len(image_bytes) <= max_size_bytes:
        return image_bytes

    try:
        img = GriptapeImage.open(image_bytes)
        # Prefer WEBP for better compression and alpha support
        target_format = ImageFormat.WEBP

        orig_w, orig_h = img.width, img.height

        # Try lossless first (best quality)
        data = img.encode(target_format, lossless=True)
        image_size_bytes = len(data)
        logger.info(
            "%s downscale attempt: lossless size=%.2fMB",
            context_name,
            image_size_bytes / (1024 * 1024),
        )
        if image_size_bytes <= max_size_bytes:
            logger.info("%s shrunk image to %.2fMB (lossless)", context_name, image_size_bytes / (1024 * 1024))
            return data

        # Finer-grained scales for better quality preservation
        scales = [1.0, 0.75, 0.5]
        qualities = [100, 95, 85]

        for scale in scales:
            w = max(1, int(orig_w * scale))
            h = max(1, int(orig_h * scale))
            resized = img
            if (w, h) != (orig_w, orig_h):
                resized = GriptapeImagePipeline(img).resize(w, h, ResizeFilter.BICUBIC).end()

            for q in qualities:
                data = resized.encode(target_format, quality=q)
                image_size_bytes = len(data)
                logger.info(
                    "%s downscale attempt: scale=%.2f quality=%d size=%.2fMB",
                    context_name,
                    scale,
                    q,
                    image_size_bytes / (1024 * 1024),
                )
                if image_size_bytes <= max_size_bytes:
                    logger.info("%s shrunk image to %.2fMB (q=%d)", context_name, image_size_bytes / (1024 * 1024), q)
                    return data
    except Exception as e:
        logger.warning("%s downscale failed: %s", context_name, e)
    logger.warning("%s returning original image bytes after downscale attempts", context_name)
    return image_bytes


def resize_image_for_resolution(image_bytes: bytes, max_dimension: int, context_name: str = "image") -> bytes:
    """Resize image to fit within max_dimension while maintaining aspect ratio.

    Args:
        image_bytes: Raw image bytes
        max_dimension: Maximum width or height in pixels
        context_name: Name for logging context (e.g., node name)

    Returns:
        Resized image bytes, or original if resizing fails or unnecessary
    """
    try:
        img = GriptapeImage.open(image_bytes)
        width, height = img.width, img.height

        # Check if resizing is needed
        if width <= max_dimension and height <= max_dimension:
            return image_bytes

        scale = min(max_dimension / width, max_dimension / height)
        new_width = int(width * scale)
        new_height = int(height * scale)

        logger.info(
            "%s resizing image from %dx%d to %dx%d",
            context_name,
            width,
            height,
            new_width,
            new_height,
        )

        resized = GriptapeImagePipeline(img).resize(new_width, new_height, ResizeFilter.LANCZOS).end()

        # Keep the source's format if it can be written, otherwise use WEBP
        match img.format:
            case ImageFormat.MPO:
                img_format = ImageFormat.JPEG
            case ImageFormat() as source_format if source_format in WRITABLE_IMAGE_FORMATS:
                img_format = source_format
            case _:
                img_format = ImageFormat.WEBP
        return resized.encode(img_format)
    except Exception as e:
        logger.warning("%s resize failed: %s", context_name, e)
        return image_bytes
