---
description: Uses a mask as the alpha channel of an image, with options to grow, shrink, invert and blur the mask first.
when_to_use: Use it to cut out a subject, or to apply a roto matte to a plate so it can be composited over another background.
---

## When to use it

You have an image and a matte and want the image to be transparent where the matte is black. For example, apply a roto matte to a plate to get a cutout with soft edges, ready to be composited with [Image Blend Compositor](ImageBlendCompositor.md).

## Example

1. Add a **Load Image** node for the plate and another for the matte, or use **Paint Mask** to draw one.
1. Add **Apply Mask**. Connect the plate to `input_image` and the matte to `input_mask`.
1. Set `channel` to the channel that holds the matte. A black and white matte has the same values in red, green and blue, so `red` works.
1. Set `grow_shrink` to 2 to pull the matte edge in slightly, and `blur_mask` to 3 to soften it.
1. Run the workflow. The cutout is in `output`.

## Parameters

### input_image, input_mask

Both must be connected. The node fails validation if either is missing. The image is converted to RGBA. The mask is resized to the image size with nearest-neighbor sampling.

### channel

`red` (default), `green`, `blue` or `alpha`. Which channel of the mask is used as the alpha. How it is read depends on the mask mode:

- RGB: the selected channel. `alpha` gives a fully opaque result because an RGB mask has no alpha.
- RGBA: the selected channel.
- Grayscale (L): used as is, and `channel` is ignored.
- Grayscale with alpha (LA): the alpha for `alpha`, otherwise the gray channel.

Other modes fail with an unsupported mode error.

### invert_mask

Off by default. Flips the mask after any grow or shrink.

### grow_shrink

-100 to 100, default 0. Positive values shrink the white area and negative values grow it. The value is the number of 3x3 erosion or dilation passes. Large values are slower.

### blur_mask

0 to 100, default 0. A blur radius applied to the mask after inversion.

### apply_mask_blur_to_edges

Off by default. When on and `blur_mask` is above 0, the image's outer edges also fade to transparent over a distance of `blur_mask` pixels. This is computed pixel by pixel and is slow on large images.

### output

The image with the processed mask as straight (unpremultiplied) alpha. The RGB values are not changed. It is a PNG saved to the project location set in `output_file`.

## Notes

The node recomputes the output when you change a connection or one of the mask settings, as well as when it runs.
