---
description: Creates a mask from an image's alpha channel, lets you paint on it in the editor, and applies it back to the image.
when_to_use: Use it when you need to hand-paint a matte, such as cleaning up a roto or isolating part of a plate, and want both the mask and the masked image.
---

## When to use it

You have an image and want to paint which areas stay opaque and which become transparent. Connect the image to `input_image`. The node builds a starting mask from the image's alpha channel (white where opaque, black where transparent). Open the paint tools on `output_mask` to edit it. `output_image` is the input image with the mask applied as its alpha.

The mask you paint is kept when the input image changes, but only if you have edited it. An unedited mask is rebuilt from the new image's alpha.

## Parameters

### output_mask

The editable mask, saved with the name in `mask_file` (default `mask.png`). White areas stay opaque and black areas become transparent. The red channel of the painted mask is used as the alpha.

### invert_mask

Flips the mask so white and black swap roles in the result. Order of operations is grow or shrink, then invert, then blur.

### grow_shrink

From -100 to 100, in steps of about 1 pixel. Positive values shrink the white area and negative values grow it.

### blur_mask

From 0 to 100. Softens the mask edge.

### output_image

The input image with the transformed mask as its alpha, saved as PNG with the name in `output_file` (default `output.png`). The transforms change `output_image` only. They are not written back into `output_mask`.
