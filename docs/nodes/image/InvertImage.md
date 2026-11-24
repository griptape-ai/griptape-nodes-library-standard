---
description: Creates a negative of an image by inverting its color values.
when_to_use: Use it to flip an image's tones, for example to check a matte or to make a negative for a look test.
---

## When to use it

You want the negative of an image. For example, invert a luminance matte, or flip a reference frame to check its contrast.

## Example

1. Add a **Load Image** node and pick a frame.
1. Add **Invert Image** and connect the image to `input_image`.
1. Run the workflow. The negative is in `output`.

## Parameters

### input_image

The image to invert. The node reads it according to its mode:

- RGBA: the color channels are inverted and the original alpha is kept.
- RGB: all channels are inverted.
- Any other mode: converted to RGB first, then inverted. Any alpha in those modes is dropped.

### output

The inverted image as a PNG, saved to the project location set in `output_file`. The node has a live preview above its settings. The preview is a display only and the saved result is produced when the node runs.

If `input_image` is empty, the node fails with a message asking you to connect an image.
