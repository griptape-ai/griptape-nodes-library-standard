---
description: Pulls one channel (red, green, blue or alpha) out of an image as a grayscale mask.
when_to_use: Use it to turn a plate's alpha or a single color channel into a mask you can view or pass to other mask nodes.
---

## When to use it

You have an image and want one of its channels as a grayscale image. For example, take the alpha of a keyed element to see its matte, or the green channel of a render pass to inspect it.

## Example

1. Add a **Load Image** node and pick a keyed element with transparency.
1. Add **Display Mask** and connect the image to `input_image`.
1. Leave `channel` on `alpha`.
1. Run the workflow. The matte is in `output_mask`. Connect it to [Invert Mask](InvertMask.md) or [Apply Mask](ApplyMask.md).

## Parameters

### channel

`red`, `green`, `blue` or `alpha` (default). The node has a live preview of the chosen channel above its settings. The preview is a display only. The saved result is produced when the node runs.

How the channel is read depends on the image mode:

- RGBA: the selected channel.
- RGB: the selected channel. An RGB image has no alpha, so `alpha` gives a fully white mask.
- Grayscale (L): the image as is, whatever `channel` is.
- Grayscale with alpha (LA): the alpha for `alpha`, otherwise the gray channel.

Other modes, such as palette images, fail with an unsupported mode error.

### output_mask

The extracted channel as a grayscale PNG. It is saved to the project location set in `output_file`.
