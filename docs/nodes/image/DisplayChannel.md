---
description: Extracts the red, green, blue or alpha channel of an image as a grayscale image.
when_to_use: Use it when you need to inspect one channel of a plate or render, or use a channel as a mask.
---

## When to use it

You want to see or use a single channel of an image. Typical uses are checking the intensity of one color channel, debugging channel problems in a pipeline, and making a mask from color or alpha data. A mask here is a grayscale image where white is a high value and black is a low value in the selected channel.

This node is like [Display Mask](DisplayMask.md) but defaults to `red`. Display Mask defaults to `alpha`, which suits transparency work. Display Channel suits color analysis.

## Example

1. Generate or load an image with **Generate Image** or **Load Image**.
1. Connect it to `input_image`.
1. Leave `channel` on `red` or pick another.
1. Connect `output` to [Display Image](DisplayImage.md) to view the channel, or to other mask nodes.

## Parameters

### channel

`red` (default), `green`, `blue` or `alpha`.

An RGB image has no alpha channel, so selecting `alpha` on it gives an all-white image, because every pixel is fully opaque.

### input_image

Supports RGB, RGBA, grayscale (L) and grayscale+alpha (LA) images. A grayscale (L) image is passed through unchanged, whatever channel is selected. Other modes, such as palette (P) or CMYK, are rejected. Convert the image to RGB first.

### output

The channel as a grayscale image, saved as a PNG in the location set by `output_file`.

## Common issues

- **No output**: Check that an image is connected to `input_image`.
- **Solid black mask**: The selected channel has no variation or only low values.
- **Solid white mask**: The selected channel has high values throughout. For `alpha` on an RGB image this is expected.
- **Unsupported mode error**: See `input_image` above.
