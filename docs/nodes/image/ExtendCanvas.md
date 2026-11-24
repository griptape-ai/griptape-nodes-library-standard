---
description: Adds empty space around an image to reach an aspect ratio or to extend each side by a set number of pixels, and outputs a mask of the new area.
when_to_use: Use it when you need more frame around an image without cropping it, for example to reformat a plate to another aspect ratio or to make room for an outpaint.
---

## When to use it

You want a larger canvas around an image and the original pixels untouched. Pick an aspect ratio preset and the node grows the canvas on one dimension until it matches that ratio. Or pick `custom` and set how many pixels to add on each side. A second output, `canvas_mask`, marks the original area so you can feed the extended image and mask to an outpainting or inpainting node.

## Example

1. Load a 4:3 plate and connect it to `input_image`.
1. Set `aspect_ratio_preset` to `21:9 landscape`.
1. Set `position` to `center` and `background_color` to `black`.
1. Run the workflow. `extended_image` is the plate with black bars on the sides, and `canvas_mask` marks where the bars are.

## Parameters

### aspect_ratio_preset

A ratio from the list, or `custom`. The default is `1:1 square`. With a preset, the node keeps one dimension of the image and enlarges the other until the canvas matches the ratio. It never shrinks the image, which keeps its original pixel size.

### top, bottom, left, right

Visible when `aspect_ratio_preset` is `custom`. Pixels to add on each side. Negative values are treated as 0. The original image is placed at the `left` and `top` offsets.

### position

Visible for presets. Where the original sits in the new canvas: `center`, `top_left`, `top_right`, `bottom_left`, `bottom_right`, `top`, `bottom`, `left` or `right`. Not used with `custom`.

### background_color

`black` (default), `white`, `transparent`, `magenta`, `green` or `blue`. The result is always RGBA.

### upscale_factor

Multiplies the calculated canvas width and height. Default is 1.0. It makes the canvas larger and does not rescale the original image.

### extended_image

The image on the larger canvas, saved as PNG with the name in `output_file` (default `extended.png`).

### canvas_mask

A PNG with black where the original image is. The extended area is white, or transparent when `background_color` is `transparent`. Saved with the name in `mask_file` (default `mask.png`).
