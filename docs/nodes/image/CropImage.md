---
description: Crops an image to a rectangle, with optional zoom and rotation, using an interactive editor or numeric values.
when_to_use: Use it to cut a region out of a plate or render, such as a hero area for review or a detail for a closer look.
---

## When to use it

You want part of an image, such as the area of a frame where a comp issue shows up, or a reframed render.

## Example

1. Connect a frame from **Load Image** to `input_image`. The editor shows the image.
1. Drag the crop box in the editor, or open the crop coordinates group and enter `left`, `top`, `width` and `height`.
1. Run the workflow. The cropped image is in `output`.

## Parameters

### crop_editor

Drag the crop area here. It reads and writes `left`, `top`, `width`, `height`, `zoom` and `rotate`. A value with an incoming connection is locked.

### left, top, width, height

Pixels. `width` and `height` of 0 mean the full image.

Values outside the image are adjusted and logged, not failed. Negative `width` or `height` fails validation.

### zoom

Percent, 0 to 500, default 100. Zoom resizes the crop window around its center. 200 halves its width and height, and 50 doubles them, up to the image edges. The output is not scaled back up. The applied factor is limited to 10% through 1000%, so a zoom of 0 acts as 10%.

### rotate

Degrees, -180 to 180, default 0. Positive values rotate counter-clockwise. The image rotates around the center of the crop window, and areas from outside it are filled with `background_color`.

### background_color

Default `#00000000` (transparent). Fills areas exposed by rotation. A color that cannot be parsed falls back to transparent.

### output_format

`PNG` (default), `JPEG` or `WEBP`. For `JPEG`, transparent areas are flattened onto black.

### output_quality

0.0 to 1.0, default 0.9. Applies to `JPEG` only. `WEBP` is saved lossless regardless of this value.

### output_file

Where the result is saved in the project.

### output

The cropped image.

## Edge cases

- An origin past the image edge moves to the last pixel column or row, and a box that runs past the edge is cut at the edge.
- A negative `left` or `top` with a set `width` or `height` keeps the box's far edge, so only the part on the image is cropped.
- A box entirely off the image becomes a one pixel strip at the edge.
