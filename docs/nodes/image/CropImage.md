---
description: Crops an image to a rectangle, with optional zoom and rotation, using an interactive editor or numeric values.
when_to_use: Use it to cut a region out of a plate or render, such as a hero area for review or a detail for a closer look.
---

## When to use it

You want part of an image. For example, crop the area of a frame where a comp issue shows up so it can be reviewed at a larger size, or reframe a render before sending it on.

## Example

1. Add a **Load Image** node and pick a frame.
1. Add **Crop Image** and connect the image to `input_image`. The editor shows the image.
1. Drag the crop box in the editor, or open the crop coordinates group and enter `left`, `top`, `width` and `height`.
1. Run the workflow. The cropped image is in `output`.

## Parameters

### crop_editor

An interactive widget where you drag the crop area. It reads and writes `left`, `top`, `width`, `height`, `zoom` and `rotate`. A value that has an incoming connection is locked in the editor.

### left, top, width, height

Pixels. `left` and `top` default to 0. `width` and `height` default to 0, which means the full image width or height. When an image is set, the slider limits update to the image size.

The node adjusts values that fall outside the image instead of failing:

- An origin past the image edge is moved to the last pixel column or row.
- A box that extends past the edge is cut at the edge.
- A negative `left` or `top` with a set `width` or `height` keeps the box's far edge, so only the part on the image is cropped.
- A box entirely off the image becomes a one pixel strip at the edge.

The adjustment is logged. Negative `width` or `height` fails validation.

### zoom

Percent, 0 to 500, default 100. Zoom changes the size of the crop window around its center. 200 halves the window's width and height, so the output covers a quarter of the area. 50 doubles them, up to the image edges. The output is not scaled back up. Values are limited to between 10% and 1000% of the window when applied.

### rotate

Degrees, -180 to 180, default 0. The image is rotated around the center of the crop window, and then the window is cut out. The window stays axis aligned. Positive values rotate counter-clockwise. Areas that come from outside the original image are filled with `background_color`.

### background_color

Default `#00000000` (transparent). Used to fill areas exposed by rotation. If a color cannot be parsed, transparent is used.

### output_format

`PNG` (default), `JPEG` or `WEBP`. For `JPEG`, transparent areas are flattened onto black.

### output_quality

0.0 to 1.0, default 0.9. Applies to `JPEG`. `WEBP` is currently saved lossless regardless of this value.

### output_file

Where the result is saved. The node writes the image to the project file location set here.

### output

The cropped image.
