---
description: Resizes an image by percentage, by width or height, to exact dimensions, to an aspect ratio, or to dimensions divisible by a number.
when_to_use: Use it to make proxies, thumbnails or model-ready sizes from a plate, render or reference frame.
---

## When to use it

You have an image at one size and need it at another. For example, scale a 4K plate to 25% for a proxy, fit a frame into a 1920x1080 delivery canvas, or round a render's size to a multiple of 64 before sending it to a generation model.

## Example

1. Add a **Load Image** node and pick a plate.
1. Add **Rescale Image** and connect the image to `input_image`.
1. Set `resize_mode` to `percentage` and `percentage_scale` to 50 for a half-size proxy.
1. Run the workflow. The resized image is in `output`.

## Parameters

### resize_mode

Decides which of the other settings are shown.

- `width`: sets the width to `target_size` and scales the height to keep the aspect ratio.
- `height`: sets the height to `target_size` and scales the width to keep the aspect ratio.
- `width and height`: produces exactly `target_width` by `target_height`, using `fit_mode`.
- `percentage` (default): scales both sides by `percentage_scale`.
- `aspect ratio`: changes the image to the ratio in `aspect_ratio`, using `fit_mode`. A landscape or square ratio keeps the original width and computes a new height. A portrait ratio keeps the original height and computes a new width.
- `divisible by`: rounds the size to multiples of `divisor`.

### target_size

Pixels, 1 to 8000, default 1000. Used by the `width` and `height` modes.

### percentage_scale

1 to 500, default 100. 100 is the original size, 50 is half and 200 is double.

### target_width, target_height

Pixels, 1 to 8000, default 1000 each. Used by the `width and height` mode.

### aspect_ratio

Default `16:9`. Accepts ratio notation such as `16:9` or a decimal such as `1.77`. The scan button next to the field fills in the ratio of the connected input image.

### fit_mode

Used by the `width and height` and `aspect ratio` modes.

- `fit` (default): scales the whole image to fit inside the target size and centers it on a canvas of exactly the target size. The empty area is filled with `background_color`.
- `fill`: scales the image to cover the target size and crops the overflow from the center.
- `stretch`: resizes to the exact size and ignores the original proportions.

### background_color

Default `#000000`. Shown only when `fit_mode` is `fit`. If the image has an alpha channel, the image is pasted onto the canvas using its alpha, so the background shows through transparent areas.

### divisor, rounding, preserve_aspect_ratio, anchor_dimension

Used by the `divisible by` mode.

- `divisor`: 1 to 1024, default 64. Width and height become multiples of this number.
- `rounding`: `round` (default), `floor` or `ceil`.
- `preserve_aspect_ratio`: on by default. The anchor dimension is rounded first and the other side is scaled by the same factor and then rounded. Off rounds width and height independently.
- `anchor_dimension`: `auto` (default), `width` or `height`. Shown only when `preserve_aspect_ratio` is on. `auto` uses the larger side.

### resample_filter

`nearest`, `box`, `bilinear`, `hamming`, `bicubic` or `lanczos` (default). `lanczos` is the slowest and generally the sharpest. `nearest` is the fastest and gives a blocky result.

### output_format

`auto` (default) keeps the input image's format. Otherwise pick `PNG`, `JPEG` or `WEBP`. Converting an image with transparency to `JPEG` flattens it onto white.

### output_file

Where the result is saved. The node writes the image to the project file location set here.

### output

The resized image.

## Notes

The node has a live preview of the result above its settings. The preview is only a display. The real processing happens when you run the node. Results and errors are reported in the status group, and the `logs` output is hidden by default.
