---
description: Resizes an image by percentage, by width or height, to exact dimensions, to an aspect ratio, or to dimensions divisible by a number.
when_to_use: Use it to make proxies, thumbnails or model-ready sizes from a plate, render or reference frame.
---

## When to use it

You have an image at one size and need it at another. For example, scale a 4K plate to 25% for a proxy, fit a frame into a 1920x1080 delivery canvas, or round a render's size to a multiple of 64 before sending it to a generation model.

## Example

1. Connect a **Load Image** plate to `input_image` on **Rescale Image**.
1. Set `resize_mode` to `percentage` and `percentage_scale` to 50 for a half-size proxy.
1. Run the workflow. The resized image is in `output`.

## Parameters

### resize_mode

Decides which other settings are shown.

- `width`: sets the width to `target_size` and scales the height to keep the aspect ratio.
- `height`: sets the height to `target_size` and scales the width to keep the aspect ratio.
- `width and height`: produces exactly `target_width` by `target_height`, using `fit_mode`.
- `percentage` (default): scales both sides by `percentage_scale`.
- `aspect ratio`: changes the image to the ratio in `aspect_ratio`, using `fit_mode`. A landscape or square ratio keeps the width. A portrait ratio keeps the height.
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

Used by `width and height` and `aspect ratio`.

- `fit` (default): scales the whole image to fit inside the target size and centers it on a canvas of exactly that size. The empty area is filled with `background_color`.
- `fill`: scales the image to cover the target size and crops the overflow from the center.
- `stretch`: resizes to the exact size and ignores the original proportions.

### background_color

Default `#000000`. Shown only when `fit_mode` is `fit`. The background shows through transparent areas of the image.

### divisor, rounding, preserve_aspect_ratio, anchor_dimension

Used by the `divisible by` mode.

- `divisor`: 1 to 1024, default 64. Width and height become multiples of this number.
- `rounding`: `round` (default), `floor` or `ceil`.
- `preserve_aspect_ratio`: on by default. The anchor dimension is rounded first and the other side follows. Off rounds width and height independently.
- `anchor_dimension`: `auto` (default), `width` or `height`. Shown only when `preserve_aspect_ratio` is on. `auto` uses the larger side.

### resample_filter

`nearest`, `box`, `bilinear`, `hamming`, `bicubic` or `lanczos` (default). `lanczos` is the slowest and generally the sharpest. `nearest` is the fastest and gives a blocky result.

### output_format

`auto` (default) keeps the input format, or pick `PNG`, `JPEG` or `WEBP`. Converting transparency to `JPEG` flattens it onto white.

### output_file

Where the result is saved in the project.

## Notes

The live preview is display only. Processing happens when you run the node.
