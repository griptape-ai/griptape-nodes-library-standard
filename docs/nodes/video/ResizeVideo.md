---
description: Scales a video by percentage, width, height, exact size or to dimensions divisible by a number.
when_to_use: Use it when you need a proxy or a delivery size, or when a model needs video dimensions that are a multiple of 16 or 64.
---

## When to use it

You want a different resolution. For example, make a half-size proxy of a plate for a review, or scale a clip to 1280x720 with black bars to match a delivery spec. To cut part of the frame without scaling, use [Crop Video](CropVideo.md).

The settings shown change with `resize_mode`. The audio track is copied unchanged. The ffmpeg run times out after 300 seconds.

## Parameters

### resize_mode

`percentage` (default), `width`, `height`, `width and height` or `divisible by`.

- `percentage` uses `percentage`.
- `width` and `height` use `target_size` and keep the aspect ratio. The other side is chosen to keep proportions and an even value.
- `width and height` uses `target_width`, `target_height` and `fit_mode`.
- `divisible by` uses `divisor`, `rounding`, `preserve_aspect_ratio` and `anchor_dimension`.

Output dimensions are always made even, because video encoders need that.

### percentage

1 to 500, default 100. Scales both sides by this percentage.

### target_size

1 to 8000 pixels, default 1000. The width in `width` mode or the height in `height` mode.

### target_width and target_height

1 to 8000 pixels, default 1000 each.

### fit_mode

How the video fills `target_width` by `target_height`. `fit` (default) keeps the aspect ratio and pads the rest with `background_color`. `fill` keeps the aspect ratio and crops the excess. `stretch` ignores the aspect ratio.

### background_color

Hex color, default `#000000`. The padding color for `fit`. It is shown only when `fit_mode` is `fit`.

### divisor

An even number from 2 to 1024, default 64. Width and height become multiples of it.

### rounding

`round` (default) goes to the nearest multiple. `floor` goes down and `ceil` goes up. A side is never smaller than the divisor.

### preserve_aspect_ratio

On by default. The anchor side is rounded first and the other side is scaled by the same factor and then rounded. When off, each side is rounded on its own.

### anchor_dimension

`auto` (default), `width` or `height`. The side rounded first when `preserve_aspect_ratio` is on. `auto` uses the larger side. Shown only while `preserve_aspect_ratio` is on.

### scaling_algorithm

`bicubic` (default), `neighbor`, `bilinear` or `lanczos`.

### lanczos_parameter

1.0 to 10.0, default 3.0. The Lanczos alpha value. Higher values are sharper and can ring. Lower values are smoother. Shown only for `lanczos`.

### resized_video

The resized video. It is saved to the project using the `output_file` setting, whose default name is `resized_video.mp4`.
