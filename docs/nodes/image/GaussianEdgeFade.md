---
description: Fades the edges of an image to transparent with an adjustable curve and blur.
when_to_use: Use it to soften the border of a plate or element so it blends into the shot when composited, or to make a vignette-style fade.
---

## When to use it

You have an image whose hard rectangular border would show when you layer it over something else. For example, fade the edges of a set extension plate so it blends into the main plate.

## Example

1. Add a **Load Image** node and pick the plate.
1. Add **Gaussian Edge Fade** and connect the image to `input_image`.
1. Leave `fade_mode` on `percentage` and set `fade_distance` to 10.
1. Turn off `apply_bottom` if the plate should stay solid along the bottom.
1. Run the workflow. The result, with the fade in its alpha channel, is in `output_image`. Use it as a layer in [Image Blend Compositor](ImageBlendCompositor.md).

## Parameters

### fade_mode and fade_distance

In `percentage` mode (the default) the distance is a percentage of the smaller of the image's width and height. In `pixels` mode it is an absolute number of pixels. `fade_distance` defaults to 5. The slider runs from 0 to 50.

A fade of 0 pixels leaves the alpha unchanged. In `square` shape, an edge is skipped when the fade distance is more than half of the image's size in that direction. The node does not shrink the fade to fit.

### blur_radius

0 to 100, default 10. A Gaussian blur applied to the fade mask, so higher values soften the transition. It is applied after the fade is built, so it also spreads the fade inward and outward.

### fade_curve

0.5 to 4.0, default 2.0. The alpha across the fade is `(distance from edge / fade distance)` raised to this power. 1.0 is linear. Above 1.0 the edge stays more transparent for longer. Below 1.0 the edge stays more opaque.

### edge_shape

`square` (default) fades each edge independently with straight transitions. `rounded` fades by distance from an inner rectangle, which rounds the corners. `rounded` is calculated pixel by pixel and is slow on large images.

### replace_mask

Off by default. When off and the input is RGBA, the fade is multiplied with the image's existing alpha, so existing cutouts are kept. When on, the existing alpha is ignored and only the fade is used. Images without an alpha channel are converted to RGBA either way.

### apply_top, apply_bottom, apply_left, apply_right

All on by default. Turn an edge off to leave it solid.

### output_image

The image as an RGBA PNG, saved to the project location set in `output_file`. The node has a live preview above its settings. The saved result is produced when the node runs. If `input_image` is empty, the node does nothing.
