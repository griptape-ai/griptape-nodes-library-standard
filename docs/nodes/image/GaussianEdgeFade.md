---
description: Fades the edges of an image to transparent with an adjustable curve and blur.
when_to_use: Use it to soften the border of a plate or element so it blends into the shot when composited, or to make a vignette-style fade.
---

## When to use it

Use it when an image's hard rectangular border would show once you layer it over something else, such as a set extension plate that has to blend into the main plate.

## Example

1. Connect the plate from **Load Image** to `input_image`.
1. Leave `fade_mode` on `percentage` and set `fade_distance` to 10.
1. Turn off `apply_bottom` to keep the bottom edge solid.
1. Run the workflow. The result, with the fade in its alpha channel, is in `output_image`. Use it as a layer in [Image Blend Compositor](ImageBlendCompositor.md).

## Parameters

### fade_mode and fade_distance

`percentage` (default) is a percent of the image's smaller side. `pixels` is an absolute size. `fade_distance` runs from 0 to 50 and defaults to 5. A fade of 0 leaves the alpha unchanged.

### blur_radius

0 to 100, default 10. A Gaussian blur on the fade mask. Higher values soften the transition and spread it inward and outward.

### fade_curve

0.5 to 4.0, default 2.0. 1.0 is linear. Above 1.0 the edge stays transparent for longer. Below 1.0 the edge stays more opaque.

### edge_shape

`square` (default) fades each edge independently with straight transitions. `rounded` rounds the corners and is slow on large images.

### replace_mask

Off by default. When off and the input is RGBA, the fade is multiplied with the existing alpha, so existing cutouts are kept. When on, the existing alpha is ignored and only the fade is used.

### apply_top, apply_bottom, apply_left, apply_right

All on by default. Turn an edge off to leave it solid.

### output_image

The image as an RGBA PNG, saved to the location set in `output_file`. If `input_image` is empty, the node does nothing.

## Edge cases

- In `square` shape, an edge is skipped when the fade distance is more than half of the image's size in that direction. The node does not shrink the fade to fit.
