---
description: Blends one image over another with a blend mode, opacity and position, like two layers in a compositing app.
when_to_use: Use it when you need a quick layer composite, such as laying a grain plate, a light wrap or a graphic over a background without opening a compositing package.
---

## When to use it

You have a base image and a second image to lay on top of it. Connect the base to `input_image` and the layer to `blend_image`. For example, multiply a dirt pass over a background plate, or screen a smoke element over a frame.

The node processes both images as RGBA. The output is always saved as PNG, so the `output_format` and `quality` settings have no effect on this node.

## Example

1. Load a background plate and a smoke element.
1. Connect the plate to `input_image` and the smoke to `blend_image`.
1. Set `blend_mode` to `screen`.
1. Lower `opacity` to 0.7 for a lighter pass.
1. Check `output`. It updates as you change the settings and when you run the node.

## Parameters

### blend_mode

`normal`, `multiply`, `screen`, `overlay`, `soft_light`, `hard_light`, `darken`, `lighten`, `difference`, `exclusion`, `add`, `subtract`, `logical_and`, `logical_or` or `logical_xor`.

With any mode other than `normal`, the blend image is resized to the base image's size before blending, and `resize_blend_to_fit` has no extra effect. The `exclusion` mode currently produces the same result as `logical_xor`, and `logical_and` returns the blend image unchanged.

### opacity

From 0.0 to 1.0. Default is 1.0. It scales the blend image's alpha, so soft edges in the layer stay soft.

### blend_position_x and blend_position_y

Pixel position of the blend image's top-left corner on the base. When both are 0 (the default), the blend image is centered on the base. Setting only one of them to a non-zero value places the image from the top-left corner on both axes.

### resize_blend_to_fit

Resizes the blend image to the base image's dimensions before positioning.

### preserve_alpha

On by default. Uses the blend image's alpha when compositing. Turn it off to paste the blend image's color channels without its transparency.

### invert_blend

Inverts the blend image's colors before compositing. Its alpha is kept.

### output

The composited image, at the base image's size.
