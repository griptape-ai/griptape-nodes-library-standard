---
description: Adjusts an image's tonal range with input levels, a midtones gamma and output levels, like the Levels tool in Photoshop.
when_to_use: Use it to fix exposure or contrast, or to bring a render's tonal range closer to a plate or reference.
---

## When to use it

You want to control shadows, midtones and highlights separately. Use it to fix exposure, make a high or low contrast look, or match the tonal range of several images.

## Example

1. Connect an image from **Load Image** to `input_image`.
1. Set `shadows` to 20 and `highlights` to 240 to clip a little of each end of the range.
1. Set `midtones` to 1.2 to brighten the midtones.
1. Set `output_shadows` to 10 and `output_highlights` to 245 to slightly lift the blacks and lower the whites.
1. Connect `output` to [Display Image](DisplayImage.md) to view the result.

## Parameters

### shadows

0 to 255, default 0. Pixels below this are mapped to `output_shadows`, so higher values clip more shadow detail. If `shadows` is not below `highlights`, the node logs a warning and uses `highlights - 1`.

### midtones

0.1 to 10.0, default 1.0, which is no change. The node applies `output = input^(1/midtones)`, which matches Photoshop's gamma. Values above 1.0 brighten the midtones and values below 1.0 darken them.

> [!NOTE]
> The parameter tooltip says the opposite (below 1.0 lightens, above 1.0 darkens). The tooltip is wrong.

### highlights

0 to 255, default 255. Pixels above this are mapped to `output_highlights`, so lower values clip more highlight detail.

### output_shadows

0 to 255, default 0. The input shadows level is mapped to this value. Higher values brighten the image overall.

### output_highlights

0 to 255, default 255. The input highlights level is mapped to this value. Lower values darken the image overall.

### output_format

`auto` (default) saves in the input image's format. Also `PNG`, `JPEG` or `WEBP`. An RGBA image saved as JPEG is flattened onto white. `quality` is not used.

### output

The adjusted image.

## Edge cases

- A grayscale image with alpha (LA) comes out fully opaque.
- The same adjustment applies to the red, green and blue channels. Transparency in RGBA images is preserved.

