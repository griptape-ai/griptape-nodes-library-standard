---
description: Adjusts an image's tonal range with input levels, a midtones gamma and output levels, like the Levels tool in Photoshop.
when_to_use: Use it to fix exposure or contrast, or to bring a render's tonal range closer to a plate or reference.
---

## When to use it

You want control over shadows, midtones and highlights separately. Typical uses are fixing exposure, creating high or low contrast looks, matching the tonal range of several images, and preparing an image for a specific output range.

## Example

1. Load an image with **Load Image** and connect it to `input_image`.
1. Set `shadows` to 20 and `highlights` to 240 to clip a little of each end of the range.
1. Set `midtones` to 1.2 to brighten the midtones.
1. Set `output_shadows` to 10 and `output_highlights` to 245 to slightly lift the blacks and lower the whites.
1. Connect `output` to [Display Image](DisplayImage.md) to view the result.

## Parameters

### shadows

0 to 255, default 0. Pixels below this value are mapped to `output_shadows`. Higher values clip more shadow detail. If `shadows` is not below `highlights`, the node logs a warning and uses `highlights - 1` for `shadows`.

### midtones

0.1 to 10.0 (enforced when you set it), default 1.0, which is no change. The node applies `output = input^(1/midtones)`, which matches Photoshop's gamma. Values above 1.0 brighten the midtones and values below 1.0 darken them.

> [!NOTE]
> The parameter tooltip says the opposite (below 1.0 lightens, above 1.0 darkens). The tooltip is wrong. See #1078.

### highlights

0 to 255, default 255. Pixels above this value are mapped to `output_highlights`. Lower values clip more highlight detail.

### output_shadows

0 to 255, default 0. The input shadows level is mapped to this value. Higher values brighten the image overall.

### output_highlights

0 to 255, default 255. The input highlights level is mapped to this value. Lower values darken the image overall.

### output_format

`auto` (default), `PNG`, `JPEG` or `WEBP`. `auto` saves in the input image's format, so a JPEG input gives a JPEG output. JPEG has no alpha, so an RGBA image saved as JPEG is flattened onto white. The `quality` parameter is not used by this node.

### output

The adjusted image.

## Notes

The node applies three steps: input levels clip and map the range between `shadows` and `highlights`, the gamma from `midtones` is applied, and output levels map the result to the output range. The adjustment is computed once per run as a 256-entry table and applied equally to the red, green and blue channels. Transparency in RGBA images is preserved. Grayscale (L) images are adjusted directly. Other modes are converted to RGB and back, and a grayscale image with alpha (LA) comes out fully opaque.

## Common issues

- **No change visible**: The sliders are still at their defaults.
- **Over-processed look**: Extreme values rarely work well. Subtle adjustments are usually enough.
- **Lost detail**: Setting `shadows` too high or `highlights` too low clips detail.
