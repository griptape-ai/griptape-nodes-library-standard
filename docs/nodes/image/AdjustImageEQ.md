---
description: Adjusts an image's brightness, contrast, saturation and gamma.
when_to_use: Use it for a quick exposure and color balance pass, such as matching a generated element to a plate before compositing.
---

## When to use it

You want a simple tonal correction. For example, brighten a dark plate, add contrast to a render pass, or pull the saturation down on a generated element so it sits in a scene. Connect an image to `input_image`. The preview in the editor redraws as you move the sliders. The output image is created when the node runs.

## Parameters

### brightness, contrast, saturation

Each runs from 0.0 to 3.0 and defaults to 1.0, which leaves the image unchanged. Below 1.0 reduces the effect and above 1.0 increases it. Saturation at 0.0 gives a grayscale image.

### gamma

From 0.1 to 10.0, default 1.0. Values above 1.0 brighten the midtones and values below 1.0 darken them. Alpha is left alone.

### output_format

`auto` saves in the same format as the input. Otherwise `PNG`, `JPEG` or `WEBP`. `quality` applies to `JPEG` and `WEBP`.

### output

The adjusted image, saved with the name in `output_file`. After the run, `was_successful` and `result_details` report the outcome, and `result_details` lists any failure.
