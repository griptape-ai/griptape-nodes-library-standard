---
description: Grows or shrinks the white area of a mask video by a number of pixels.
when_to_use: Use it when a matte is a little tight or loose, for example to expand a roto matte to cover an edge or shrink it to remove a fringe.
---

## When to use it

You have a mask video and want its edges moved. Positive values grow the white area. Negative values shrink it. For example, expand a roto matte by 3 pixels to cover a soft edge before a paint step.

The node runs ffmpeg's `dilation` or `erosion` filter once per pixel of adjustment, on a grayscale version of the mask. The output is re-encoded as H.264. At an adjustment of 0 the input is passed through unchanged and nothing is processed. The ffmpeg run times out after 600 seconds.

## Parameters

### mask_video

The mask to adjust, shown as "Mask Video". Required.

### original_video

Optional, shown as "Original Video". It is only used by the preview on the node to overlay the mask on the picture. It does not affect the output.

### adjustment

From -25 to 25 pixels, default 0. Can be set with the slider in the preview or on the node.

### output_mask

The adjusted mask, shown as "Adjusted Mask Video". It is saved to the project using the `output_file` setting, whose default name is `adjusted_mask.mp4`.
