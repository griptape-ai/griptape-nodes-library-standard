---
description: Merges several mask videos into one by taking the brightest pixel across all of them on each frame.
when_to_use: Use it when you have separate mask videos, such as one per tracked object, and want a single matte that covers all of them.
---

## When to use it

You have a list of mask videos and want their union. For example, combine the per-object roto mattes for a shot into one matte for a paint or comp step. A pixel is white in the output if it is white in any input.

All masks must have the same resolution. They can have different lengths. Shorter masks are treated as black after they end, and the output runs as long as the longest one. The frame rate of the first video is used for the output. Frames are matched by position, so masks with different frame rates will not line up in time. The output has no audio.

## Parameters

### mask_videos

The list of mask videos. Connect several videos to it. It is shown as "Mask Videos". At least one is required. Each frame is converted to grayscale before combining.

### output_mask

The combined video, shown as "Combined Mask Video". It is saved to the project using the `output_file` setting, whose default name is `combined_mask.mp4`.

A progress bar shows the frame-by-frame work and the final encode. Each ffmpeg step times out after 300 seconds for frame extraction and 600 seconds for the final encode.
