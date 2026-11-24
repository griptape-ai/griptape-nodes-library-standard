---
description: Applies one of ffmpeg's built-in color curve presets to a video, such as vintage, cross process or strong contrast.
when_to_use: Use it when you want a quick look or contrast change on a clip, such as a dailies grade or a stylized insert.
---

## When to use it

You want a preset look rather than manual grading. Pick a `curve_preset` and run. For example, apply `increase_contrast` to a flat log render so a review clip is easier to read.

The node offers presets only. There is no way to enter custom curve points. With `none` the video is re-encoded unchanged.

The audio track is re-encoded to AAC at 192 kbps, not copied. The ffmpeg run times out after 300 seconds.

## Parameters

### curve_preset

One of `none` (default), `color_negative`, `cross_process`, `darker`, `increase_contrast`, `lighter`, `linear_contrast`, `medium_contrast`, `negative`, `strong_contrast` or `vintage`. These are the preset names of ffmpeg's `curves` filter.

### output_frame_rate

`auto` keeps the frame rate of the input. A specific rate converts the video to it.

### processing_speed

Trades encode time against quality: `fast`, `balanced` or `quality`.
