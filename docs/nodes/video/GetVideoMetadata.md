---
description: Reads technical details from a video, such as size, aspect ratio, frame rate, duration and codec.
when_to_use: Use it when another node needs a video's width, height or frame rate, or when you want to check what a clip actually contains.
---

## When to use it

You want facts about a video file without opening it elsewhere. Connect a video and read the outputs. The values are filled in as soon as a video is set, and again when the node runs. With no video connected, the numeric outputs are 0, `codec_name` is `unknown` and `aspect_ratio_string` is `0:0`.

Outputs that start with `optional_` stay empty when the file does not report them.

## Parameters

### width and height

The size of the picture in pixels, as shown on screen. `optional_coded_width` and `optional_coded_height` are the stored size, which can differ.

### frame_rate

The playback rate as a decimal, such as 29.97. It comes from `avg_frame_rate` when that is valid, then from `r_frame_rate`, then from `optional_nb_frames` divided by the duration.

`r_frame_rate` and `avg_frame_rate` are the raw fractions as ffprobe reports them. Prefer `frame_rate` for playback.

### aspect_ratio_decimal and aspect_ratio_string

Width divided by height, and the same ratio written as text such as `16:9`.
