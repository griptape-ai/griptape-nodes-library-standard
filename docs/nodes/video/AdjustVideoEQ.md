---
description: Adjusts the brightness, contrast, saturation and gamma of a video with ffmpeg's eq filter.
when_to_use: Use it when you want a basic exposure and color correction on a clip, such as brightening a dark plate or desaturating a reference.
---

## When to use it

You need a simple correction on the whole clip. For example, lift a dark render and raise its contrast slightly so it reads better in a dailies review.

At the defaults the node changes nothing except the re-encode. The audio track is copied unchanged. The ffmpeg run times out after 300 seconds.

## Parameters

### brightness

From -1.0 to 1.0, default 0.0. Negative values darken, positive values brighten.

### contrast

From 0.0 to 3.0, default 1.0.

### saturation

From 0.0 to 3.0, default 1.0. At 0 the video is grayscale.

### gamma

From 0.1 to 10.0, default 1.0. This is the value passed to ffmpeg's `eq` filter, where values above 1 lighten the mid-tones and values below 1 darken them.

### output_frame_rate

`auto` keeps the frame rate of the input. A specific rate converts the video to it.

### processing_speed

Trades encode time against quality: `fast`, `balanced` or `quality`.
