---
description: Darkens the edges of a video, or lightens them, around a center point you choose.
when_to_use: Use it when you want to pull the eye toward the middle of the frame or add lens-style light falloff to a clip.
---

## When to use it

You want the corners of a clip to fall off. For example, add a light vignette to a graded sequence so a comped element sits better in the frame.

The node uses ffmpeg's `vignette` filter. The audio track is copied unchanged. The ffmpeg run times out after 300 seconds.

## Parameters

### angle

From 0.1 to 3.14 (pi), default 0.628 (pi/5). This is the lens angle for the filter. Smaller values give a stronger vignette. Larger values push the falloff toward the corners.

### center_x and center_y

From -1.0 to 1.0, default 0. Moves the center of the vignette from the middle of the frame. `center_x` of -1 is the left edge and 1 is the right edge. `center_y` of -1 is the top edge and 1 is the bottom edge.

### aspect

From 0.1 to 10.0, default 1.0. The aspect ratio of the vignette shape.

### mode

`forward` (default) darkens the edges. `backward` lightens them.

### output_frame_rate

`auto` keeps the frame rate of the input. A specific rate converts the video to it.

### processing_speed

Trades encode time against quality: `fast`, `balanced` or `quality`.
