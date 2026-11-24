---
description: Offsets the red, green and blue channels of a video against each other to create chromatic aberration or a glitch look, with an optional horizontal tear.
when_to_use: Use it when you want lens-fringing or a digital glitch on a clip, such as a damaged-monitor insert or a transition effect.
---

## When to use it

You want the color channels to separate. Each channel gets its own horizontal and vertical shift in pixels. The defaults move red 6 pixels left and green 6 pixels right, which gives a classic fringe. For example, apply it to a screen insert shot to make the display look unstable.

The audio track is copied unchanged. The ffmpeg run times out after 300 seconds.

## Parameters

### red, green and blue shifts

`red_horizontal`, `red_vertical`, `green_horizontal`, `green_vertical`, `blue_horizontal` and `blue_vertical`. Each is from -50 to 50 pixels. Defaults: red horizontal -6, green horizontal 6, everything else 0.

### intensity

From 0.0 to 1.0, default 1.0. Multiplies all six channel shifts. At 0 there is no shift. It does not scale `tear_offset`.

### tear_enabled

Off by default. When on, the frame is split at `tear_position`. The top part gets the normal shifts. The bottom part gets the same shifts plus `tear_offset` horizontally on all three channels.

### tear_position

From 0.0 to 1.0, default 0.5. Where the split sits as a fraction of the frame height, measured from the top.

### tear_offset

From -50 to 50 pixels, default 10. The extra horizontal shift for the part of the frame below the split.

### output_frame_rate

`auto` keeps the frame rate of the input. A specific rate converts the video to it.

### processing_speed

Trades encode time against quality: `fast`, `balanced` or `quality`.
