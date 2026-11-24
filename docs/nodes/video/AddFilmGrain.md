---
description: Adds animated film grain to a video, weighted by how bright each pixel is.
when_to_use: Use it when you want a clean digital plate or render to match film-shot footage, or to give a clip a filmic texture.
---

## When to use it

You want grain on a whole clip. The grain changes every frame, and its strength depends on each pixel's brightness. For example, add grain to a clean CG render so it sits with the filmed plates around it in the cut.

The audio track is copied unchanged. The ffmpeg run times out after 300 seconds.

## Parameters

### grain_intensity

From 0.05 to 1.0, default 0.15. Higher values give heavier grain.

### luminance_threshold

From 50 to 100, default 75. The brightness at which grain is strongest. The filter measures pixel brightness on a 0 to 255 scale, and grain gets weaker as a pixel moves away from this value.

### grain_scale

From 1.0 to 4.0, default 2.0. The grain texture is generated at the frame size divided by this value and then scaled up, so higher values give larger, coarser grain.

### output_frame_rate

`auto` keeps the frame rate of the input. A specific rate converts the video to it.

### processing_speed

Trades encode time against quality: `fast`, `balanced` or `quality`.
