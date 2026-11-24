---
description: Blurs a video with a box, gaussian or average blur at a strength you set.
when_to_use: Use it when you want to soften a whole clip, for example to make a background plate or to hide detail.
---

## When to use it

You want one even blur over the whole frame. The node applies a fast ffmpeg blur filter to every frame and copies the audio through unchanged. It does not blur only part of the image.

## Parameters

### blur_type

Chooses the filter. Only the settings that apply to the chosen type are shown.

| Type | Uses | Notes |
| --- | --- | --- |
| `box` | `radius`, `power` | Fastest. |
| `gaussian` | `sigma` | Smoother falloff. |
| `average` | `radius` | A simple box average. |

### radius

Blur size in pixels, from 1 to 50. Used by `box` and `average`.

### sigma

Strength of the gaussian blur, from 0.1 to 50. Used only by `gaussian`.

### power

How many times the box filter runs, from 1 to 5. Higher values blur more. Used only by `box`.

### output_frame_rate

`auto` keeps the frame rate of the input. A specific rate converts the video to it.

### processing_speed

Trades encode time against quality: `fast`, `balanced` or `quality`.
