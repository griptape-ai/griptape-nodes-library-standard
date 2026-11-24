---
description: Speeds a video up or slows it down by a multiplier, with the audio adjusted to match or removed.
when_to_use: Use it when you want a clip to play faster or slower, such as a quick review of a long take or a slowed-down reference of fast action.
---

## When to use it

You want to change how fast a clip plays. A `speed` of 2 plays it twice as fast and halves the length. A `speed` of 0.5 plays it at half speed and doubles the length. For example, speed up a long playblast to scan it quickly.

The node retimes the existing frames. It does not interpolate new ones, so heavy slow-down looks stepped. To hold frames deliberately without changing length, see [Hold Video Frames](HoldVideoFrames.md).

The ffmpeg run times out after 300 seconds.

## Parameters

### speed

From 0.1 to 10.0, default 1.0.

### include_audio

On by default. When on, the audio is sped up or slowed down with the node's ffmpeg `atempo` filter and re-encoded to AAC at 192 kbps. When off, the output has no audio track.

The audio matches the picture across the full range. `atempo` takes 0.5 to 2.0 per filter, so outside that range the node chains several, for example four for a `speed` of 10.

### output_frame_rate

`auto` keeps the frame rate of the input. A specific rate converts the video to it.

### processing_speed

Trades encode time against quality: `fast`, `balanced` or `quality`.
