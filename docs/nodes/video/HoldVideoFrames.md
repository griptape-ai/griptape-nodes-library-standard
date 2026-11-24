---
description: Repeats each frame so the video updates only every few frames, giving a stepped look at the same speed and length.
when_to_use: Use it when you want smooth footage to look like it was animated on twos or threes, or to give a clip a stop-motion feel.
---

## When to use it

You want a stepped look without changing timing. The node drops the video to a lower frame rate and then brings it back to the original rate, so each kept frame is shown several times. The clip keeps its original duration and frame rate. This is not slow motion. To change speed, use [Change Speed](ChangeSpeed.md).

For example, set `hold_frames` to 2 on a smooth CG render to get the look of animation on twos.

The audio track is copied unchanged, so it stays in sync. The ffmpeg run times out after 300 seconds.

## Parameters

### hold_frames

From 1 to 10, default 2. The video updates every this many frames. At 1 the picture is unchanged.

### output_frame_rate

`auto` keeps the frame rate of the input. A specific rate converts the video to it after the hold is applied.

### processing_speed

Trades encode time against quality: `fast`, `balanced` or `quality`.
