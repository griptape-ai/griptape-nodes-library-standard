---
description: Mirrors a video left to right, top to bottom, or both.
when_to_use: Use it when a clip is mirrored or upside down, or when you want to flip a plate to match another shot.
---

## When to use it

You need to mirror a clip. For example, flip a plate horizontally to get a reverse angle that matches an adjacent shot.

The audio track is copied unchanged. The ffmpeg run times out after 300 seconds.

## Parameters

### direction

`horizontal` (default) mirrors left to right. `vertical` flips top to bottom. `both` does both, which is the same as a 180 degree rotation.

### output_frame_rate

`auto` keeps the frame rate of the input. A specific rate converts the video to it.

### processing_speed

Trades encode time against quality: `fast`, `balanced` or `quality`.
