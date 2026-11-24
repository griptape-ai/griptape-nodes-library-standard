---
description: Plays a video backwards, with the audio reversed, muted or left as it was.
when_to_use: Use it when you want a clip to run in reverse, such as a rewind effect or a reverse-motion test of an animation.
---

## When to use it

You want the last frame first. For example, reverse a short effects element to check that it loops cleanly.

The ffmpeg run times out after 300 seconds.

## Parameters

### audio_handling

`reverse` (default) reverses the audio along with the picture. `mute` removes the audio. `keep` leaves the original audio running forwards against the reversed picture, so it will not line up with the video.

If the input has no audio, the output has none, whatever you pick.

### output_frame_rate

`auto` keeps the frame rate of the input. A specific rate converts the video to it.

### processing_speed

Trades encode time against quality: `fast`, `balanced` or `quality`.
