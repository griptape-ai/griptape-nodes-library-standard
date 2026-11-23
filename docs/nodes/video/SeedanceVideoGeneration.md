---
description: Generates a video from a prompt, and optionally first and last frame images, with the Seedance 1.x models.
when_to_use: Use it when you want a short clip from text or from a still image and the older Seedance 1.0 or 1.5 models are enough.
---

## When to use it

You want to turn a prompt into a video, or animate from a still. Add a first frame to start from an image. Add a last frame to also control where the clip ends. For reference images, reference videos, audio or editing, use [Seedance 2.0 Video Generation](Seedance20VideoGeneration.md) or [Seedance 2.5 Video Generation](Seedance25VideoGeneration.md).

The node runs through Griptape Cloud and uses your Griptape key. Turn on the API Key Provider option to use your own ByteDance Seed key instead. The node waits for the result and saves the video. Use the `timeout` parameter (600 seconds by default) to change how long it waits. If a run times out, the Refresh button on the node fetches the result later.

## Parameters

### model_id

The model changes which inputs appear.

| Model | `first_frame` | `last_frame` | `generate_audio` |
|-------|---------------|--------------|------------------|
| `seedance-1-5-pro-251215` (default) | Yes | Yes | Yes |
| `seedance-1-0-pro-250528` | Yes | Yes | No |
| `seedance-1-0-pro-fast-251015` | Yes | No | No |

### first_frame / last_frame

Both are optional and connect as inputs only. Leave both empty for text-to-video. Images that cannot be loaded are skipped without an error.

### ratio

`adaptive` lets the model choose the aspect ratio.

### duration

Choose 4 to 12 seconds. The node does not check the value against the model, so a length the model does not support is rejected by the provider.

### camerafixed

Asks the model to keep the camera still.

### generate_audio

Only used with `seedance-1-5-pro-251215`. The node ignores it for the other models.

### output_file

The file name used when saving the video. The default is `seedance_video.mp4`.
