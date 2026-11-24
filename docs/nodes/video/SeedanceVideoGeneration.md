---
description: Deprecated. Generates a video with the Seedance 1.x models, which Griptape Cloud no longer serves.
when_to_use: "Don't use it for new work. Use Seedance 2.0 Video Generation or Seedance 2.5 Video Generation instead."
---

## When to use it

> [!WARNING]
> This node is deprecated. BytePlus no longer serves Seedance 1.0 Pro, 1.0 Pro Fast or 1.5 Pro through Griptape Cloud. Use [Seedance 2.0 Video Generation](Seedance20VideoGeneration.md) or [Seedance 2.5 Video Generation](Seedance25VideoGeneration.md) for new work.

The node is kept so that saved workflows still open. It turns a prompt into a video, or animates from a still. A first frame starts the clip from an image, and a last frame also controls where it ends. For reference images, reference videos, audio or editing, use the 2.0 or 2.5 node.

The node runs through Griptape Cloud and uses your Griptape key. Turn on the API Key Provider option to use your own ByteDance Seed key instead. The node waits for the result and saves the video. Use the `timeout` parameter (600 seconds by default) to change how long it waits. If a run times out, the Refresh button on the node fetches the result later.

## Parameters

### model_id

These models are no longer served through Griptape Cloud. The model changes which inputs appear.

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
