---
description: Animates an image into a video using LTX models, with an optional last frame to move toward.
when_to_use: Use it when you have a still image and want a 6 to 20 second clip that starts from it.
---

## When to use it

You have a picture and want it to move. Connect it to `image`, describe the motion in `prompt`, and run the node. Connect a second image to `last_frame` and the clip moves from the first image toward the second. For video from text alone, use [LTX Text to Video Generation](LTXTextToVideoGeneration.md).

The node runs through the Griptape model proxy and needs a Griptape Cloud API key. Switch `api_key_provider` to **Customer** to use your own LTX Studio key instead, stored in the `LTX_API_KEY` secret. `timeout` sets how long the node waits, in seconds, and defaults to 1200. 0 means no limit. If the wait ends first, the Refresh button in the Status group retrieves a finished result.

## Parameters

### image

The starting image. It is required. The node stops with a message if it is missing.

### last_frame

Optional. The clip interpolates from `image` to this frame.

### model, resolution, fps and duration

These work as in [LTX Text to Video Generation](LTXTextToVideoGeneration.md). The default model is `ltx-2-5-fast`. The allowed resolution, frame rate and duration combinations depend on the model, and an invalid one stops the node before anything is sent. The model table on that page applies here.

### camera_motion

`static`, `dolly_in`, `dolly_out`, `dolly_left`, `dolly_right`, `jib_up`, `jib_down` or `focus_shift`. The default is `static`.

### video_url

The generated video. It is also saved to the project at `output_file`.
