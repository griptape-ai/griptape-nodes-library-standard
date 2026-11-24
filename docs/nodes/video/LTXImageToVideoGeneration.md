---
description: Animates an image into a video using LTX models, with an optional last frame to move toward.
when_to_use: Use it when you have a still image and want a 6 to 20 second clip that starts from it.
---

## When to use it

You have a picture and want it to move. Connect it to `image`, describe the motion in `prompt`, and run the node. Connect a second image to `last_frame` and the clip moves from the first image toward the second. For video from text alone, use [LTX Text to Video Generation](LTXTextToVideoGeneration.md).

## What it needs

- A Griptape Nodes license or a Griptape Cloud API key (`GT_CLOUD_API_KEY`). To use your own LTX Studio key (`LTX_API_KEY`) instead, set **API Key Provider** to **Customer**.
- The node waits for the result. `timeout` sets how long, 1200 seconds by default, or 0 for no limit. If a run times out, the **Refresh** button on the node fetches the result later.

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
