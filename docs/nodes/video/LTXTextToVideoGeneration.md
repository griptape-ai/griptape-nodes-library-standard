---
description: Generates a video from a text prompt using LTX models, with optional audio and camera motion.
when_to_use: Use it when you want a 6 to 20 second clip from a prompt and want to choose the model tier, frame rate and camera move.
---

## When to use it

You want a generated clip from a description. Write a `prompt`, pick a model, resolution, duration and frame rate, and run the node. Audio is generated with the video unless you turn `generate_audio` off. To start from a picture, use [LTX Image to Video Generation](LTXImageToVideoGeneration.md).

## What it needs

- A Griptape Nodes license or a Griptape Cloud API key (`GT_CLOUD_API_KEY`). To use your own LTX Studio key (`LTX_API_KEY`) instead, set **API Key Provider** to **Customer**.
- The node waits for the result. `timeout` sets how long, 1200 seconds by default. If a run times out, the **Refresh** button on the node fetches the result later.
- Set `timeout` to 0 to wait with no limit.

## Parameters

### model

The default is `ltx-2-5-fast`. Fast models allow longer clips than Pro. `ltx-2-5-pro` stops at 1080p.

### resolution, fps and duration

Not every combination is allowed. When you change the model, resolution or fps, the `duration` choices update and a resolution the model does not offer is replaced. A combination that is still invalid stops the node with a message before anything is sent.

| Model | Resolutions | Durations in seconds |
| --- | --- | --- |
| `ltx-2-fast`, `ltx-2-3-fast` | 1920x1080 | 6 to 20 in steps of 2 at 25 fps. 6, 8, 10 at 50 fps. |
| `ltx-2-fast`, `ltx-2-3-fast` | 2560x1440, 3840x2160 | 6, 8, 10 at 25 or 50 fps. |
| `ltx-2-pro`, `ltx-2-3-pro` | 1920x1080, 2560x1440, 3840x2160. | 6, 8, 10 at 25 or 50 fps. |
| `ltx-2-5-fast` | 720p, 1080p, 1440p and 4K, landscape or portrait. | 6 to 20 in steps of 2 for 720p and 1080p at 24 or 25 fps. Otherwise 6, 8, 10. |
| `ltx-2-5-pro` | 720p and 1080p, landscape or portrait. | 6, 8, 10 at 24, 25 or 50 fps. |

Portrait sizes are the landscape sizes reversed, such as `1080x1920`. 24 fps is only offered by the 2.5 models. 48 fps is only offered by `ltx-2-5-fast`, up to 10 seconds.

### camera_motion

`static`, `dolly_in`, `dolly_out`, `dolly_left`, `dolly_right`, `jib_up`, `jib_down` or `focus_shift`. The default is `static`.

### video_url

The generated video. It is also saved to the project at `output_file`.
