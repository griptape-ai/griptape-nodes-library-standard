---
description: Generates a video from a text prompt using Kling AI models, with multi-shot support on Kling v3.
when_to_use: Use it when you want a clip from a description and want to choose between the Kling 4.0, v3, v2.6 and v2.5 Turbo models.
---

## When to use it

You want a generated clip from text. Write a `prompt`, pick a model, and run the node. Use [Kling Image to Video Generation](KlingImageToVideoGeneration.md) to start from a picture.

## What it needs

- A Griptape Nodes license or a Griptape Cloud API key (`GT_CLOUD_API_KEY`). To use your own Kling AI key (`KLING_API_KEY`) instead, set **API Key Provider** to **Customer**.
- The node waits for the result. `timeout` sets how long, 600 seconds by default. If a run times out, the **Refresh** button on the node fetches the result later.
- Set `timeout` to 0 to wait with no limit.

## Parameters

### model_name

The default is `kling-v3`. The controls shown depend on the model.

| Model | Duration | Notes |
| --- | --- | --- |
| `kling-4.0` | 3 to 30 s | `resolution` 720p or 1080p. No `mode`, `cfg_scale`, `negative_prompt` or `sound`. |
| `kling-4.0-flash` | 3 to 20 s | 720p only. Same controls as `kling-4.0`. |
| `kling-v3` | 3 to 15 s | `mode` `std`, `pro` or `4k`. Multi-shot. |
| `kling-v2-6` | 5 or 10 s | `mode` `std` or `pro`. Optional `sound`. |
| `kling-v2-5-turbo` | 5 or 10 s | `mode` `std` or `pro`. |

Workflows saved with Kling v1.6, v2 Master or v2.1 Master are switched to `kling-v2-6`, because Kling no longer offers those models.

### prompt

Up to 2500 characters on v3, v2.6 and v2.5 Turbo. Kling 4.0 accepts up to 26,000 English characters or 11,000 of other text, in proportion when mixed. A prompt is required, except for v3 in custom multi-shot mode.

### aspect_ratio

`16:9`, `9:16` or `1:1`. `21:9` is only available on Kling 4.0.

### multi_shot, shot_type and shots

Kling v3 only. With `multi_shot` on, `shot_type` chooses how the shots are made.

- `customize`: you write each shot in the `shots` editor, with its own prompt and a whole-second duration. At most 6 shots. The total of the shot durations becomes the video length, and `prompt` and `duration` are hidden.
- `intelligence`: you write one `prompt` and Kling splits it into shots. `duration` stays available.

### sound

Kling v2.6 only. It is only shown in `pro` mode and is reset to `off` in `std` mode.

### negative_prompt and cfg_scale

Not available on Kling 4.0. `cfg_scale` runs from 0 to 1. Higher values follow the prompt more strictly and give the model less freedom.

### kling_video_id

Kling's own ID for the generated video. Pass it to [Kling Video Extension](KlingVideoExtension.md) to continue the clip.
