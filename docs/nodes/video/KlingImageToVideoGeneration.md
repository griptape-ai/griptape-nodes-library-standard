---
description: Generates a video from a start frame, and optionally an end frame, using Kling AI models.
when_to_use: Use it when you have a still image and want a clip that starts from it, or moves between a first and a last frame.
---

## When to use it

You have an image and want it animated. Connect it to `image` and describe the motion in `prompt`. Add an end frame to control where the clip lands. Use [Kling Text to Video Generation](KlingTextToVideoGeneration.md) when you have no image.

The node runs through the Griptape model proxy and needs a Griptape Cloud API key. Switch `api_key_provider` to **Customer** to use your own Kling key instead, stored in the `KLING_API_KEY` secret. `timeout` sets how long the node waits, in seconds, and 0 means no limit. If the wait ends first, the Refresh button in the Status group retrieves a finished result.

## Parameters

### model_name

The default is `kling-v3`. The controls shown depend on the model.

| Model | Duration | Notes |
| --- | --- | --- |
| `kling-4.0` | 3 to 30 s | `resolution` 720p or 1080p. Start, middle and end frames. |
| `kling-4.0-flash` | 3 to 20 s | 720p only. Start frame only. |
| `kling-v3` | 3 to 15 s | `mode` `std`, `pro` or `4k`. Multi-shot. |
| `kling-v2-6` | 5 or 10 s | `mode` `std` or `pro`. Optional `sound`. |
| `kling-v2-5-turbo` | 5 or 10 s | `mode` `std` or `pro`. |

Workflows saved with older Kling models are switched to `kling-v2-6`, because Kling no longer offers them.

### image and image_tail

`image` is the start frame and `image_tail` is the end frame. On Kling v3, v2.6 and v2.5 Turbo you need at least one of them. Kling 4.0 needs the start frame.

An end frame needs `pro` mode. The node stops with a message in `std` mode, and also in `4k`. Kling 4.0 Flash takes no end frame.

### middle_frames

Kling 4.0 only. Keyframes between the start and end frame, in order. In the prompt, the start frame is `@frame_1`, the first middle frame is `@frame_2`, and so on. Kling 4.0 accepts up to 10 keyframes in total.

### prompt

A prompt is required. It can be up to 2500 characters on v3, v2.6 and v2.5 Turbo, except for v3 in custom multi-shot mode.

### multi_shot, shot_type and shots

Kling v3 only. With `multi_shot` on, `customize` lets you write each shot in the `shots` editor with its own prompt and a whole-second duration, up to 6 shots. The total becomes the video length and `prompt` and `duration` are hidden. `intelligence` takes one `prompt` and Kling splits it into shots.

### sound

Kling v2.6 only. It is only shown in `pro` mode and is reset to `off` in `std` mode.

### dynamic_masks

A JSON list that configures Kling's dynamic brush. It must be valid JSON, and any masks inside it must be URLs or base64. The node stops with a message if the JSON does not parse. `static_mask` takes an image for the static brush area. The mask inputs are hidden on Kling 4.0.

### kling_video_id

Kling's own ID for the generated video. Pass it to [Kling Video Extension](KlingVideoExtension.md) to continue the clip.
