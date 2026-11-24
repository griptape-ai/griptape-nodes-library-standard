---
description: Upscales a video to a larger resolution with Topaz Starlight Precise or Astra 2 generative models.
when_to_use: Use it when you have a low-resolution clip, or an AI-generated one, and want a sharper, larger version with added detail.
---

## When to use it

You want a bigger, cleaner version of an existing video. The Topaz models add new detail instead of only stretching pixels. Starlight Precise stays close to the source. Astra 2 can invent more.

## What it needs

- A Griptape Nodes license or a Griptape Cloud API key (`GT_CLOUD_API_KEY`). To use your own Topaz Labs key (`TOPAZ_API_KEY`) instead, set **API Key Provider** to **Customer**.
- The node waits for the result. `timeout` sets how long, 600 seconds by default, or 0 for no limit. If a run times out, the **Refresh** button on the node fetches the result later.
- The video is uploaded to Griptape Cloud storage as a short-lived public link, which is deleted after the run.
- The source must be mp4, mov or mkv. Other types such as webm or avi are rejected. A file with no extension is treated as mp4.
- The node stops before upload if the clip is too long. Starlight allows up to 9000 frames. Astra 2 allows 9000, or 450 (about 15 seconds at 30 fps) if you set a `prompt`. To shorten a clip first, use [Trim Video](TrimVideo.md) or [Split Video](SplitVideo.md).
- Starlight output can be at most 8,294,400 pixels in total (the area of 3840 x 2160). A larger size stops the run with an error. In width and height mode a badge warns you as soon as the pair is too big. Astra 2 has no size limit.

## Parameters

### model

| Model | Extra controls |
|-------|----------------|
| Starlight Precise 2.6 (default) | `sharpness` |
| Starlight Precise 2.5 | None |
| Astra 2 | `prompt`, `creativity`, `sharp`, `realism` |

### prompt

Astra 2 only. Describes the detail to generate. Leave it empty to keep the 9000-frame limit.

### sharpness

Starlight Precise 2.6 only. 1.0 is soft and natural. 5.0, the default, is the sharpest.

### creativity / sharp / realism

Astra 2 only, all from 0.0 to 1.0.

- `creativity`: 0.0 stays faithful to the source. 1.0 invents the most detail.
- `sharp`: sharpening before enhancement. 0.0 blurs, 0.5 leaves it unchanged, 1.0 sharpens.
- `realism`: pushes the detail toward photorealism.

### resize_mode

| Mode | Uses | Result |
|------|------|--------|
| percentage (default) | `percentage`, 100 to 400 | Source size scaled by that percentage. 200 doubles it. |
| width / height | `target_size` | That side. The other keeps the aspect ratio. |
| width and height | `target_width`, `target_height` | Exact size. The aspect ratio is not preserved. |

Sizes are rounded down to an even number.

### video_output

The upscaled video, saved to the project under `output_file`.
