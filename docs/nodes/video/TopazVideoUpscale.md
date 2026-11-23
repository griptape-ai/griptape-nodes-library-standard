---
description: Upscales a video to a larger resolution with Topaz Starlight Precise or Astra 2 generative models.
when_to_use: Use it when you have a low-resolution clip, or an AI-generated one, and want a sharper, larger version with added detail.
---

## When to use it

You want a bigger, cleaner version of an existing video. The Topaz models add new detail instead of only stretching pixels. Starlight Precise stays close to the source. Astra 2 has creative controls that let it invent more.

The node runs through Griptape Cloud and uses your Griptape key. Turn on the API Key Provider option to use your own Topaz key instead. The video is uploaded to Griptape Cloud storage so Topaz can fetch it. The upload is deleted when the run ends. Upscaling can take a while. Use `timeout` (600 seconds by default) to change how long the node waits. The Refresh button fetches the result after a timeout.

## Parameters

### model

| Model | Extra controls |
|-------|----------------|
| Starlight Precise 2.6 (default) | `sharpness` |
| Starlight Precise 2.5 | None |
| Astra 2 | `prompt`, `creativity`, `sharp`, `realism` |

Each control shows only for the model that uses it.

### Frame limits

The node counts the source frames and stops before upload if the clip is too long. Trim or split the video first with [Trim Video](TrimVideo.md) or [Split Video](SplitVideo.md).

- Starlight: up to 9000 frames.
- Astra 2: up to 9000 frames, or 450 if you set a `prompt`. That is about 15 seconds at 30 fps.

### input video

The source must be mp4, mov or mkv. Other types such as webm or avi are rejected. A file with no extension is treated as mp4.

### prompt

Astra 2 only. Describes the detail to generate. Leave it empty to keep the 9000-frame limit.

### sharpness

Starlight Precise 2.6 only. 1.0 is soft and natural. 5.0 is the sharpest. The default is 5.0.

### creativity / sharp / realism

Astra 2 only, all from 0.0 to 1.0.

- `creativity`: 0.0 stays faithful to the source. 1.0 invents the most detail.
- `sharp`: sharpening applied before enhancement. 0.0 blurs, 0.5 leaves it unchanged, 1.0 sharpens.
- `realism`: pushes the generated detail toward photorealism.

### resize_mode

How the output size is worked out.

| Mode | Uses | Result |
|------|------|--------|
| percentage (default) | `percentage`, 100 to 400 | Source size scaled by that percentage. 200 doubles it. |
| width | `target_size` | Output width. Height keeps the aspect ratio. |
| height | `target_size` | Output height. Width keeps the aspect ratio. |
| width and height | `target_width`, `target_height` | Exact size. The aspect ratio is not preserved. |

Sizes are rounded down to an even number. Starlight models cannot exceed 3840 x 2160 pixels. The node stops with an error instead of submitting a larger size. In width and height mode a badge warns you as soon as the pair is too big. The other modes are checked when the node runs, because the source size is not known earlier. Astra 2 has no limit in the node.

### video_output

The upscaled video, saved to the project. Change the file name with `output_file`.
