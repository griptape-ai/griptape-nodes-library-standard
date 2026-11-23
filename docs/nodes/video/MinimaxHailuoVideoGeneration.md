---
description: Generates a video from a prompt, and optionally first and last frame images, with the MiniMax Hailuo models.
when_to_use: Use it when you want a 6 or 10 second clip from text or from a still image with Hailuo 2.3 or Hailuo 02.
---

## When to use it

You want a short clip from a prompt, or you want to animate a still. Hailuo 2.3 Fast only works from an image. Hailuo 02 can also take a last frame, so you control where the clip ends.

The node runs through Griptape Cloud and uses your Griptape key. Turn on the API Key Provider option to use your own MiniMax key instead. It waits for the result and saves the video. `timeout` defaults to 1200 seconds. The Refresh button fetches a result after a timeout.

## Parameters

### model_id

| Model | Durations | Resolutions | Frames | `fast_pretreatment` |
|-------|-----------|-------------|--------|---------------------|
| Hailuo 2.3 | 6s, 10s | 6s: 768p, 1080p. 10s: 768p | First | Yes |
| Hailuo 02 | 6s, 10s | 6s: 768p, 1080p. 10s: 768p | First and last | Yes |
| Hailuo 2.3 Fast | 6s | 720p, 1080p | First, required | No |

`last_frame_image` and `fast_pretreatment` only show for models that support them.

### prompt

Required. An empty prompt stops the run with an error, including for image-to-video.

### first_frame_image / last_frame_image

Images must be JPG, PNG or WebP, under 20 MB, with a short edge over 300 px and an aspect ratio between 2:5 and 5:2. Connect them as inputs.

### duration / resolution

The dropdowns list every value, not only the ones your model supports. The node checks the pair before it submits and tells you which resolutions work for that duration. The default resolution is 768p, so switch to 720p or 1080p when you select Hailuo 2.3 Fast.

### prompt_optimizer

Lets the service rewrite your prompt before generating.

### fast_pretreatment

Reduces the time the prompt optimization takes. Hailuo 2.3 and 02 only.
