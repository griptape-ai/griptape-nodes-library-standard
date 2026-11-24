---
description: Upscales an MP4 clip by 1.5x to 3x using FLUX 3 Video Upscale, with an option to add detail.
when_to_use: Use it when you want a higher-resolution version of a short clip and are happy for the model to add detail.
---

## When to use it

You have a clip of up to 20 seconds and want it larger. Connect it to `input_video`, set `upscale_factor`, and run the node. To prepare a clip, use [Resize Video](ResizeVideo.md) or a trim node.

## What it needs

- A Griptape Nodes license or a Griptape Cloud API key (`GT_CLOUD_API_KEY`). To use your own Black Forest Labs key (`BFL_API_KEY`) instead, set **API Key Provider** to **Customer**.
- The node waits for the result. `timeout` sets how long, 600 seconds by default, or 0 for no limit. If a run times out, the **Refresh** button on the node fetches the result later.
- The input must be an MP4 of 20 seconds or less, no larger than 2560 x 1440, and no more than 50 MB. A clip that is not an MP4, or is too long, stops the run with a message. The clip is sent inline with the request.

## Parameters

### upscale_factor

The output size relative to the source, from 1.5 to 3.0. The default is 2.0. The output frame is capped at 13.75 megapixels.

### creativity

`precise` stays faithful to the source. `creative` lets the model enhance detail. The default is `creative`.

### prompt

Optional. A short description of what is in the clip, to steer the added detail. Leave it empty for a neutral upscale.

### safety_tolerance

`least restrictive`, `moderate` or `most restrictive`. The default is `moderate`.
