---
description: Upscales an MP4 clip by 1.5x to 3x using FLUX 3 Video Upscale, with an option to add detail.
when_to_use: Use it when you want a higher-resolution version of a short clip and are happy for the model to add detail.
---

## When to use it

You have a clip of up to 20 seconds and want it larger. Connect it to `input_video`, set `upscale_factor`, and run the node.

The input must be an MP4 of 20 seconds or less and no larger than 2560 x 1440. The node checks the format and length before it submits. To prepare a clip, use [Resize Video](ResizeVideo.md) or a trim node.

The node runs through the Griptape model proxy and needs a Griptape Cloud API key. Switch `api_key_provider` to **Customer** to use your own Black Forest Labs key instead, stored in the `BFL_API_KEY` secret. `timeout` sets how long the node waits, in seconds, and 0 means no limit. If the wait ends first, the Refresh button in the Status group retrieves a finished result.

## Parameters

### upscale_factor

The output size relative to the source, from 1.5 to 3.0. The default is 2.0. The tooltip notes that the output frame is capped at 13.75 megapixels.

### creativity

`precise` stays faithful to the source. `creative` lets the model enhance detail. The default is `creative`.

### prompt

Optional. A short description of what is in the clip, to steer the added detail. Leave it empty for a neutral upscale.

### safety_tolerance

`least restrictive`, `moderate` or `most restrictive`. The default is `moderate`.
