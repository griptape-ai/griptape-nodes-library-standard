---
description: Generates or edits an image with Black Forest Labs FLUX.1 Kontext Pro from a prompt and an optional input image.
when_to_use: Use it when you want FLUX.1 Kontext Pro for text-to-image, or to change an existing image with a text instruction.
---

## When to use it

Write a prompt to generate an image. Connect `input_image` to start from an existing image instead. The only model offered is FLUX.1 Kontext Pro. For newer models, see [FLUX.2 Image Generation](Flux2ImageGeneration.md) and [FLUX 3 Image Generation](Flux3ImageGeneration.md).

It runs through the Griptape model proxy. You need a Griptape Cloud credential, or you can switch `api_key_provider` to use your own Black Forest Labs key (`BFL_API_KEY` in Settings, Secrets). The run waits up to `timeout` seconds, 600 by default.

## Parameters

### aspect_ratio

You choose the ratio, not the pixel size. Every output is about 1 megapixel.

### seed

Use the same seed with the same inputs to repeat a result. Turn on `randomize_seed` to get a new seed each run.

### prompt_upsampling

Lets the service expand your prompt before generating.

### safety_tolerance

Sets how strict content moderation is. The default is `least restrictive`. If the content is moderated, the run fails and the moderation reasons are shown in `result_details`.

### image_url

The generated image, saved with the name and format set by `output_file` and `output_format`.
