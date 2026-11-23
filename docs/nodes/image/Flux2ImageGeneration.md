---
description: Generates or edits images with Black Forest Labs FLUX.2 models at an exact pixel size, using up to eight reference images.
when_to_use: Use it when you want FLUX.2 results at a specific width and height, or want reference images to guide the output.
---

## When to use it

FLUX.2 offers five models: pro, flex, max, klein 4B and klein 9B. The default is pro. Set the exact width and height, and optionally add reference images for image-to-image work.

It runs through the Griptape model proxy. You need a Griptape Cloud credential, or you can switch `api_key_provider` to use your own Black Forest Labs key (`BFL_API_KEY` in Settings, Secrets). The run waits up to `timeout` seconds, 600 by default.

## Parameters

### input_images

Up to eight reference images are used. Any beyond the eighth are ignored.

### width

In pixels, a multiple of 16. The total image size can't exceed 4 megapixels.

### height

In pixels, a multiple of 16. The total image size can't exceed 4 megapixels.

### force_output_dimension

Rounds `width` and `height` to the nearest multiple of 16 before sending. Turn it on if you type sizes by hand. FLUX.2 [flex] needs it to work correctly.

### seed

Use the same seed with the same inputs to repeat a result. Turn on `randomize_seed` to get a new seed each run.

### safety_tolerance

Sets how strict content moderation is. The default is `least restrictive`.

### steps

Only shown for the flex model. More steps give more detail and take longer. Maximum 50.

### guidance

Only shown for the flex model. Higher values follow the prompt more closely. The range is 1.5 to 10 and the default is 4.5.
