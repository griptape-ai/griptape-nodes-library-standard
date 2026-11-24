---
description: Generates or edits images with Black Forest Labs FLUX.2 models at an exact pixel size, using up to eight reference images.
when_to_use: Use it when you want FLUX.2 results at a specific width and height, or want reference images to guide the output.
---

## When to use it

Use it when you need FLUX.2 output at an exact width and height. Add reference images for image-to-image work.

It offers five models: pro (the default), flex, max, klein 4B and klein 9B.

## What it needs

- A Griptape Nodes license or a Griptape Cloud API key (`GT_CLOUD_API_KEY`). To use your own Black Forest Labs key (`BFL_API_KEY`) instead, set **API Key Provider** to **Customer**.
- The node waits for the result. `timeout` sets how long, 600 seconds by default. If a run times out, the **Refresh** button on the node fetches the result later.

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
