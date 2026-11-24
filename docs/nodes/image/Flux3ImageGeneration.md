---
description: Generates or edits images with Black Forest Labs FLUX 3, using a resolution tier or exact size and up to ten reference images.
when_to_use: Use it when you want FLUX 3 for text-to-image, editing, or combining several reference images into one result.
---

## When to use it

Write a prompt to generate an image. Add reference images to edit one or to compose a result from several.

Two models are offered: `flux-3-image` (the default) and `flux-3-image-klein`. Some settings only apply to one of them.

## What it needs

- A Griptape Nodes license or a Griptape Cloud API key (`GT_CLOUD_API_KEY`). To use your own Black Forest Labs key (`BFL_API_KEY`) instead, set **API Key Provider** to **Customer**.
- The node waits for the result. `timeout` sets how long, 600 seconds by default, or 0 for no limit. If a run times out, the **Refresh** button on the node fetches the result later.

## Parameters

### input_images

Up to 10 reference images. Running with more fails validation.

### size_mode

`Resolution Tier` uses `aspect_ratio` and `resolution`. `Custom Dimensions` uses `width` and `height`. Only the settings for the chosen mode are shown.

### aspect_ratio

`auto` lets the model pick from the prompt or the first reference image.

### resolution

The tiers are `768sq`, `1k`, `1.5k`, `2k` and `4k`. `match_input` keeps the size of the reference image and fails validation if no image is connected.

### width and height

Shown in `Custom Dimensions` mode. From 256 to 6256 pixels for `flux-3-image`, and from 64 to 6256 for Klein. Width times height can't exceed 16,822,272 pixels. With reference images on `flux-3-image`, the limit is 4,194,304 pixels.

### grounding

`flux-3-image` only. Lets the model use web and image search to ground the prompt. On by default.

### safety_tolerance

Sets how strict content moderation is. The default is `moderate`.

### seed, output_format and steps

Klein only. `steps` runs from 1 to 16 and defaults to 16. Use the same `seed` to repeat a result, or turn on `randomize_seed`. `flux-3-image` doesn't expose these settings.
