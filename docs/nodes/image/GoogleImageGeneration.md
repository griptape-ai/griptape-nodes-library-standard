---
description: Generates or edits images with Google Gemini image models (Nano Banana Pro and Nano Banana 2), with optional reference images and search grounding.
when_to_use: Use it when you want Gemini image results, including edits and compositions guided by up to 14 reference images.
---

## When to use it

Write a prompt to generate an image. Add reference images to edit one or to combine several. Two models are offered. `gemini-3-pro-image` (Nano Banana Pro) is the default. `gemini-3.1-flash-image` (Nano Banana 2) adds smaller and wider formats and image search.

It runs through the Griptape model proxy. You need a Griptape Cloud credential, or you can switch `api_key_provider` to use your own Google Cloud service account JSON (`GOOGLE_SERVICE_ACCOUNT_JSON` in Settings, Secrets). The run waits up to `timeout` seconds, 600 by default.

## Parameters

### input_images

Up to 14 reference images.

### aspect_ratio

The choices depend on the model. Nano Banana 2 adds `1:4`, `1:8`, `4:1` and `8:1`.

### image_size

`1K`, `2K` or `4K`. Nano Banana 2 also offers `512`.

### auto_image_resize

On by default. Reference images over 7 MB are shrunk to fit. Turn it off to fail instead.

### temperature and top_p

Control how random the output is. Lower values give more predictable results. The defaults are 1.0 and 0.95.

### use_google_search

Lets the model ground its response with Google Search.

### use_google_image_search

Nano Banana 2 only. Lets the model ground its response with Google Image Search. It can be combined with `use_google_search`.

### image

The first generated image.

### all_images

Every generated image as a list. It is hidden by default.

### text

Any text the model returned alongside the images. It is hidden by default. It is also added to `result_details`.
