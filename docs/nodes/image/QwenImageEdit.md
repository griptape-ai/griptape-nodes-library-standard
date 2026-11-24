---
description: Edits one to three images from a text instruction with Alibaba's Qwen-Image model.
when_to_use: Use it when you want to change an existing image by describing the edit, or combine details from a few images.
---

## When to use it

Connect up to three images and describe the change. When you use more than one, refer to them as "Image 1", "Image 2" and so on in the instruction. To generate from text alone, use [Qwen Image Generation](QwenImageGeneration.md).

## What it needs

- A Griptape Nodes license or a Griptape Cloud API key (`GT_CLOUD_API_KEY`). To use your own Alibaba Cloud Model Studio key (`DASHSCOPE_API_KEY`) instead, set **API Key Provider** to **Customer**.
- The node waits for the result. `timeout` sets how long, 600 seconds by default, or 0 for no limit. If a run times out, the **Refresh** button on the node fetches the result later.
- One to three input images, each JPG, PNG, BMP, TIFF or WEBP, 1 to 8000 pixels on each side (384 to 2048 gives the best results) and under 10 MB. The node refuses to run if you connect more than three.
- Workflows saved with the older Qwen Image Edit models are migrated to `qwen-image-3.0`.

## Parameters

### images

At least one and at most three images. See What it needs for format and size expectations. However many you connect, the node makes one edited image, in `image_url`.

### negative_prompt

Describes what to keep out of the result. It is only sent when it isn't empty.

### seed

Use the same seed with the same inputs to repeat a result. Turn on `randomize_seed` to get a new seed each run.

### watermark

Adds a "Qwen-Image" mark in the bottom-right corner. Off by default.
