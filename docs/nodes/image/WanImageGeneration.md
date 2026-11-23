---
description: Generates one to four images from a text prompt with Alibaba's Wan 2.7 image models.
when_to_use: Use it when you want text-to-image results from Wan, including up to 4K output with the Pro model.
---

## When to use it

Write a prompt, choose a size and how many images you want. The node offers Wan 2.7 Image Pro (the default) and Wan 2.7 Image. It takes a text prompt only and has no image input.

It runs through the Griptape model proxy. You need a Griptape Cloud credential, or you can switch `api_key_provider` to use your own Alibaba Cloud Model Studio key (`DASHSCOPE_API_KEY` in Settings, Secrets). The run waits up to `timeout` seconds, 600 by default.

## Parameters

### size

`1K` is 1024x1024, `2K` is 2048x2048 and `4K` is 4096x4096. The tooltip says 4K is for the Pro model only. The node doesn't block 4K on the other model, so the provider may reject it.

### n

How many images to generate, 1 to 4. Each image gets its own output, `image_url_1` through `image_url_4`. Only the outputs that received an image are shown. If one image can't be downloaded, its slot stays empty and the others are kept.

### thinking_mode

On by default. Gives better quality at the cost of a longer run.

### watermark

Adds an "AI generated" mark in the bottom-right corner. Off by default.

### seed

Leave it empty for a random seed. Set a value from 0 to 2147483647 to repeat a result. The seed is only sent when you set one.
