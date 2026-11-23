---
description: Generates one or more images from a text prompt with xAI's Grok Imagine model.
when_to_use: Use it when you want text-to-image results from Grok, including several variations from one prompt.
---

## When to use it

Write a prompt, choose an aspect ratio and get up to 10 images back. To change an existing image instead, use [Grok Image Edit](GrokImageEdit.md).

It runs through the Griptape model proxy. You need a Griptape Cloud credential, or you can switch `api_key_provider` to use your own xAI key (`GROK_API_KEY` in Settings, Secrets). The run waits up to `timeout` seconds, 600 by default.

## Parameters

### n

How many images to generate, 1 to 10. Each image gets its own output.

### quality

Currently has no effect on the result.

### resolution

Only `1k` is currently supported, even though `2k` is listed.

### image_url

The first generated image. When `n` is greater than 1, `image_url_2` through `image_url_10` appear. Each output holds the image in the position the provider returned. If one image can't be downloaded, its slot stays empty and the others are kept.

### provider_response

The provider's response, unchanged. It is hidden by default.
