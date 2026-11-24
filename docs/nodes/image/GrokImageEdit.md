---
description: Edits an image from a text instruction with xAI's Grok Imagine model, using up to five source images.
when_to_use: Use it when you want to change or restyle an existing image by describing the edit.
---

## When to use it

Connect an image, describe the change and get up to 10 edited versions back. You can add up to four more source images, for example to bring in a second object or a style reference. To generate from text alone, use [Grok Image Generation](GrokImageGeneration.md).

## What it needs

- A Griptape Nodes license or a Griptape Cloud API key (`GT_CLOUD_API_KEY`). To use your own xAI key (`GROK_API_KEY`) instead, set **API Key Provider** to **Customer**.
- The node waits for the result. `timeout` sets how long, 600 seconds by default, or 0 for no limit. If a run times out, the **Refresh** button on the node fetches the result later.

## Parameters

### image

Required. The main image to edit.

### additional_images

Up to four extra source images sent with `image`, so five in total.

### n

How many edited images to generate, 1 to 10. Each image gets its own output.

### quality

Currently has no effect on the result.

### resolution

Only `1k` is currently supported, even though `2k` is listed.

### image_url

The first edited image. When `n` is greater than 1, `image_url_2` through `image_url_10` appear. If one image can't be downloaded, its slot stays empty and the others are kept.

### provider_response

The provider's response, unchanged. It is hidden by default.
