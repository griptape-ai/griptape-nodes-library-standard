---
description: Builds an image model configuration for xAI's Grok image model, using your own xAI key.
when_to_use: Use it when you want Generate Image to call Grok instead of Griptape Cloud.
---

## When to use it

You want [Generate Image](../image/GenerateImage.md) to create images with Grok. The node generates nothing itself. It outputs `image_model_config`, which you connect to the `model` input of Generate Image.

## What it needs

- An xAI API key, stored as `GROK_API_KEY`. You can get one from [console.x.ai](https://console.x.ai).
- The node calls xAI directly. It does not use a Griptape Nodes license or a Griptape Cloud API key.
- The workflow fails validation if the key is missing.

## Example

1. Add **Grok Image**.
1. Add **Generate Image** and connect `image_model_config` to its `model` input.
1. Write a prompt such as "concept frame, abandoned launch pad at dusk" and run.

## Parameters

### model

The only choice is `grok-2-image-1212`. Workflows saved with the older names `Grok 2 Image` or `xai_grok_2_image_1212` load as this model.

The node has no size or quality settings.

### image_model_config

The configuration to connect to Generate Image.
