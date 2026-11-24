---
description: Builds an image model configuration for the OpenAI image models offered through Griptape Cloud.
when_to_use: Use it when you want to set image size and quality once and pass them to Generate Image through a connection.
---

## When to use it

You want to pass a Griptape Cloud image model, size and quality to [Generate Image](../image/GenerateImage.md) through a connection. The node generates nothing itself. It outputs `image_model_config`, which you connect to the `model` input of Generate Image.

Generate Image already has its own `model` and `image_size` settings that use Griptape Cloud. Use this node when you want `quality`, which Generate Image does not offer, or when you want to share one configuration across several nodes.

## What it needs

- A Griptape Nodes license or a Griptape Cloud API key (`GT_CLOUD_API_KEY`). You do not need an OpenAI key.
- The workflow fails validation if neither is available.

## Example

1. Add **Griptape Cloud Image** and set `model` to `gpt-image-1.5`.
1. Set `image_size` to `1536x1024` for a wide frame and `quality` to `high`.
1. Add **Generate Image** and connect `image_model_config` to its `model` input.
1. Run with a prompt such as "matte painting of a canyon at golden hour".

## Parameters

### model

`gpt-image-1-mini` (default) or `gpt-image-1.5`. Workflows saved with `dall-e-3` load as `gpt-image-1-mini`.

### image_size

`1024x1024` (default), `1536x1024` or `1024x1536`.

### quality

`low`, `medium` (default) or `high`.

### image_model_config

The configuration to connect to Generate Image. When connected, Generate Image hides its own `image_size` and uses the settings from this node.
