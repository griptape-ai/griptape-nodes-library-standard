---
description: Builds an image model configuration for OpenAI's image models, using your own OpenAI key.
when_to_use: Use it when you want Generate Image to call OpenAI directly, for example to get a transparent background on a render pass element or a JPEG with set compression.
---

## When to use it

You want [Generate Image](../image/GenerateImage.md) to use an OpenAI image model with settings the Generate Image node does not offer, such as `background`, `moderation` or `output_format`. The node generates nothing itself. It outputs `image_model_config`, which you connect to the `model` input of Generate Image.

## What it needs

- An OpenAI API key, stored as `OPENAI_API_KEY`.
- The node calls OpenAI directly. It does not use a Griptape Nodes license or a Griptape Cloud API key.
- The workflow fails validation if the key is missing.

## Example

1. Add **Open Ai Image** and set `model` to `gpt-image-1`.
1. Set `background` to `transparent` and `output_format` to `png`.
1. Add **Generate Image** and connect `image_model_config` to its `model` input.
1. Write a prompt such as "weathered brass lantern, studio lighting" and run. The result is a prop asset on a transparent background that you can drop into a comp.

## Parameters

### model

`gpt-image-1` (default), `dall-e-3` or `dall-e-2`. The other parameters change with the model. Workflows saved with older display names such as `DALL-E 3` load as the matching model id.

### image_size

For `gpt-image-1`: `1024x1024`, `1536x1024` or `1024x1536`. For `dall-e-3`: `1024x1024`, `1024x1792` or `1792x1024`. For `dall-e-2`: `256x256`, `512x512` or `1024x1024`. Changing the model resets the size to the first choice for that model.

### quality

Shown for `gpt-image-1` (`low`, `medium`, `high`; default `low`) and `dall-e-3` (`hd`, `standard`). Hidden for `dall-e-2`.

### style

`vivid` or `natural`. It is only sent to OpenAI for `dall-e-3`. The control is also shown for `gpt-image-1`, but the value is ignored there.

### background

`opaque` or `transparent`. Shown and sent only for `gpt-image-1`.

### moderation

`low` or `auto`. Shown and sent only for `gpt-image-1`.

### output_format

`png` or `jpeg`. Shown and sent only for `gpt-image-1`.

### output_compression

0 to 100, default 80. Only visible when `output_format` is `jpeg`.

### image_model_config

The configuration to connect to Generate Image. When connected, Generate Image hides its own `image_size` and uses the size from this node.
