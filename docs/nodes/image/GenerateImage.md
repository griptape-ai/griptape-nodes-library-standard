---
description: Generates an image from a text prompt with an OpenAI GPT Image model, through Griptape Cloud.
when_to_use: Use it when you want a quick image from a description, such as a concept frame, a texture reference or a placeholder plate.
---

## When to use it

You have a prompt and want an image. The node runs through Griptape Cloud and uses your Griptape credential (`GT_CLOUD_API_KEY` or a Griptape Nodes license). It does not offer an API Key Provider option.

Connect an [Agent](../agents/Agent.md) to `agent` to give the image model the agent's conversation as context. The image model has no memory of its own, so the node prefixes your prompt with that history. After the run, the exchange is added to the agent's conversation as a short note that an image was created, not the image itself.

## Example

1. Add **Generate Image**.
1. Set `prompt` to "Wide establishing shot of a rain-soaked harbor at dusk, cinematic lighting".
1. Pick `1536x1024` for `image_size` to get a landscape frame.
1. Run the workflow and connect `output` to a [Save Image](SaveImage.md) node to write it to disk.

## Parameters

### model

Shown as `image model` in the editor. The choices are `gpt-image-1-mini` (the default) and `gpt-image-1.5`. Workflows saved with `dall-e-3` or `gpt-image-1` load as `gpt-image-1-mini`. You can also connect an Image Generation Driver node here. When one is connected, its settings replace this dropdown and `image_size` is hidden.

### prompt

The description of the image. It is required unless the input is connected.

### image_size

`1024x1024` (the default), `1536x1024` or `1024x1536`.

### enhance_prompt

Off by default. When on, the prompt is first rewritten by a chat model (`gpt-4o`, or the connected agent's model) with instructions to add depth of field, time of day, vignetting and color palette detail. The enhanced text is then sent to the image model. This is an extra model call, so it adds time and cost, and the result may drift from what you wrote.

### output

The generated image, saved with the name in `output_file` (default `generated.png`).

### include_details and logs

Hidden group. `logs` shows whether the prompt was enhanced and when generation started and finished.
