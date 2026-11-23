---
description: Generates a text description of one or more images with a vision-capable model.
when_to_use: Use it when you need an image turned into text, such as a lookdev note about a reference frame or alt text.
---

## When to use it

You want to turn what is in an image into text for notes, accessibility or later nodes. Typical uses are alt text, extracting information from photographs or diagrams, and describing a reference frame for a lookdev note.

With the Griptape Cloud provider, the node needs a Griptape Cloud credential: your `GT_CLOUD_API_KEY`, or a Griptape Nodes License. The check is skipped when an agent, a connected prompt model config or another provider supplies the model. Other providers use the key configured for that provider. Your organization's model policy can deny a model, and the run then fails before any request is sent.

## Example

You have a reference frame for a shot and want its lighting described for a lookdev note:

1. Add a **Describe Image** node and connect the frame from a **Load Image** node to `images`.
1. In `prompt`, type "Describe the key light direction, color temperature, shadow softness and any visible lens artifacts".
1. Run the node. The description is in `output`.

Be specific about what you want. For color, try "Describe the color palette, the color of light and shadow, the saturation and value". For a character, try "Describe the character's gender, age, clothing, posture, demeanor, and actions".

### Structured output

To get JSON in a specific shape:

1. Add a [Create Agent Schema](../agents/create_agent_schema.md) node and give it an example JSON object of the shape you want back.
1. Connect its `schema` output to `output_schema` on **Describe Image**.
1. Run **Describe Image**. `output` is JSON, and the model is asked to follow the schema. The node does not check the result against the schema. In an offline test, output that omitted required fields and had the wrong types was passed through, so validate it downstream if the shape matters.

## Parameters

### agent

Optional. Connect an [Agent](../agents/Agent.md) and the node uses its model and configuration. `model_provider` and `model` are hidden while an agent is connected.

### model_provider

Where the model runs. The list comes from the providers configured in the engine, with Griptape Cloud first. Use the refresh button to update it.

### model

A vision-capable model from the selected provider. For Griptape Cloud the list contains only models that accept images, and the default is the first one. You can also connect a prompt model config, such as [Griptape Cloud Prompt](../config/GriptapeCloudPrompt.md), which replaces the dropdown.

### images

One or more images to describe. Add rows to send several images in one request. Each can be an image or a path string. At least one is required.

The old `image` input is deprecated and hidden. Workflows that used it are moved onto the `images` list when they load.

### prompt

Instructions for how to describe the image. If empty, the node uses "Describe the image".

### description_only

On by default. Adds "Output image description only." to the prompt so the model returns the description without conversation.

### output_schema

Optional JSON Schema for structured output, as an object or a JSON string. It is an input only. When it is connected, `output` becomes JSON and the schema is sent to the model as the required output shape. The result is not checked against the schema. A string that is not valid JSON fails the run, and so does a value that is not a JSON object.

### output

The description as text, or JSON when `output_schema` is connected.

### agent (output)

The agent used for the run, with this exchange added to its history as text. It can be connected to other nodes. The image itself is not stored in the history, so downstream agents do not resend it.

## Common issues

- **Missing API key**: Set `GT_CLOUD_API_KEY` in Settings, or choose another provider.
- **"An image is required"**: No image is connected to `images`, or every row is empty.
- **Unable to parse output schema**: The `output_schema` string is not valid JSON. Use the Create Agent Schema node to generate one.
- **Poor description quality**: Make the prompt more specific, or try a different model.
