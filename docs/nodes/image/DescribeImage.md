## What is it?

The DescribeImage node uses AI to generate textual descriptions of an image you provide it. It leverages vision-capable models to analyze image content and produce detailed descriptions based on your specific prompting instructions.

## When would I use it?

Use this node when you want to:

- Generate textual descriptions of images
- Extract information from visual content
- Create alt text for accessibility purposes
- Analyze the content of photographs, diagrams, or other visual media
- Convert visual information into text format for further processing

## How to use it

### Basic Setup

1. Add the DescribeImage node to your workflow
1. Connect an image source to the "image(s)" input
1. Optionally provide specific instructions in the "description prompt" field
1. Run the node to generate a description of the image

> [!NOTE]
> **Be Specific About What You Want**
>
> When creating your prompt, clearly state the elements you are interested in. For example:
>
> - For color information, try prompts like "Describe the color palette, the color of light and shadow, the saturation and value"
> - For character details, try "Describe the character's gender, age, clothing, posture, demeanor, and actions"
>
> The more precise you are with your prompts, the more likely it is you'll get the kind of output you desire.

### Parameters

- **agent**: Optional. Connect an [Agent](../agents/Agent.md) and the node uses its model and configuration. The `provider` and `prompt model` parameters are hidden while an agent is connected.
- **provider** (`model_provider`): Where the model runs. The list comes from the providers configured in the engine, with Griptape Cloud first. Use the refresh button to update it.
- **prompt model** (`model`): A vision-capable model from the selected provider. For Griptape Cloud the list contains only models that accept images, and the default is the first one in the list. You can also connect a prompt model config, such as [Griptape Cloud Prompt](../config/GriptapeCloudPrompt.md), which replaces the dropdown.
- **image(s)** (`images`): One or more images to describe. Add rows to send several images in one request. Each can be an image or a path string. At least one is required.
- **description prompt** (`prompt`): Instructions for how to describe the image. If left empty the node uses "Describe the image".
- **description_only** (default: on): Adds "Output image description only." to the prompt so the model returns the description without conversation.
- **output_schema**: Optional JSON Schema for structured output, as an object or a JSON string. The [Create Agent Schema](../agents/create_agent_schema.md) node builds one. It is an input only. When it is connected, `output` becomes JSON, and the run fails if the result does not match the schema.

### Outputs

- **output**: The description as text, or JSON when `output_schema` is connected.
- **agent**: The agent used for the run, with this exchange added to its history as text. It can be connected to other nodes. The image itself is not stored in the history.

The old `image` input is deprecated and hidden. Workflows that used it are moved onto the `image(s)` list when they load.

## Example

You have a reference frame for a shot and want its lighting described for a lookdev note:

1. Add a DescribeImage node to your workflow
1. Connect the frame from a Load Image node to "image(s)"
1. In "description prompt", type: "Describe the key light direction, color temperature, shadow softness and any visible lens artifacts"
1. Run the node
1. The "output" contains the description

### Structured Output Example (JSON)

If you want the output in a specific JSON shape:

1. Add a `CreateAgentSchema` node and provide an example JSON object (the shape you want back)
1. Connect `CreateAgentSchema.schema` to `DescribeImage.output_schema`
1. Run `DescribeImage`
1. The `output` will be JSON matching your schema (or the run will fail validation)

## Important Notes

- With the Griptape Cloud provider, the node needs a Griptape Cloud credential: your `GT_CLOUD_API_KEY`, or a Griptape Nodes License. The check is skipped when an agent, a connected prompt model config or another provider supplies the model. Other providers use the key configured for that provider.
- Your organization's model policy can deny a model. The run then fails before any request is sent.
- Only the text of the exchange is kept in the `agent` output, so downstream agents do not resend the image.

## Common Issues

- **Missing API Key**: Set `GT_CLOUD_API_KEY` in Settings, or choose another provider.
- **"An image is required"**: No image is connected to `image(s)`, or every row is empty.
- **Unable to parse output schema**: The `output_schema` string is not valid JSON. Use the Create Agent Schema node to generate one.
- **Poor Description Quality**: Make the prompt more specific about what you want described, or try a different model.
