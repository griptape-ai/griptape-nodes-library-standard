## What is it?

The Agent node lets you configure an AI Agent with customizable capabilities like tools and rulesets. This node can create an Agent for immediate use given it's own prompt, or can be passed to other nodes' "agent" inputs.

## When would I use it?

Use this node when you want to:

- Create a configurable AI Agent from scratch
- Set up an Agent with specific tools and rulesets
- Prepare an Agent that can be reused across your workflow
- Get immediate responses from your agent using a custom prompt

## How to use it

### Basic Setup

1. Add the Agent to your workflow
1. Configure the agent's capabilities (tools and rulesets)

### Parameters

- **agent**: An existing Agent configuration (optional). If specified, it will use the existing Agent when prompting.
- **provider**: The AI provider to use (e.g., Griptape Cloud, Ollama, LM Studio, or a custom endpoint). See [AI Providers](https://docs.griptapenodes.com/en/stable/guides/agent/providers/) for setup instructions.
- **prompt model**: The specific model from the selected provider. You can also connect a prompt model config from a node such as [Griptape Cloud Prompt](../config/GriptapeCloudPrompt.md), [Open Ai Prompt](../config/OpenAiPrompt.md) or [Anthropic Prompt](../config/AnthropicPrompt.md). The connected configuration replaces the dropdown.
- **prompt**: The instructions or question you want to ask the Agent.
- **additional_context**: String or key-value pairs providing additional context to the Agent.
- **tools**: Capabilities you want to give your Agent.
- **Behaviors (Rulesets)** (`rulesets`): Rules that tell your Agent what it can and cannot do. Connect Ruleset nodes, or a Text Input node to write the behavior as plain text. You can also type a behavior straight into the parameter.
- **output_schema**: A JSON Schema template that defines the exact format you want the Agent's response to follow (optional).

### Outputs

- **output**: The text response from your agent (if a prompt was provided). It is a JSON object when an `output_schema` is connected.
- **agent**: The configured agent object, which can be connected to other nodes

## Example

Imagine you want an Agent that drafts a dailies note for a shot, based on additional_context:

1. Add a KeyValuePair
1. Set the "key" to "shot" and "value" to "SH010_comp"
1. Add an Agent
1. Set the Agent "prompt" to "Write a two-sentence dailies note for shot {{shot}}, asking the compositor to match the plate's grain"
1. Connect the KeyValuePair dictionary output to the Agent "additional_context" input
1. Run the workflow
1. The Agent "output" will contain a dailies note for SH010_comp

## Using Output Schemas

### What is an Output Schema?

Think of an output schema as a form you're asking the AI to fill out. Instead of getting a free-form text response, you can specify exactly what pieces of information you want and how they should be organized.

For example, instead of asking "Tell me about this shot" and getting a paragraph of text, you can ask for:

- A shot code (text)
- The number of frames left to finish (number)
- Whether the shot is approved (yes/no)
- A list of open notes (multiple text items)

The AI will then respond with structured data that matches exactly what you asked for.

### Why Use an Output Schema?

Use output schemas when you need:

- **Consistent formats**: Every response follows the same structure, making it easier to process
- **Specific data types**: Guarantee you get numbers where you need numbers, lists where you need lists, etc.
- **Easier automation**: Structured data is much easier to connect to other nodes in your workflow
- **Validation**: The AI must provide all required fields in the correct format

### How to Create an Output Schema

Use the [**Create Agent Schema**](create_agent_schema.md) node. You give it an example of the data you want, and it writes the schema for you. You don't need to write any JSON Schema yourself.

1. Add a **Create Agent Schema** node to your workflow
1. Pick an `example_template` that is close to what you need, or leave it on `Custom`
1. Edit the `example` JSON so it has the fields and value types you want back
1. Connect the node's `schema` output to your Agent's `output_schema` input
1. If the template came with rules, connect `agent_ruleset` to your Agent's **Behaviors (Rulesets)** input so the Agent follows them

The schema is built from the types in your example. A string gives a string field, a whole number gives an integer, a list gives an array, and a nested object gives a nested object. A field set to `null` becomes optional. Every other field is required.

An example that is a list at the top level, such as `["tag1", "tag2"]`, gives a schema with a single `items` array. The Agent's output then looks like `{"items": [...]}`.

### Output Schema Example

Let's say you track shots in a spreadsheet, and you want to turn a supervisor's free-text production note into one row of status data:

1. Add a **Create Agent Schema** node, leave `example_template` on `Custom`, and paste this into `example`:

    ```json
    {
      "shot_code": "SH010",
      "department": "comp",
      "status": "in_progress",
      "artist": "Maria",
      "frames_remaining": 48,
      "blockers": ["waiting on final plate", "roto not approved"]
    }
    ```

1. Connect the node's `schema` output to your Agent's `output_schema` input

1. Optionally add rules in `ruleset_example`, such as "status must be one of: pending, in_progress, review, approved". Connect `agent_ruleset` to the Agent's **Behaviors (Rulesets)** input so the Agent follows them.

1. Set your Agent's prompt: "Extract the shot status from this production note: [note text]"

1. The Agent will now respond with structured data instead of plain text, containing exactly those fields. You can connect `frames_remaining` or `status` to other nodes, for example to a [Compare Numbers](../execution/CompareNumbers.md) node that flags shots with frames still to do.

1. The generated schema will look like this:

```json
{
  "type": "object",
  "properties": {
    "shot_code": { "type": "string" },
    "department": { "type": "string" },
    "status": { "type": "string" },
    "artist": { "type": "string" },
    "frames_remaining": { "type": "integer" },
    "blockers": {
      "type": "array",
      "items": { "type": "string" }
    }
  },
  "required": ["shot_code", "department", "status", "artist", "frames_remaining", "blockers"]
}
```

### What Changes When Using a Schema?

- **Output type**: The Agent's output changes from plain text to structured data (JSON format)
- **Validation**: If the AI cannot provide data in the requested format, it will try again or return an error

## Important Notes

- If you don't provide a prompt, the node will create the agent without running it and the output will contain exactly "Agent created."
- Tools and rulesets can be provided as individual items or as lists
- The additional_context parameter allows you to provide additional_context to the agent as a string or dictionary of key/value pairs
- By default the node uses Griptape Cloud, which requires a valid `GT_CLOUD_API_KEY`. You can switch to a local or custom provider using the **provider** parameter — see [AI Providers](https://docs.griptapenodes.com/en/stable/guides/agent/providers/).
- When you pass an Agent from one node to another using the agent input/output pins, the conversation memory is maintained, which means:
    - The Agent "remembers" previous interactions in the same flow
    - Context from previous prompts influences how the Agent interprets new prompts
    - You can build multi-turn conversations across multiple nodes
    - The Agent can reference information provided in earlier steps of your workflow
- To get structured output, connect a [**Create Agent Schema**](create_agent_schema.md) node to the `output_schema` input. See [How to Create an Output Schema](#how-to-create-an-output-schema).

## Controlling Agent Behavior in Workflows

To shape how an Agent behaves in a workflow, use the **Behaviors (Rulesets)** input (the `rulesets` parameter). It takes either of these:

- **Ruleset nodes**, for rules you want to reuse across Agents.
- A **Text Input** node, to write the behavior as plain text. You can also type it straight into the parameter.

Use them to define what the Agent can and cannot do, set its persona, or constrain its output style. For example, "Always name shots with the show prefix and a three-digit sequence, like ABC_010_0020" or "Keep notes under 50 words and address the artist directly".

> [!NOTE]
> **Skills are not supported in the Agent node**
>
> The `.agents/skills` folder used by the [chat sidebar](https://docs.griptapenodes.com/en/stable/guides/agent/skills/) does not apply to Agent nodes in workflows. Use a **Ruleset node** or a **Text Input** node connected to the Agent's **Behaviors (Rulesets)** input to control agentic behavior instead.

## Common Issues

- **No provider configured**: If no provider is set, the node uses Griptape Cloud by default, which requires a valid `GT_CLOUD_API_KEY` or a Griptape Nodes license. See [AI Providers](https://docs.griptapenodes.com/en/stable/guides/agent/providers/) to add Ollama, LM Studio, or a custom endpoint.
- **Missing credential**: The workflow fails validation when the Agent would use Griptape Cloud and neither a `GT_CLOUD_API_KEY` nor a Griptape Nodes license is available. A connected prompt model config or a connected Agent supplies its own model, so this check is skipped.
