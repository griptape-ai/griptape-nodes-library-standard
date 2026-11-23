---
description: Runs an AI Agent with a prompt, tools and rulesets, and passes the configured Agent on to other nodes.
when_to_use: Use it when you want a configurable Agent that answers a prompt, or one you can reuse and continue in later nodes.
---

## When to use it

The Agent node configures an AI Agent with tools and rulesets. Given a prompt, it runs the Agent and returns the reply. Its `agent` output can also feed the `agent` input of other nodes, so the same Agent is reused across the workflow.

By default the node uses Griptape Cloud, which requires a valid `GT_CLOUD_API_KEY` or a Griptape Nodes license. Switch to Ollama, LM Studio or a custom endpoint with the provider parameter. See [AI Providers](https://docs.griptapenodes.com/en/stable/guides/agent/providers/) for setup.

## Example

Draft a dailies note for a shot, using `additional_context`:

1. Add a **Key Value Pair** node. Set `key` to `shot` and `value` to `SH010_comp`.
1. Add an **Agent**. Set `prompt` to "Write a two-sentence dailies note for shot {{shot}}, asking the compositor to match the plate's grain".
1. Connect the Key Value Pair `dictionary` output to the Agent's `additional_context` input.
1. Run the workflow. The Agent's `output` contains a dailies note for SH010_comp.

## Parameters

### agent

Connect an existing Agent to continue its conversation. Leave it empty to create a new one. When an Agent is connected, the provider, tools and rulesets parameters are hidden. The connected Agent keeps its own tools and rulesets, and any set on this node are added to them. When you pass an Agent from one node to another, its conversation memory goes with it. The Agent remembers earlier prompts in the flow, those prompts shape how it reads new ones, and you can build multi-turn conversations across several nodes.

### model_provider

Shown as **provider** in the editor. Choose from the configured providers. Griptape Cloud is the default and is always listed. Ollama, LM Studio and custom endpoints appear once you have set them up. See [AI Providers](https://docs.griptapenodes.com/en/stable/guides/agent/providers/).

### model

Shown as **prompt model** in the editor. Pick a model from the selected provider, or connect a prompt model config from a node such as [Griptape Cloud Prompt](../config/GriptapeCloudPrompt.md), [Open Ai Prompt](../config/OpenAiPrompt.md) or [Anthropic Prompt](../config/AnthropicPrompt.md). The connected configuration replaces the dropdown.

### prompt

The prompt is stripped of leading and trailing whitespace. If it is empty or only whitespace, the node creates the Agent without running it and `output` is exactly "Agent created." A string or number in `additional_context` is appended to the prompt, so an empty prompt with that context set does run, with the context as the prompt. A dictionary in `additional_context` does not count as a prompt.

### additional_context

A string, number or dictionary of key-value pairs such as the output of a [Key Value Pair](../dict/KeyValuePair.md) node.

- A string or number is appended to the prompt on a new line.
- A dictionary turns the prompt into a Jinja2 template, and the keys fill the `{{name}}` placeholders. The dictionary is not appended to the prompt, so a key that has no placeholder is not seen by the Agent. A placeholder with no matching key becomes empty text.
- Other values, such as a list, are converted to text and appended on a new line.

### tools

Connect tools as individual items or as a list.

### rulesets

Shown as **Behaviors (Rulesets)** in the editor. Use it to define what the Agent can and cannot do, set its persona, or constrain its output style. It accepts either of these:

- Ruleset nodes, for rules you want to reuse across Agents.
- A Text Input node, to write the behavior as plain text. You can also type it straight into the parameter.

Rules can be as specific as "Always name shots with the show prefix and a three-digit sequence, like ABC_010_0020" or "Keep notes under 50 words and address the artist directly".

> [!NOTE]
> The `.agents/skills` folder used by the [chat sidebar](https://docs.griptapenodes.com/en/stable/guides/agent/skills/) does not apply to Agent nodes in workflows. Use a Ruleset node or a Text Input node connected to this input instead.

### output_schema

A JSON schema that sets the exact format of the Agent's response. With a schema connected, `output` is JSON text in that format instead of free text. If the response does not match the schema, the run retries up to 3 times and then fails with an error. A schema that cannot be turned into a model fails the node with an error that suggests Create Agent Schema.

To build the schema from an example, connect a [Create Agent Schema](create_agent_schema.md) node. See [Structured output](#structured-output) below.

### output

The response as a string. When `output_schema` is connected, the output type changes to `json` and the value is the JSON text of the structured response. [JSON Extract Value](../json/JsonExtractValue.md) and the other JSON nodes parse that text. Connecting a schema removes any outgoing connection to an input that does not accept `json`.

## Structured output

A schema lets you ask for specific fields instead of a paragraph. Say you track shots in a spreadsheet and want to turn a supervisor's free-text production note into one row of status data.

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

1. Connect its `schema` output to the Agent's `output_schema` input.
1. Optionally add rules in `ruleset_example`, such as "status must be one of: pending, in_progress, review, approved". Connect `agent_ruleset` to the Agent's **Behaviors (Rulesets)** input so the Agent follows them.
1. Set the Agent's prompt to "Extract the shot status from this production note: [note text]".
1. The Agent responds with structured data containing exactly those fields. The node has one `output` that holds the whole object, not a separate output for each field. To use one field, connect `output` to a [JSON Extract Value](../json/JsonExtractValue.md) node with `path` set to `frames_remaining`. Its result is typed `json`, so check that a downstream input accepts it. For example, [Compare Numbers](../execution/CompareNumbers.md) takes only `int` or `float` inputs.

The schema types, the `null` rule and the top-level list case are described on the [Create Agent Schema](create_agent_schema.md) page, along with the generated schema for a similar example.

## Common issues

- **No provider configured**: If no provider is set, the node uses Griptape Cloud, which requires a valid `GT_CLOUD_API_KEY` or a Griptape Nodes license. See [AI Providers](https://docs.griptapenodes.com/en/stable/guides/agent/providers/) to add Ollama, LM Studio or a custom endpoint.
- **Missing credential**: The workflow fails validation when the Agent would use Griptape Cloud and neither a `GT_CLOUD_API_KEY` nor a Griptape Nodes license is available. A connected prompt model config or a connected Agent supplies its own model, so this check is skipped.
