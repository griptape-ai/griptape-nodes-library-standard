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

Connect an existing Agent to continue its conversation. Leave it empty to create a new one. When you pass an Agent from one node to another, its conversation memory goes with it. The Agent remembers earlier prompts in the flow, those prompts shape how it reads new ones, and you can build multi-turn conversations across several nodes.

### model_provider

Shown as **provider** in the editor. Choose Griptape Cloud, Ollama, LM Studio or a custom endpoint. See [AI Providers](https://docs.griptapenodes.com/en/stable/guides/agent/providers/).

### model

Shown as **prompt model** in the editor. Pick a model from the selected provider, or connect a prompt model config from a node such as [Griptape Cloud Prompt](../config/GriptapeCloudPrompt.md), [Open Ai Prompt](../config/OpenAiPrompt.md) or [Anthropic Prompt](../config/AnthropicPrompt.md). The connected configuration replaces the dropdown.

### prompt

If you leave it empty, the node creates the Agent without running it and `output` is exactly "Agent created."

### additional_context

A string, or a dictionary of key-value pairs such as the output of a [Key Value Pair](../dict/KeyValuePair.md) node.

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

A JSON schema that sets the exact format of the Agent's response. With a schema connected, `output` is structured JSON instead of plain text. If the Agent cannot produce data in that format, it tries again or returns an error.

To build the schema from an example, connect a [Create Agent Schema](create_agent_schema.md) node. See [Structured output](#structured-output) below.

### output

The text response, or a JSON object when `output_schema` is connected.

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
1. The Agent responds with structured data containing exactly those fields. The node has one `output` that holds the whole object, not a separate output for each field. To use one field, connect `output` to a [JSON Extract Value](../json/JsonExtractValue.md) node with `path` set to `frames_remaining`, then send the result on, for example to a [Compare Numbers](../execution/CompareNumbers.md) node that flags shots with frames still to do.

The schema types, the `null` rule and the top-level list case are described on the [Create Agent Schema](create_agent_schema.md) page, along with the generated schema for a similar example.

## Common issues

- **No provider configured**: If no provider is set, the node uses Griptape Cloud, which requires a valid `GT_CLOUD_API_KEY` or a Griptape Nodes license. See [AI Providers](https://docs.griptapenodes.com/en/stable/guides/agent/providers/) to add Ollama, LM Studio or a custom endpoint.
- **Missing credential**: The workflow fails validation when the Agent would use Griptape Cloud and neither a `GT_CLOUD_API_KEY` nor a Griptape Nodes license is available. A connected prompt model config or a connected Agent supplies its own model, so this check is skipped.
