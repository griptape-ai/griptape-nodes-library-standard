---
description: Runs an AI Agent with a prompt, tools and rulesets, and passes the configured Agent on to other nodes.
when_to_use: Use it when you want a configurable Agent that answers a prompt, or one you can reuse and continue in later nodes.
---

## When to use it

You want an AI that answers a prompt inside a workflow, with tools it can call and rules it must follow. Pass its `agent` output to later nodes to continue the same conversation.

## What it needs

- A Griptape Nodes license or a Griptape Cloud API key (`GT_CLOUD_API_KEY`). Griptape Cloud is the default provider. The workflow fails validation if neither is available.
- No credential is checked when a prompt model config or an Agent is connected, or when the provider is not Griptape Cloud.
- To use Ollama, LM Studio or a custom endpoint, pick it in the provider parameter. See [AI Providers](https://docs.griptapenodes.com/en/stable/guides/agent/providers/).

## Example

Draft a dailies note for a shot, using `additional_context`:

1. Add a **Key Value Pair** node. Set `key` to `shot` and `value` to `SH010_comp`.
1. Add an **Agent**. Set `prompt` to "Write a two-sentence dailies note for shot {{shot}}, asking the compositor to match the plate's grain".
1. Connect the Key Value Pair `dictionary` output to the Agent's `additional_context` input and run. The `output` is a dailies note for SH010_comp.

## Parameters

### agent

Connect an existing Agent to continue its conversation, or leave it empty to create a new one. Its memory goes with it, so you can build multi-turn conversations across nodes.

With an Agent connected, the provider, tools and rulesets parameters are hidden. The connected Agent keeps its own tools and rulesets, and any set on this node are added to them.

### model_provider

Shown as **provider** in the editor. Griptape Cloud is the default and is always listed. Ollama, LM Studio and custom endpoints appear once you have set them up.

### model

Shown as **prompt model** in the editor. Pick a model, or connect a config from [Griptape Cloud Prompt](../config/GriptapeCloudPrompt.md), [Open Ai Prompt](../config/OpenAiPrompt.md) or [Anthropic Prompt](../config/AnthropicPrompt.md). A connected config replaces the dropdown.

### prompt

The prompt is stripped of leading and trailing whitespace. With an empty or whitespace-only prompt, the node creates the Agent without running it and `output` is "Agent created." A string or number in `additional_context` counts as a prompt, so the Agent runs on that alone. A dictionary does not.

### additional_context

A string, number or dictionary, such as the output of a [Key Value Pair](../dict/KeyValuePair.md) node.

- A string or number is appended to the prompt on a new line. Other values, such as a list, are converted to text and appended the same way.
- A dictionary turns the prompt into a Jinja2 template, and the keys fill the `{{name}}` placeholders. A key with no placeholder is not seen by the Agent. A placeholder with no matching key becomes empty text.

### tools

Connect tools as individual items or as a list.

### rulesets

Shown as **Behaviors (Rulesets)** in the editor. Sets what the Agent can and cannot do, its persona, or its output style. Connect Ruleset nodes for rules you reuse across Agents, or a Text Input node for plain text, or type the text into the parameter. An example rule: "Keep notes under 50 words and address the artist directly".

> [!NOTE]
> The `.agents/skills` folder used by the [chat sidebar](https://docs.griptapenodes.com/en/stable/guides/agent/skills/) does not apply to Agent nodes in workflows. Use a Ruleset node or a Text Input node instead.

### output_schema

A JSON schema that sets the exact format of the response. If the response does not match, the run retries up to 3 times and then fails. To build the schema from an example, connect a [Create Agent Schema](create_agent_schema.md) node. See [Structured output](#structured-output). A schema that cannot be turned into a model fails the node with an error that suggests Create Agent Schema.

### output

The response as a string. With `output_schema` connected, the type is `json` and the value is the JSON text of the response, which [JSON Extract Value](../json/JsonExtractValue.md) and the other JSON nodes can parse. Connecting a schema removes any outgoing connection to an input that does not accept `json`.

## Structured output

To turn a supervisor's free-text production note into one row of shot status data:

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
1. Optionally put rules such as "status must be one of: pending, in_progress, review, approved" in `ruleset_example`, and connect `agent_ruleset` to the Agent's **Behaviors (Rulesets)** input.
1. Set the prompt to "Extract the shot status from this production note: [note text]". The single `output` holds the whole object with exactly those fields.

To use one field, connect `output` to a [JSON Extract Value](../json/JsonExtractValue.md) node with `path` set to `frames_remaining`. The result is typed `json`, and [Compare Numbers](../execution/CompareNumbers.md) takes only `int` or `float`, so check that the downstream input accepts it.

The schema types, the `null` rule and the top-level list case are on the [Create Agent Schema](create_agent_schema.md) page.
