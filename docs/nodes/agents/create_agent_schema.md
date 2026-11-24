---
description: Builds a JSON schema from an example of the data you want, for an Agent's output_schema.
when_to_use: Use it when you want an Agent to answer with structured data instead of free text, and you'd rather show an example than write a JSON Schema.
---

## When to use it

You want an [Agent](Agent.md) to return fields you can connect to other nodes, such as a shot code, a status and a list of blockers. Type out one example of that data and this node works out the schema. Connect its `schema` output to the Agent's `output_schema` input.

## How the schema is built

The schema follows the types in your example.

- Strings, whole numbers, decimals and booleans give fields of the same type.
- A nested object gives a nested object.
- A list gives an array. If every item has the same type, the array is limited to that type.
- A field set to `null` is optional. Every other field is required.
- A list at the top level, such as `["tag1", "tag2"]`, is wrapped in an object with one `items` array. The Agent's output is then `{"items": [...]}`.

If the example isn't valid JSON, the node tries to repair it before giving up.

## Example

Set `example` to:

```json
{
  "shot_code": "SH010",
  "department": "comp",
  "status": "in_progress",
  "frames_remaining": 48,
  "blockers": ["waiting on final plate", "roto not approved"]
}
```

The `schema` output is:

```json
{
  "type": "object",
  "properties": {
    "shot_code": { "type": "string" },
    "department": { "type": "string" },
    "status": { "type": "string" },
    "frames_remaining": { "type": "integer" },
    "blockers": { "type": "array", "items": { "type": "string" } }
  },
  "required": ["shot_code", "department", "status", "frames_remaining", "blockers"]
}
```

## Parameters

### example_template

Fills in `example` from a ready-made template, such as Task List, Image Generation Prompt or Restaurant Review. Choosing a template replaces whatever is in `example`. Pick **Custom** to write your own.

### example

The sample data to build the schema from. Type it in or connect a JSON input. The schema updates as soon as the example changes.

### ruleset_example

Some templates come with rules, such as "Status should be one of: 'in_progress', 'pending', 'completed', 'cancelled'". They show here and you can edit them. Separate rules with a blank line.

### agent_ruleset

The rules from `ruleset_example` as a ruleset. Connect it to the Agent's **Behaviors (Rulesets)** input so the Agent follows them. A schema can say a status is a string, but only a rule can say it must be one of pending, in_progress, review or approved.

### schema

The generated JSON schema. Connect it to the Agent's `output_schema` input.
