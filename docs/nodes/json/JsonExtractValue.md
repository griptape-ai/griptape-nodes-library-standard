---
description: Pulls a value out of JSON with a JMESPath expression.
when_to_use: Use it when you need one field, a list of fields, or a filtered or computed result from a JSON structure.
---

## When to use it

You want one field, a list of fields, or a computed result from JSON. Typical sources are a production tracker export, a render manager's response, a delivery spec, or the structured JSON an [Agent](../agents/Agent.md) returns when you give it an output schema.

The node reads values with [JMESPath](https://jmespath.org) expressions: nested access, array indexing, wildcards, filters and functions such as `sum()`. See the [JMESPath tutorial](https://jmespath.org/tutorial.html) for the full syntax.

## Examples

These examples read from the same shot list:

```json
{
  "project": "Starfall",
  "shots": [
    { "code": "SH010", "status": "in_progress", "artist": "Maria", "frames": 96, "tags": ["comp", "cg"] },
    { "code": "SH020", "status": "approved", "artist": "Dev", "frames": 120, "tags": ["roto"] },
    { "code": "SH030", "status": "in_progress", "artist": "Maria", "frames": 48, "tags": ["paint", "comp"] }
  ],
  "delivery": { "date": "2026-12-04", "format": "EXR" }
}
```

| `path` | What it asks for | Output |
| --- | --- | --- |
| `project` | A top-level value | `"Starfall"` |
| `shots[0].code` | A field of the first shot | `"SH010"` |
| `shots[*].code` | The code of every shot | `["SH010", "SH020", "SH030"]` |
| `shots[*].frames` | The frame count of every shot | `[96, 120, 48]` |
| `sum(shots[*].frames)` | The total frames across all shots | `264` |
| `shots[?status=='in_progress'].code` | Only the shots still in progress | `["SH010", "SH030"]` |
| `delivery` | A whole object | `{"date": "2026-12-04", "format": "EXR"}` |
| `delivery.contact` | A key that isn't there | `{}` |

Other path forms: `[*].assignee` for a root array, and `projects[*].tasks[*].assignee` for all assignees across all tasks.

The output keeps its type, so a list can go straight into a list node and a number into a math node. Connect the in-progress shot codes to [Create Text List](../lists/CreateTextList.md) or an [Agent](../agents/Agent.md) prompt to write a "still open" summary for the daily.

## Parameters

### json

A JSON string must be valid JSON. This node does not repair it, and a string that cannot be parsed, including an empty one, fails when the node runs. While you edit the value on the node, `output` is cleared instead. Use [JSON Input](JsonInput.md) or [Display JSON](DisplayJson.md) first if the text may be malformed.

### path

A JMESPath expression. If it is empty, the whole input is returned. An invalid expression, or a function given the wrong type such as `sum()` on a number, fails when the node runs.

### output

The extracted value as a native type (string, number, list or dict). If the path does not exist, or the value is JSON `null`, the output is `{}`.

## Related nodes

- [JSON Input](JsonInput.md) creates JSON data from inputs.
- [JSON Replace](JsonReplace.md) replaces values in JSON.
- [Display JSON](DisplayJson.md) displays and formats JSON data.
- [To JSON](../convert/ToJson.md) converts other data types to JSON.
