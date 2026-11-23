---
description: Pulls a value out of JSON with a JMESPath expression.
when_to_use: Use it when you need one field, a list of fields, or a filtered or computed result from a JSON structure.
---

## When to use it

Use it to read specific values from JSON with [JMESPath](https://jmespath.org) expressions. It supports nested object access, array indexing, wildcards, filters and functions such as `sum()`. For the full syntax, see the [JMESPath tutorial](https://jmespath.org/tutorial.html).

Typical sources are shot codes, statuses or assignees from a production tracker export, frame counts or job IDs from a render manager's response, the format or date in a delivery spec, or one field from the structured JSON an [Agent](../agents/Agent.md) returns when you give it an output schema.

## Examples

The examples below all read from the same shot list, the kind of data a production tracker or a pipeline script hands you:

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
| `shots[?artist=='Maria'].frames` | Frame counts for one artist's shots | `[96, 48]` |
| `shots[0].tags[0]` | The first tag on the first shot | `"comp"` |
| `delivery` | A whole object | `{"date": "2026-12-04", "format": "EXR"}` |
| `delivery.contact` | A key that isn't there | `{}` |

Other path forms: `user.profile.email` for nested objects, `users[2].name` for the third user, `[*].assignee` for a root array, and `projects[*].tasks[*].assignee` for all assignees across all tasks.

The output keeps its type, so a list can go straight into a list node and a number into a math node. For example, `shots[?status=='in_progress'].code` gives a list of shot codes. Connect it to [Create Text List](../lists/CreateTextList.md) or an [Agent](../agents/Agent.md) prompt to write a "still open" summary for the daily.

## Parameters

### json

A JSON string must be valid JSON. This node does not repair it. When the node runs, a string that cannot be parsed (including an empty string) fails with an error. While you edit the value on the node, a string that cannot be parsed yet clears `output` instead of raising. Use [JSON Input](JsonInput.md) or [Display JSON](DisplayJson.md) first if the text may be malformed.

### path

A JMESPath expression. If it is empty, the whole input is returned. An invalid expression, or a function given the wrong type (such as `sum()` on a number), fails with an error when the node runs.

### output

The extracted value as a native type (string, number, list or dict), so there is no JSON text to parse afterwards. If the path does not exist, or the value at the path is JSON `null`, the output is an empty dict `{}`.

## Related nodes

- [JSON Input](JsonInput.md) creates JSON data from inputs.
- [JSON Replace](JsonReplace.md) replaces values in JSON.
- [Display JSON](DisplayJson.md) displays and formats JSON data.
- [To JSON](../convert/ToJson.md) converts other data types to JSON.
