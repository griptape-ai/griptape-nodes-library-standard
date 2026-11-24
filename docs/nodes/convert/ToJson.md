---
description: Converts a value to JSON, repairing malformed JSON text on the way.
when_to_use: Use it to turn a model reply, a pasted string or another value into JSON that the JSON nodes can read, such as a shot list returned by an Agent.
---

## When to use it

You have text that is meant to be JSON but may have small mistakes, such as unquoted keys, single quotes or a trailing comma, and you want to hand it to a node that expects JSON. Dictionaries pass through unchanged.

## Example

1. Connect an [Agent](../agents/Agent.md) reply such as `{'shot': 'SH010', status: 'ip'}` to `from`.
1. `output` is the repaired text `{"shot": "SH010", "status": "ip"}`.
1. Connect `output` to [JSON Extract Value](../json/JsonExtractValue.md) with path `status` to read one field.

## Parameters

### from

The value to convert. It accepts a connection only, and any type.

- A dictionary is passed through as it is.
- A string is repaired with `json-repair`. The result is the repaired JSON text, not a parsed object.
- Any other value, such as a list or number, is converted to text first and then repaired.

### output

The converted value, typed as `json`. The node updates as soon as `from` changes, and again when it runs. Text that is not JSON-like at all, such as `simple string`, comes out as an empty string. The node fails with an error only if the repair itself raises.

## Related nodes

- [JSON Input](../json/JsonInput.md)
- [JSON Extract Value](../json/JsonExtractValue.md)
- [JSON Replace](../json/JsonReplace.md)
- [Display JSON](../json/DisplayJson.md)
