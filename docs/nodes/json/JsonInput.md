---
description: Holds JSON you type or paste into the node and sends it to other nodes.
when_to_use: Use it when you want to keep a fixed block of JSON, such as a shot list, in the workflow, or repair JSON you pasted in.
---

## When to use it

JSON Input has a single `json` parameter that you edit on the node. It has no input connection. Use it to keep fixed data, such as a shot list, in the workflow, or to repair malformed JSON that you paste in. To convert a value coming from another node, use [To JSON](../convert/ToJson.md).

When it runs, dictionaries pass through unchanged and strings go through `json-repair`, which fixes common mistakes such as missing quotes. Other values are converted to text and repaired.

## Examples

A well-formed object passes through:

```python
# Input: {"name": "John", "age": 30}
# Output: {"name": "John", "age": 30}
```

Malformed JSON is repaired:

```python
# Input: '{"name": "John", age: 30}'  # Missing quotes around age
# Output: {"name": "John", "age": 30}
```

A dictionary is used as it is:

```python
# Input: {"status": "active", "count": 5}
# Output: {"status": "active", "count": 5}
```

## Parameters

### json

Set on the node only. When the value is a string, the output is the repaired JSON text rather than a parsed object. Pass it through [JSON Extract Value](JsonExtractValue.md) or another JSON node to work with the contents.

## Related nodes

- [JSON Extract Value](JsonExtractValue.md) extracts specific values from JSON.
- [JSON Replace](JsonReplace.md) replaces values in JSON.
- [Display JSON](DisplayJson.md) displays and formats JSON data.
- [To JSON](../convert/ToJson.md) converts other data types to JSON.
