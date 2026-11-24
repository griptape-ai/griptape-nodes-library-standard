---
description: Shows JSON on the node and passes it through, parsing JSON or YAML from text.
when_to_use: Use it when you want to inspect JSON during a workflow, or turn text such as a model's reply into JSON.
---

## When to use it

You want to inspect JSON while you build a workflow, such as an API response, a configuration file, or data moving between nodes. It also turns text into JSON, such as a model's reply.

How the input is handled:

- Dictionaries pass through. Other values are converted to text and repaired, so the list `[1, 2]` becomes the string `'[1, 2]'`.
- A string is parsed as JSON. YAML, or JSON or YAML inside a markdown code block, is extracted and parsed, so `a: 1` gives `{"a": 1}`.
- Malformed JSON is repaired with `json-repair`. The result is text, not an object.
- Text that is not JSON at all gives an empty string, with no error.
- The node fails with an error when the text looks like YAML but cannot be parsed, such as truncated JSON like `{"name": "John"` or valid JSON followed by extra text. The error is raised when you set the value as well as when the node runs.

## Examples

A well-formed object is shown as it is:

```python
# Input: {"name": "John", "age": 30, "active": true}
# Output: {"name": "John", "age": 30, "active": true}
```

Malformed JSON is repaired, and the result is text:

```python
# Input: '{"name": "John", age: 30, "city": "New York"}'  # Missing quotes around age
# Output: '{"name": "John", "age": 30, "city": "New York"}'  # a string, not an object
```

## Parameters

### json

You can set the value on the node or connect it. The output `json` is the parsed object for valid JSON and for dictionaries, and text for repaired JSON.

## Edge cases

- Truncated JSON that is a list, such as `[1,2`, is repaired to the text `[1, 2]`.

## Related nodes

- [JSON Input](JsonInput.md) creates JSON data from inputs.
- [JSON Extract Value](JsonExtractValue.md) extracts values from JSON.
- [JSON Replace](JsonReplace.md) replaces values in JSON.
- [To JSON](../convert/ToJson.md) converts other data types to JSON.
