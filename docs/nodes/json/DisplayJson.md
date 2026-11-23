---
description: Shows JSON on the node and passes it through, parsing JSON or YAML from text.
when_to_use: Use it when you want to inspect JSON during a workflow, or turn text such as a model's reply into JSON.
---

## When to use it

Display JSON parses its input and shows the result. Use it to inspect JSON while you build a workflow, check a structure, look at an API response or a configuration file, or follow data as it moves between nodes.

A string is parsed as JSON first, with `json-repair` as a fallback for malformed JSON. The fallback returns text, not an object, so repaired JSON is shown as JSON text. Text that is not JSON at all gives an empty string, with no error. If the text is YAML, or JSON or YAML inside a markdown code block such as a model's reply, that is extracted and parsed. The node fails with an error only when the text looks like YAML but cannot be parsed as YAML, such as truncated JSON like `{"name": "John"` or valid JSON followed by extra text. The error is raised when you set the value on the node as well as when the node runs. Truncated JSON that is a list, such as `[1,2`, is repaired to the text `[1, 2]`. Dictionaries pass through. Other values are converted to text and repaired, so a list becomes the string `'[1, 2]'`. YAML is parsed to an object, so `a: 1` gives `{"a": 1}`.

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

A JSON string is parsed:

```python
# Input: '{"items": [{"title": "Book", "price": 25}, {"title": "Magazine", "price": 10}]}'
# Output: {"items": [{"title": "Book", "price": 25}, {"title": "Magazine", "price": 10}]}
```

## Parameters

### json

You can set the value on the node or connect it. The output `json` is the parsed object for valid JSON and for dictionaries. For repaired JSON it is text.

## Related nodes

- [JSON Input](JsonInput.md) creates JSON data from inputs.
- [JSON Extract Value](JsonExtractValue.md) extracts values from JSON.
- [JSON Replace](JsonReplace.md) replaces values in JSON.
- [To JSON](../convert/ToJson.md) converts other data types to JSON.
