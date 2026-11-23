Holds JSON data that you type or paste into the node and sends it to other nodes.

## Description

The JSON Input node has a single `json` parameter that you edit on the node (it has no input connection). When it runs, it passes dictionaries through unchanged and runs strings through `json-repair`, which fixes common mistakes such as missing quotes. To convert a value coming from another node, use [To JSON](../convert/ToJson.md).

## Parameters

### Input Parameters

| Parameter | Type            | Description                                      | Default |
| --------- | --------------- | ------------------------------------------------ | ------- |
| `json`    | json, str, dict | The JSON data, set on the node (property only)   | `{}`    |

### Output Parameters

| Parameter | Type | Description             |
| --------- | ---- | ----------------------- |
| `json`    | json | The processed JSON data |

## Features

- **Automatic JSON Repair**: Handles malformed JSON strings using `json-repair`
- **Multiple Value Types**: Accepts dictionaries and strings. Other values are converted to text and repaired
- **Repaired Text Output**: When the value is a string, the output is the repaired JSON text rather than a parsed object. Pass it through [JSON Extract Value](JsonExtractValue.md) or another JSON node to work with the contents

## Examples

### Basic Usage

```python
# Input: {"name": "John", "age": 30}
# Output: {"name": "John", "age": 30}
```

### Malformed JSON Repair

```python
# Input: '{"name": "John", age: 30}'  # Missing quotes around age
# Output: {"name": "John", "age": 30}  # Repaired JSON
```

### String to JSON Conversion

```python
# Input: '{"user": {"name": "Alice", "active": true}}'
# Output: {"user": {"name": "Alice", "active": true}}
```

### Dictionary Input

```python
# Input: {"status": "active", "count": 5}
# Output: {"status": "active", "count": 5}  # Used as-is
```

## Use Cases

- **Fixed Data**: Keep a block of JSON, such as a shot list, in the workflow
- **Data Cleaning**: Repair malformed JSON that you paste in

## Related Nodes

- [JSON Extract Value](JsonExtractValue.md) - Extract specific values from JSON
- [JSON Replace](JsonReplace.md) - Replace values in JSON
- [Display JSON](DisplayJson.md) - Display and format JSON data
- [To JSON](../convert/ToJson.md) - Convert other data types to JSON
