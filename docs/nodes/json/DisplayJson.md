Shows JSON data on the node and passes it through. It also reads JSON or YAML from text.

## Description

The Display JSON node parses its input and shows the result. A string is parsed as JSON first, with `json-repair` as a fallback for malformed JSON. If the text is YAML, or JSON or YAML inside a markdown code block (for example a model's reply), that is extracted and parsed. If the text cannot be parsed as either, the node fails with an error.

## Parameters

### Input Parameters

| Parameter | Type            | Description              | Default |
| --------- | --------------- | ------------------------ | ------- |
| `json`    | json, str, dict | The JSON data to display (can also be set on the node) | `{}`    |

### Output Parameters

| Parameter | Type | Description             |
| --------- | ---- | ----------------------- |
| `json`    | json | The formatted JSON data |

## Features

- **Automatic JSON Repair**: Handles malformed JSON strings using `json-repair`
- **YAML and Markdown Input**: Parses YAML text and unwraps JSON or YAML fenced code blocks
- **Multiple Input Types**: Accepts dictionaries and strings. Other values are converted to text and repaired
- **Real-time Processing**: Updates automatically when input changes

## Examples

### Basic JSON Display

```python
# Input: {"name": "John", "age": 30, "active": true}
# Output: {"name": "John", "age": 30, "active": true}
```

### Malformed JSON Repair and Display

```python
# Input: '{"name": "John", age: 30, "city": "New York"}'  # Missing quotes around age
# Output: {"name": "John", "age": 30, "city": "New York"}  # Repaired and formatted
```

### Complex JSON Structure

```python
# Input: {"user": {"name": "Alice", "profile": {"email": "alice@example.com", "preferences": {"theme": "dark"}}}}
# Output: {"user": {"name": "Alice", "profile": {"email": "alice@example.com", "preferences": {"theme": "dark"}}}}
```

### String to JSON Conversion and Display

```python
# Input: '{"items": [{"title": "Book", "price": 25}, {"title": "Magazine", "price": 10}]}'
# Output: {"items": [{"title": "Book", "price": 25}, {"title": "Magazine", "price": 10}]}
```

## Use Cases

- **Debugging**: View and inspect JSON data during workflow development
- **Data Validation**: Verify JSON structure and content
- **API Response Inspection**: Examine responses from external APIs
- **Configuration Review**: Check JSON configuration files
- **Data Flow Monitoring**: Monitor JSON data as it moves through your workflow

## Related Nodes

- [JSON Input](JsonInput.md) - Create JSON data from inputs
- [JSON Extract Value](JsonExtractValue.md) - Extract values from JSON
- [JSON Replace](JsonReplace.md) - Replace values in JSON
- [To JSON](../convert/ToJson.md) - Convert other data types to JSON
