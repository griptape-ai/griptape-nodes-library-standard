Extract values from JSON using [JMESPath](https://jmespath.org) expressions.

## Description

The JSON Extract Value node allows you to extract specific values from JSON data using JMESPath expressions. It supports nested object access, array indexing, and powerful wildcard operations, making it easy to navigate and extract data from complex JSON structures.

## Parameters

### Input Parameters

| Parameter | Type            | Description                                                                                | Default |
| --------- | --------------- | ------------------------------------------------------------------------------------------ | ------- |
| `json`    | json, str, dict | The JSON data to extract from                                                              | `{}`    |
| `path`    | str             | JMESPath expression to extract data (e.g., 'user.name', 'items[0].title', '[\*].assignee') | `""`    |

### Output Parameters

| Parameter | Type | Description         |
| --------- | ---- | ------------------- |
| `output`  | json | The extracted value |

## Features

- **JMESPath Expressions**: Use powerful JMESPath syntax for flexible data extraction
- **Dot Notation Paths**: Use simple dot notation to navigate JSON structure
- **Array Indexing**: Access array elements using `[index]` syntax
- **Wildcard Operations**: Extract all values from arrays using `[*]` syntax
- **Native Python Types**: Returns raw Python values (strings, dicts, lists, etc.) - no JSON serialization needed
- **Real-time Updates**: Automatically updates when input changes
- **Safe Extraction**: Returns empty dict `{}` if path doesn't exist
- **Empty Path**: If `path` is empty, the whole input is returned
- **Strict Input Parsing**: A JSON string must be valid JSON. The node does not repair it, and fails with an error if it cannot be parsed or if the JMESPath expression is invalid

## JMESPath Syntax

> **📚 Learn More**: For complete JMESPath syntax and advanced features, see the [JMESPath Documentation](https://jmespath.org/tutorial.html).

### Basic Object Access

```python
# Access nested object properties
path = "user.name"  # Gets user's name
path = "user.profile.email"  # Gets nested email
```

### Array Indexing

```python
# Access array elements
path = "items[0]"  # Gets first item
path = "items[0].title"  # Gets title of first item
path = "users[2].name"  # Gets name of third user
```

### Wildcard Operations

```python
# Extract all values from arrays
path = "items[*].title"  # Gets all titles from items array
path = "users[*].name"  # Gets all names from users array
path = "[*].assignee"  # Gets all assignees from root array
```

### Complex Paths

```python
# Combine object and array access
path = "orders[0].items[1].price"  # Gets price of second item in first order
path = "projects[*].tasks[*].assignee"  # Gets all assignees from all project tasks
```

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

### Feeding the result to other nodes

The output keeps its type, so a list can go straight into a list node and a number into a math node. For example, `shots[?status=='in_progress'].code` gives a list of shot codes. Connect it to [Create Text List](../lists/CreateTextList.md) or an [Agent](../agents/Agent.md) prompt to write a "still open" summary for the daily.

> [!NOTE]
> The node returns native Python types (string, number, list, dict). There is no JSON text to parse afterwards. An empty `path` returns the whole input.

## Use Cases

- **Production tracker data:** pull shot codes, statuses or assignees from a tracker's JSON export.
- **Render and farm reports:** read frame counts, error counts or job IDs from a render manager's response.
- **Asset and delivery specs:** extract the format, resolution or date a client asked for.
- **Agent output:** read one field out of the structured JSON an [Agent](../agents/Agent.md) returns when you give it an output schema.

## Related Nodes

- [JSON Input](JsonInput.md) - Create JSON data from inputs
- [JSON Replace](JsonReplace.md) - Replace values in JSON
- [Display JSON](DisplayJson.md) - Display and format JSON data
- [To JSON](../convert/ToJson.md) - Convert other data types to JSON
