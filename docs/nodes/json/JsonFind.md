---
description: Finds items in a JSON array by matching a field, and returns the first match or all matches.
when_to_use: Use it when you need to look up an item in a list of records by name, status or assignee, such as a shot, task or version.
---

## When to use it

You have a list of records and want the ones that match a field value, such as a shot, task or version by name, status or assignee. It supports exact, contains and starts-with matching, and returns the first match or all of them.

If the input is an object, the node searches the first array it finds under the keys `data`, `items`, `results` or `list`, in that order. If `data` holds an array, `items` is never searched.

## Examples

Find a task by content:

```python
# Input JSON: [
#   {"attributes": {"content": "Design", "status": "wtg"}},
#   {"attributes": {"content": "Design Review", "status": "ip"}}
# ]
# search_field: "attributes.content"
# search_value: "Design"
# search_mode: "exact"
# return_mode: "first"
# Result: {"attributes": {"content": "Design", "status": "wtg"}}
```

Find all tasks with a status:

```python
# Input JSON: [{"attributes": {"content": "Design", "status": "wtg"}},
#   {"attributes": {"content": "Model", "status": "wtg"}},
#   {"attributes": {"content": "Review", "status": "ip"}}]
# search_field: "attributes.status"
# search_value: "wtg"
# search_mode: "exact"
# return_mode: "all"
# Result: [{"attributes": {"content": "Design", "status": "wtg"}},
#          {"attributes": {"content": "Model", "status": "wtg"}}]
```

Nested search:

```python
# Input JSON: [{"relationships": {"entity": {"data": {"name": "convertible", "type": "Asset"}}}}]
# search_field: "relationships.entity.data.name"
# search_value: "convertible"
# search_mode: "exact"
# return_mode: "first"
# Result: The matching item
```

## Parameters

### json

An array, or an object that contains an array under one of the keys above. A string that cannot be parsed, including an empty string, fails the node when it runs. While you edit values on the node, the same input clears the outputs instead.

### search_field

Dot notation with optional `[n]` array indexes, such as `attributes.content` or `items[0].title`. An index must follow a key, so `[0].title` does not work. It does not support JMESPath wildcards or filters. Use [JSON Extract Value](JsonExtractValue.md) for those. An item without the path does not match.

### search_value

The field value and `search_value` are both converted to text before they are compared, so the number `5` matches `"5"`. A float such as `5.0` becomes `"5.0"` and does not match `"5"`. `true` becomes `"True"`.

### search_mode

An unrecognised mode behaves like `exact`.

For a search value of `Design`:

| Mode | Matches | Does not match |
| --- | --- | --- |
| `exact` (default) | `Design` | `Design Task`, `design`, `My Design` |
| `contains` | `Design`, `Design Task`, `My Design Work` | `design` when `case_sensitive` is on |
| `starts_with` | `Design`, `Design Task`, `Designer` | `My Design`, `design` |

### return_mode

`first` (default) returns the first match. `all` returns every match as an array.

### case_sensitive

On by default.

### found_item

The single item for `first`, or an array for `all`. With no match it is `{}` for `first` and `[]` for `all`.

### found_count

The number of all matches, even when `return_mode` is `first`. It is 0 with no match.

### found_index

The index of the first match, or -1 if nothing matched. It is set in both return modes.

## Edge cases

- If `search_field` or `search_value` is empty, or the input has no array, `found_item` is `{}` in both modes.
- A key that contains a dot cannot be matched.
- A JSON `null` field never matches.

## Related nodes

- [JSON Extract Value](JsonExtractValue.md) extracts single values by path.
- [JSON Input](JsonInput.md) creates JSON data from inputs.
- [JSON Replace](JsonReplace.md) replaces values in JSON.
- [Display JSON](DisplayJson.md) displays and formats JSON data.
- [To JSON](../convert/ToJson.md) converts other data types to JSON.
