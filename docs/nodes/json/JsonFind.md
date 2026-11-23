---
description: Finds items in a JSON array by matching a field, and returns the first match or all matches.
when_to_use: Use it when you need to look up an item in a list of records by name, status or assignee, such as a shot, task or version.
---

## When to use it

Use it to search a JSON array for items that match a field value. It supports exact, contains and starts-with matching, case-sensitive or not, and returns either the first match or all matches. Typical uses are finding a shot, task or version in a list by name, status or assignee, filtering an API response, or looking up an entry in a configuration.

If the input is an object that contains arrays, the node looks for an array under the field names `data`, `items`, `results` or `list`. For example, with `{"data": [{"name": "Item 1"}, {"name": "Item 2"}]}` it searches the `data` array.

## Examples

Find a task by content:

```python
# Input JSON: [
#   {"attributes": {"content": "Design", "status": "wtg"}},
#   {"attributes": {"content": "Model", "status": "wtg"}},
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
# Input JSON: [
#   {"attributes": {"content": "Design", "status": "wtg"}},
#   {"attributes": {"content": "Model", "status": "wtg"}},
#   {"attributes": {"content": "Review", "status": "ip"}}
# ]
# search_field: "attributes.status"
# search_value: "wtg"
# search_mode: "exact"
# return_mode: "all"
# Result: [{"attributes": {"content": "Design", "status": "wtg"}},
#          {"attributes": {"content": "Model", "status": "wtg"}}]
```

Case-insensitive search:

```python
# Input JSON: [
#   {"name": "Design Task"},
#   {"name": "design review"},
#   {"name": "DESIGN WORK"}
# ]
# search_field: "name"
# search_value: "design"
# search_mode: "contains"
# case_sensitive: false
# return_mode: "all"
# Result: All three items
```

Items starting with a value:

```python
# Input JSON: [
#   {"title": "Design Task"},
#   {"title": "Design Review"},
#   {"title": "My Design Work"}
# ]
# search_field: "title"
# search_value: "Design"
# search_mode: "starts_with"
# return_mode: "all"
# Result: [{"title": "Design Task"}, {"title": "Design Review"}]
```

Nested search:

```python
# Input JSON: [
#   {
#     "relationships": {
#       "entity": {"data": {"name": "convertible", "type": "Asset"}}
#     }
#   }
# ]
# search_field: "relationships.entity.data.name"
# search_value: "convertible"
# search_mode: "exact"
# return_mode: "first"
# Result: The matching item
```

## Parameters

### json

An array, or an object that contains an array under one of the names listed above. A string that cannot be parsed makes the node fail with an error when it runs.

### search_field

Plain dot notation with optional `[n]` array indexes, such as `attributes.content` or `items[0].title`. It does not support JMESPath features such as wildcards or filters, which [JSON Extract Value](JsonExtractValue.md) does. A path that does not exist in an item means that item does not match.

### search_value

The field value and `search_value` are both converted to text before they are compared, so `5` matches `"5"`.

### search_mode

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

The single item for `first`, or an array for `all`.

### found_count

The number of all matches, even when `return_mode` is `first`.

### found_index

The index of the first match, or -1 if nothing matched.

## No match and empty inputs

If `search_field` or `search_value` is empty, or the input has no array to search, nothing is found. With nothing found, `found_item` is `{}` for `first` and `[]` for `all`, `found_count` is 0 and `found_index` is -1.

## Related nodes

- [JSON Extract Value](JsonExtractValue.md) extracts single values by path.
- [JSON Input](JsonInput.md) creates JSON data from inputs.
- [JSON Replace](JsonReplace.md) replaces values in JSON.
- [Display JSON](DisplayJson.md) displays and formats JSON data.
- [To JSON](../convert/ToJson.md) converts other data types to JSON.
