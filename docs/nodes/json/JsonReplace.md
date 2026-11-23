---
description: Replaces the value at a dot-notation path in JSON data and outputs the modified copy.
when_to_use: Use it when you need to change one field in a JSON record, such as a shot's status, without touching the rest.
---

## When to use it

You have JSON data and want to change one value inside it, for example a shot status in a shot record, a field in a request payload, or a setting in a configuration. Give the node a path in dot notation and the new value. It works on a deep copy, so the source data is not modified, and it creates missing paths. Typical uses are updating records, filling in template JSON with dynamic values and adjusting payloads for different systems.

## Example

### Basic value replacement

```python
# Original JSON: {"user": {"name": "John", "age": 30}}
# Path: "user.name"
# Replacement Value: "Jane"
# Output: {"user": {"name": "Jane", "age": 30}}
```

### Array element replacement

```python
# Original JSON: {"items": [{"title": "Book", "price": 25}, {"title": "Magazine", "price": 10}]}
# Path: "items[0].title"
# Replacement Value: "Novel"
# Output: {"items": [{"title": "Novel", "price": 25}, {"title": "Magazine", "price": 10}]}
```

### Creating new paths

```python
# Original JSON: {"user": {"name": "John"}}
# Path: "user.email"
# Replacement Value: "john@example.com"
# Output: {"user": {"name": "John", "email": "john@example.com"}}
```

### Nested array replacement

```python
# Original JSON: {"orders": [{"items": [{"name": "Product A"}, {"name": "Product B"}]}]}
# Path: "orders[0].items[1].name"
# Replacement Value: "Product C"
# Output: {"orders": [{"items": [{"name": "Product A"}, {"name": "Product C"}]}]}
```

## Parameters

### json

A JSON string must be valid JSON. The node does not repair it and fails with an error if it cannot be parsed. The node deep-copies the input, so the original is never changed.

### path

Dot notation with list indexes, such as `user.profile.email`, `items[0].title` or `orders[0].items[1].price`.

- A path that does not exist is created.
- If the path cannot be followed, for example because it goes through a value that is not an object or list, the output is an unchanged copy of the input.
- If `path` is empty, the output is the replacement value itself.

> [!WARNING]
> A path that ends in a list index, such as `items[0]` or `items[2]`, does not set the value. This is a known bug (#1081). `items[0]` leaves the list unchanged, and an index past the end pads the list with `null` without writing the value:
>
> ```python
> # Original JSON: {"items": ["a", "b"]}
> # Path: "items[2]"
> # Replacement Value: "X"
> # Output: {"items": ["a", "b", null]}
> ```
>
> Paths that continue into the item work, for example `items[0].title`. To replace a whole list item for now, change the item with [JSON Extract Value](JsonExtractValue.md) and rebuild the list with the list nodes.

### replacement_value

The value is used as it arrives. A string such as `5` stays text unless it comes from a number or JSON connection.

### output

The modified JSON.

## Related nodes

- [JSON Input](JsonInput.md) creates JSON data from inputs.
- [JSON Extract Value](JsonExtractValue.md) extracts values from JSON.
- [Display JSON](DisplayJson.md) displays and formats JSON data.
- [To JSON](../convert/ToJson.md) converts other data types to JSON.
