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

A JSON string or a dictionary. A string must be valid JSON. The node does not repair it, and a run with an unparseable string (including an empty string) fails with a `ValueError` that says `Invalid JSON string provided`. While you edit inputs, unparseable JSON clears `output` instead of raising. The node deep-copies the input, so the original is never changed. The default is `{}`.

### path

Dot notation with list indexes, such as `user.profile.email`, `items[0].title` or `orders[0].items[1].price`.

- A missing key is created, so `a.b.c` on `{}` gives `{"a": {"b": {"c": ...}}}`. A missing list index is not: see the warning below.
- If the path goes through a value that is not an object or list, the output is an unchanged copy of the input. A path on a top-level list such as `0` or `[0]` also returns the list unchanged.
- If `path` is empty, the output is the replacement value itself.

> [!WARNING]
> A list index in the path does not work as a way to set a value, whether it comes last or in the middle. This is a known bug (#1081).
>
> - A path that ends in a list index does not set the value. `items[0]` leaves the list unchanged, and an index past the end pads the list with `null` without writing anything: `items[2]` on `{"items": ["a", "b"]}` gives `{"items": ["a", "b", null]}`.
> - An index past the end anywhere in the path pads the same way. `items[3].t` on `{"items": ["a"]}` gives `{"items": ["a", null, null, null]}`, and `items[0].title` on `{}` gives `{"items": [null]}`.
> - A path that continues into an item that already exists works, for example `items[0].title` on `{"items": [{"title": "Book"}]}`.
>
> To change a list for now, rebuild the whole list, connect it to `replacement_value`, and set `path` to the list's key (`items`). A list or dictionary connected to `replacement_value` replaces the value as it is.

### replacement_value

The value is used as it arrives. A string such as `5` is written as the string `"5"`, and the node does not parse it. A number is written as a number. A dictionary or list is written as it is. The default is an empty string.

### output

The modified JSON.

## Related nodes

- [JSON Input](JsonInput.md) creates JSON data from inputs.
- [JSON Extract Value](JsonExtractValue.md) extracts values from JSON.
- [Display JSON](DisplayJson.md) displays and formats JSON data.
- [To JSON](../convert/ToJson.md) converts other data types to JSON.
