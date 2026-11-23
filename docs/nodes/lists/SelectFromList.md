---
description: Turns a list into a dropdown so you can pick one item and pass it on as text.
when_to_use: Use it when someone running the workflow should choose one value from a list, such as a shot status or a department.
---

## When to use it

You have a list of choices and want a dropdown to pick one. The node converts every item to text, so the output is always a string.

## Example

1. Add a [Create Text List](CreateTextList.md) node with `In Progress`, `Review`, `Approved` and `Omitted`.
1. Connect its `output` to `list` on **Select From List**.
1. Pick `Review` in the `selected_item` dropdown. `selected_item` outputs `Review`.

## Parameters

### list

Accepts a list, a JSON string that holds a list or an object, or a dictionary. For an object or dictionary, the keys become the choices. Items are converted with `str()` and duplicates are removed, keeping the first occurrence.

### selected_item

The dropdown. It keeps the current selection when the list changes and the value is still present. Otherwise it falls back to the first item. If the list is empty or cannot be read, the output is an empty string.
