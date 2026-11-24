---
description: Adds an item to a list at the start, the end or a chosen index, and outputs the new list.
when_to_use: Use it to append or insert one value into a list, such as adding a new shot to a shot list.
---

## When to use it

You have a list and want to add one entry. The input list is not modified. If `items` is not connected, the node starts from an empty list.

## Example

1. Connect `["SH010", "SH020"]` to `items`.
1. Set `item` to `SH030` and leave `position` at `end`.
1. `output` is `["SH010", "SH020", "SH030"]`.

## Parameters

### position

`end` (the default), `start` or `index`. `index` shows the `index` field.

### index

Zero-based position for the new entry. Required when `position` is `index`, and it must be a whole number. Python insert rules apply: a value past the end appends, and a negative value counts from the end.

### skip_empty_values

In the collapsed Advanced group, on by default. When on, an empty `item` (no value) is not added and the list passes through unchanged. Turn it off to add an empty entry.

### output

The list with the new entry.
