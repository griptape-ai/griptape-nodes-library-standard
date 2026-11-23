---
description: Removes one item from a list by position or by value and outputs the shorter list.
when_to_use: Use it to drop an entry from a list, such as taking an omitted shot out of a shot list.
---

## When to use it

You have a list and want to remove its first entry, its last entry, the entry at an index, or the first entry equal to a value. The input list is not modified.

## Example

A shot list is `["SH010", "SH020", "SH030"]` and `SH020` was omitted.

1. Connect the list to `items`.
1. Set `remove_item_by` to `item` and set `item` to `SH020`.
1. `output` is `["SH010", "SH030"]`.

## Parameters

### remove_item_by

`item` (the default), `index`, `first` or `last`. `item` shows the `item` field, `index` shows the `index` field, and `first` and `last` show neither.

### item

The value to remove. The first matching entry is removed. If no entry matches, the node fails with an error.

### index

Zero-based position. Negative values count from the end. An index outside the list fails with an error. It has no default value, so set it before running.

### output

The list without the removed entry. It is not filled when `items` is empty.
