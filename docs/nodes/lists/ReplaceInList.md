---
description: Replaces one item in a list, found either by value or by position, and outputs the changed list.
when_to_use: Use it to swap a single entry in a list, such as updating one shot in a list of shot codes.
---

## When to use it

You have a list and want to change one entry without rebuilding the list. The node outputs a modified copy and leaves the input list alone.

## Example

A list of shots `["SH010", "SH020", "SH030"]` needs `SH020` renamed to `SH025`.

1. Connect the list to `items`.
1. Set `replace_by` to `item`, set `item_to_replace` to `SH020`, and set `new_item` to `SH025`.
1. `output` is `["SH010", "SH025", "SH030"]`.

To replace by position, set `replace_by` to `index` and set `index_to_replace` to `1`.

## Parameters

### replace_by

`item` (the default) matches `item_to_replace` against the list and replaces the first match. `index` replaces the entry at `index_to_replace`. Only the field for the chosen mode is shown.

### index_to_replace

Zero-based position. Defaults to 0. If it is outside the list, the node produces no output.

### new_item

The replacement value. If it is empty or falsy (an empty string, `0` or `False`), the node produces no output.

### output

The modified list. It is not filled when `items` is empty, when the item or index is not found, or when `new_item` is falsy.
