---
description: Builds a list from any mix of values.
when_to_use: Use it to gather several values of any type into one list, such as the images for a set of shots.
---

## When to use it

You want to combine separate values into one list, for a node that needs a list input.

## Example

1. Add three images to `items`, either connected from **Load Image** nodes or added directly.
1. `output` is a list with the three images in that order.

## Parameters

### items

Entries of any type, in order. Each entry accepts a connection. Unlike the typed list nodes, typed-in values are not shown in the node body by default.

### list_options

Collapsed group of cleanup options. All are off by default.

- `flatten_list`: if an entry is itself a list, its items are merged into the result. Only one level is flattened.
- `remove_duplicates`: removes repeated entries. The result is built from a set, so the original order is not kept and entries must be hashable.
- `remove_blank`: meant for text lists. It calls `strip()` on each entry, so it fails on entries that are not text.

### output

The resulting list. If `items` is empty, the output is an empty list.
