---
description: Splits a list into two lists at an index or at the first occurrence of an item.
when_to_use: Use it to separate the front of a list from the rest, such as the shots before and after a given shot.
---

## When to use it

You need the part of a list before a point and the part from that point on.

## Example

Given `["SH010", "SH020", "SH030", "SH040"]`:

- `split_by` `index` with `split_index` 2 gives `["SH010", "SH020"]` in `output_a` and `["SH030", "SH040"]` in `output_b`.
- `split_by` `item` with `split_item` `SH030` gives the same result. With `keep_split_item` off, `output_b` is `["SH040"]`.

## Parameters

### split_by

`index` (the default) or `item`. Only the fields for the chosen mode are shown.

### split_index

Zero-based. The item at this index goes into `output_b`. It must be between 0 and the last index. Otherwise the node produces no output.

### split_item

The first matching item is the split point. If the item is not in the list, the node produces no output.

### keep_split_item

On by default. When on, the split item is the first entry of `output_b`. When off, it is dropped.

### output_a, output_b

The items before the split point and the items from it onward. Neither is filled when `items` is empty. The input list is not modified.
