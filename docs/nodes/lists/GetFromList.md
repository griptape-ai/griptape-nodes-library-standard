---
description: Returns the item at a position in a list.
when_to_use: Use it to pick out the first, last or nth entry of a list, such as the first frame in a list of renders.
---

## When to use it

You have a list and want one entry from it, by position.

## Example

1. Connect a list of five images to `items`.
1. Set `position` to `index` and `index` to `2`. `item` is the third image.
1. Set `position` to `end` to get the last image instead.

## Parameters

### position

`index` (the default), `start` or `end`. `start` and `end` hide the `index` field.

### index

Zero-based. Accepts an integer or a float, which is truncated to an integer. Negative values count from the end.

### item

The entry at that position, with the `any` type. It is empty (None) when the list is empty, the index is out of range, or the index is not set.
