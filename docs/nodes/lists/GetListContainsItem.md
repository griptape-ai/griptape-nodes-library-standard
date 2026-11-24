---
description: Checks whether a list contains a given item and outputs true or false.
when_to_use: Use it to branch or validate on whether a value is in a list, such as whether a shot code is already in a delivery list.
---

## When to use it

You want a yes or no answer to "is this value in this list?".

## Example

1. Connect a list of delivered shot codes to `items`.
1. Connect `SH020` to `item`.
1. `contains_item` is true if `SH020` is in the list.

## Parameters

### items

The list to search.

### item

The value to look for. Matching uses Python equality, so text comparisons are case-sensitive, `1` matches `1.0`, and `0` matches `False`.

### contains_item

True or false. It is false when `items` is empty or not set, or when `item` is not set. The output is also updated when `items` changes or is disconnected, without running the node.
