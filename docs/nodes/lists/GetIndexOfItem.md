---
description: Finds the position of an item in a list.
when_to_use: Use it when you need the zero-based position of a value in a list, such as where a shot sits in a shot order.
---

## When to use it

You know a value and want to know where it is in a list. Pair it with [Get From List](GetFromList.md) or [Split List](SplitList.md).

## Example

1. Connect a shot list `["SH010", "SH020", "SH030"]` to `items`.
1. Connect `SH030` to `item`.
1. `index` is `2`.

## Parameters

### items

The list to search.

### item

The value to find. Matching uses Python equality and the first match wins, so text is case-sensitive, `1` matches `1.0`, and `0` matches `False`.

### index

The zero-based position. It is `-1` when `items` is empty or not set, when `item` is not set, or when `item` is not in the list. Get From List and Remove From List read `-1` as the last entry, so check for `-1` before passing `index` on.
