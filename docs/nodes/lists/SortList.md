---
description: Sorts a list in ascending or descending order, including lists of dictionaries by a chosen key.
when_to_use: Use it when you need a list ordered by value, name or by one field of each dictionary.
---

## When to use it

Order numbers, text or records before they go on to the next node. The original list is not changed.

## How items are compared

- Numbers, and text that reads as a number such as `"42"`, are compared as numbers.
- Everything else is compared as text.
- Numbers come before text in ascending order.
- Artifacts, such as images and videos, are compared by their URL or path, not by name.
- Dictionaries without a `key` are compared by their JSON text.

An empty list, or an input that can't be read as a list, gives an empty list.

## Parameters

### items

Accepts a list, or JSON text holding a list.

### sort_order

`asc` for ascending, `desc` for descending.

### key

Appears only when every item is a dictionary, or text that reads as a dictionary. The choices are every key found in any of the items. Items missing the key sort as empty text.
