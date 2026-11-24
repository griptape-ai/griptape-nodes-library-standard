---
description: Turns a list that contains other lists into one flat list, keeping the order.
when_to_use: Use it when a list has nested lists inside it and the next node needs a single level.
---

## When to use it

Loops and splits often give you a list of lists. This node pulls the items out so you get one plain list.

## Example

With `items` set to `[[1, [2, 3]], 4]`:

| depth | output |
| ----- | ------ |
| 0 | `[1, 2, 3, 4]` |
| 1 | `[1, [2, 3], 4]` |

## Parameters

### items

Accepts a list or JSON text. A single value that isn't a list is treated as a list of one item. Only lists are opened up. Text, dictionaries and other values stay whole.

### depth

How many levels of nesting to remove. `0` removes every level.
