---
description: Converts any value to true or false.
when_to_use: Use it when a node needs a true/false input and you have text, a number or a list.
---

## When to use it

Connect a value from another node and use `output` where a switch or condition is needed.

## How values are read

| Input | Result |
| ----- | ------ |
| Nothing | False |
| `true`, `yes`, `y`, `t`, `1` (any case, spaces ignored) | True |
| `false`, `no`, `n`, `f`, `0` (any case, spaces ignored) | False |
| Any other text | True if it isn't blank |
| Numbers | True if not zero |
| Lists, dictionaries and sets | True if not empty |

> [!WARNING]
> Words such as `off`, `none` and `0.0` are not recognized. Any non-blank text outside the lists above counts as true.

## Parameters

### from

The value to convert. It accepts any type.
