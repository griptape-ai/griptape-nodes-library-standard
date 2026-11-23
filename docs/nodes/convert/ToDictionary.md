---
description: Converts text, a list or another value into a dictionary.
when_to_use: Use it when a node needs a dictionary and you have JSON text, key: value lines, or a list.
---

## When to use it

Connect any value and get a dictionary. If the value can't be read as one, the result is an empty dictionary or a dictionary with a single `value` entry.

## How values are read

| Input | Result |
| ----- | ------ |
| A dictionary | Unchanged |
| Nothing | Empty dictionary |
| Text that looks like a dictionary, in Python or JSON form | That dictionary |
| Other JSON text | `{"value": <parsed JSON>}` |
| Text with `key: value` or `key=value` lines | One entry per line |
| Other text | `{"value": "<text>"}` |
| A list | `item1`, `item2` and so on, one key per item |
| An object with public attributes | Those attributes |
| Any other value | `{"value": <value>}` |

If a conversion fails, the result is an empty dictionary.

## Parameters

### from

The value to convert. It accepts any type.
