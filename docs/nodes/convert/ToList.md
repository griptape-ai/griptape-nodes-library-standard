---
description: Converts any value to a list.
when_to_use: Use it when a node needs a list and you have a single value, a dictionary or JSON text.
---

## When to use it

Connect any value and get a list. A list passes through unchanged.

## How values are read

| Input | Result |
| ----- | ------ |
| Nothing | Empty list |
| Text starting with `[` or `{` that is valid JSON | The parsed list or dictionary, converted by these same rules |
| A dictionary with one key whose value is a list | That list |
| Any other dictionary | A list of its values |
| A set or tuple | A list of its items |
| Anything else | A list holding that one value |

Plain text is not split into pieces. Use [SplitText](../lists/SplitText.md) for that.

## Parameters

### from

The value to convert. It accepts any type.
