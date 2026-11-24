---
description: Converts text, numbers or dictionaries to a decimal number.
when_to_use: Use it when a node needs a decimal number and you have text such as "3.5 kg" or an LLM reply.
---

## When to use it

Connect any value. If no number can be found, the result is `0.0`.

## How values are read

| Input | Result |
| ----- | ------ |
| A number | That number as a decimal |
| Text | The first number found in it, so `"about 3.5 kg"` gives `3.5` |
| A dictionary | The first value that converts to a non-zero number |
| Nothing, or anything unreadable | `0.0` |

> [!WARNING]
> Only the first number in text is used. `"1,234"` gives `1.0`, and `".5"` gives `5.0`.

## Parameters

### from

The value to convert. It accepts any type.
