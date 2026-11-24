---
description: Converts text, numbers or dictionaries to a whole number.
when_to_use: Use it when a node needs a whole number and you have text such as "12 frames" or a decimal.
---

## When to use it

Connect any value. If no number can be found, the result is `0`.

## How values are read

| Input | Result |
| ----- | ------ |
| A number | Decimals are cut off, not rounded, so `3.9` gives `3` |
| Text | The first whole number found in it, so `"12 frames"` gives `12` |
| A dictionary | The first value that converts to a non-zero number |
| Nothing, or anything unreadable | `0` |

> [!WARNING]
> In text, only digits before a decimal point are read. `"3.9"` gives `3` and `"1,234"` gives `1`.

## Parameters

### from

The value to convert. It accepts any type.
