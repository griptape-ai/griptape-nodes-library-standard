---
description: Converts any value to text.
when_to_use: Use it when a node needs text and you have a number, list or dictionary.
---

## When to use it

Connect any value and use `output` where text is expected. Text passes through unchanged.

Lists and dictionaries come out in Python form, with single quotes, `True` and `None`. For JSON text use [ToJson](ToJson.md) instead.

## Parameters

### from

The value to convert. It accepts any type. Nothing at all becomes the text `None`.
