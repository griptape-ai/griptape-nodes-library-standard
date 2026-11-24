---
description: Joins the items of a list into one block of text with a separator, and optional text before and after each item.
when_to_use: Use it when you need to turn a list into a single string, such as a bulleted list or comma-separated line.
---

## When to use it

Turn a list of lines, tags or results into one text for a prompt or a file. The output updates as soon as an input changes.

## Example

| items | separator | item_prefix | output |
| ----- | --------- | ----------- | ------ |
| `["red", "blue"]` | `, ` | | `red, blue` |
| `["red", "blue"]` | `\n` | `- ` | two lines: `- red` and `- blue` |

## Parameters

### items

Items that aren't text are converted to text. Empty items become empty text.

### separator

Defaults to a new line. Type `\n` for a new line, `\t` for a tab and `\\` for a backslash. The same escapes work in `item_prefix` and `item_suffix`.

### item_prefix

Added before each item, before the separator is applied.

### item_suffix

Added after each item, before the separator is applied.
