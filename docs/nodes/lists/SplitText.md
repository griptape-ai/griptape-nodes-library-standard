---
description: Splits text into a list, either at a delimiter or by reading it as a list.
when_to_use: Use it when you have one block of text, such as lines or comma-separated words, and need a list of items.
---

## When to use it

Turn a multi-line prompt, a comma-separated line or a JSON list written as text into a list that loops and list nodes can use. The output updates as you edit.

## Parameters

### split_mode

`split` cuts the text at the chosen delimiter.

`parse_list` reads the text as a list. It tries, in order: JSON such as `["a", "b"]`, a Python list such as `['a', 'b']`, then comma-separated text. If none fits, it splits at `delimiter_type`. Every item comes out as text. In this mode `include_delimiter` is hidden and has no effect.

### delimiter_type

The character to cut at: `newlines`, `double_newline`, `space`, `comma`, `semicolon`, `colon`, `tab`, `pipe`, `dash`, `underscore`, `period`, `slash`, `backslash`, `at`, `hash`, `ampersand`, `equals` or `question mark`. The default is `newlines`.

### include_delimiter

Keeps the delimiter at the end of every item except the last.

### trim_whitespace

Removes spaces at the start and end of each item. In `parse_list` mode, comma-separated items are always trimmed.

### remove_empty

Drops blank items, including items that hold only spaces. It is off by default.

> [!NOTE]
> The node needs text connected or entered. If the text can't be split, the output is an empty list.
