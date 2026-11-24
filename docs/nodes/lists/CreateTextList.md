---
description: Builds a list of text values.
when_to_use: Use it to collect text entries into one list, such as shot codes or department names.
---

## When to use it

You want a list of strings from typed-in values or from several text outputs.

## Example

Add items `SH010`, `SH020` and `SH030`, either typed in or connected from other nodes. `output` is `["SH010", "SH020", "SH030"]`.

## Parameters

### items

Text entries, in order. Each entry accepts a typed value or a connection.

### list_options

Collapsed group of cleanup options. All are off by default.

- `flatten_list`: if an entry is itself a list, its items are merged into the result. Only one level is flattened.
- `remove_duplicates`: removes repeated entries. The result is built from a set, so the original order is not kept.
- `remove_blank`: removes empty and whitespace-only entries.

### output

The resulting list, typed as `list[str]`.
