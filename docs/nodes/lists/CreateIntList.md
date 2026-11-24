---
description: Builds a list of integer values.
when_to_use: Use it to collect whole numbers into one list, such as frame numbers for a set of review frames.
---

## When to use it

You want a list of integer values from typed-in values or from several node outputs.

## Example

Add `1001`, `1050` and `1100`, typed in or connected. `output` is `[1001, 1050, 1100]`.

## Parameters

### items

The integer entries, in order. Each entry accepts a typed value or a connection.

### list_options

Collapsed group of cleanup options. All are off by default.

- `flatten_list`: if an entry is itself a list, its items are merged into the result. Only one level is flattened.
- `remove_duplicates`: removes repeated entries. The result is built from a set, so the original order is not kept.
- `remove_blank`: meant for text lists. It calls `strip()` on each entry, so it fails on integer values. Leave it off here.

### output

The resulting list, typed as `list[int]`.
