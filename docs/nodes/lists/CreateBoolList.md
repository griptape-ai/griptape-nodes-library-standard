---
description: Builds a list of boolean values.
when_to_use: Use it to collect true/false flags into one list, such as per-shot approval flags.
---

## When to use it

You want a list of boolean values from typed-in values or from several node outputs.

## Example

Add `true`, `false` and `true`, typed in or connected. `output` is `[true, false, true]`.

## Parameters

### items

The boolean entries, in order. Each entry accepts a typed value or a connection.

### list_options

Collapsed group of cleanup options. All are off by default.

- `flatten_list`: if an entry is itself a list, its items are merged into the result. Only one level is flattened.
- `remove_duplicates`: removes repeated entries. The result is built from a set, so the original order is not kept.
- `remove_blank`: meant for text lists. It calls `strip()` on each entry, so it fails on boolean values. Leave it off here.

### output

The resulting list, typed as `list[bool]`.
