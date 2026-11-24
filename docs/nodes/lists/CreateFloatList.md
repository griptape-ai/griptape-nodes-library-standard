---
description: Builds a list of float values.
when_to_use: Use it to collect decimal numbers into one list, such as per-shot render times in minutes.
---

## When to use it

You want a list of float values from typed-in values or from several node outputs.

## Example

Add `12.5`, `8.0` and `31.25`, typed in or connected. `output` is `[12.5, 8.0, 31.25]`.

## Parameters

### items

The float entries, in order. Each entry accepts a typed value or a connection.

### list_options

Collapsed group of cleanup options. All are off by default.

- `flatten_list`: if an entry is itself a list, its items are merged into the result. Only one level is flattened.
- `remove_duplicates`: removes repeated entries. The result is built from a set, so the original order is not kept.
- `remove_blank`: meant for text lists. It calls `strip()` on each entry, so it fails on float values. Leave it off here.

### output

The resulting list, typed as `list[float]`.
