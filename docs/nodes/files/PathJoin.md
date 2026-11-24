---
description: Joins several path pieces into one path using the right separator for your operating system.
when_to_use: Use it when you build a path from a folder, a subfolder and a filename that come from different places.
---

## When to use it

You have a folder in one node and a filename in another and need the full path. Add each piece as an item, in order, and the node joins them with `/` or `\` as the system needs.

## Parameters

### path_components

The pieces to join, in order. Empty items are skipped. Line breaks are stripped from each piece.

If a later piece is an absolute path, it replaces everything before it. For example, `/projects` and `/tmp/out.png` joins to `/tmp/out.png`.

### output

The joined path. It is empty when every piece is empty.
