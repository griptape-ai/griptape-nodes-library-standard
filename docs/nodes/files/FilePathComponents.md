---
description: Splits a file path into its filename, stem, extension, parent folder, parts and query string.
when_to_use: Use it when you need one piece of a path, such as the extension or the name without the extension.
---

## When to use it

You have a path like `/projects/shots/render.v2.png?t=123` and need part of it, for example to build an output name or choose a branch by extension. Each part comes out on its own output.

## Example

For `/projects/shots/render.v2.png?t=123`:

| Output | Value |
| --- | --- |
| `filename` | `render.v2.png` |
| `stem` | `render.v2` |
| `extension` | `.png` |
| `parent` | `/projects/shots` |
| `parent_name` | `shots` |
| `query_params` | `t=123` |
| `parts` | `["/", "projects", "shots", "render.v2.png"]` |

## Parameters

### path

The path to split. Anything after the first `?` is treated as a query string and removed before the other parts are worked out. The outputs update as soon as you change the path.

### parent

For a bare filename such as `file.txt`, this is `.`.

### extension

Includes the dot. Only the last extension is returned, so `archive.tar.gz` gives `.gz`.
