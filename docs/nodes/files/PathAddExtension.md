---
description: Adds a file extension to a path, or replaces the one it has.
when_to_use: Use it when you need to change a path's extension, for example from .png to .jpg, or add one to a name that has none.
---

## When to use it

You have a path and need the same path with a different extension. The node does not rename or convert any file. It only changes the text of the path.

## Parameters

### path

The path to change. Anything after the first `?` is treated as query parameters.

### extension

The new extension, with or without the leading dot, so `txt` and `.txt` are the same. If it is empty, the current extension is removed.

Only the last extension is replaced. `archive.tar.gz` with `zip` becomes `archive.tar.zip`.

### strip_query_params

On by default. When on, anything after `?` is dropped from the result. Turn it off to keep it on the end of the new path.

### output

The path with the new extension. It is empty when `path` is empty.
