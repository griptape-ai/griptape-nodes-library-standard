---
description: Checks whether a file or folder exists at a path and reports whether it is a folder.
when_to_use: Use it when a workflow should behave differently depending on whether a file or folder is already there.
---

## When to use it

You want to know if something is on disk before you read, save or delete it. Connect `exists` to a conditional node to branch on the answer.

## Parameters

### path

The file or folder to check. Project macro paths such as `{outputs}/image.png` are resolved first. If a macro can't be resolved, the node fails with an error.

An empty path gives `exists` and `is_directory` both False.

### exists

True when something is at the path, whether a file or a folder.

### is_directory

True only when the path exists and is a folder.
