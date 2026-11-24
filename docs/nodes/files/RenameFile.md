---
description: Renames a file or folder, in place or to a full new path.
when_to_use: Use it when you need to change the name of a file or folder, or move a single item to a specific new path.
---

## When to use it

You have a file and want it called something else. Give just a new name to rename it where it is, or a full path to rename and move it in one step. For several items at once, use [MoveFiles](MoveFiles.md).

## Parameters

### old_path

The file or folder to rename. Project macro paths are resolved. The node fails if it doesn't exist.

### new_path

A bare name such as `final.png` renames the item inside its current folder. An absolute path, or a relative path with folders in it, is used as given.

### overwrite

Off by default, so the node fails if `new_path` already exists. When on, the existing item at `new_path` is deleted first, then the rename happens. If the rename then fails, the old destination is already gone.

### old_path_output

The path that was renamed. It is empty if the node failed.

### new_path_output

The path after renaming. It is empty if the node failed.
