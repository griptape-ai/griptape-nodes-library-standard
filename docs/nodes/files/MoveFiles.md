---
description: Moves files and folders to a destination folder, with support for multiple sources and patterns.
when_to_use: Use it when you want files in a new location and no longer in the old one.
---

## When to use it

You want to file outputs away, or collect files into one folder, without leaving the originals behind. To keep the originals, use [CopyFiles](CopyFiles.md). To change only a name, use [RenameFile](RenameFile.md).

## Parameters

### source_paths

The files, folders or patterns to move. Add one item per path or connect a list. A pattern such as `/path/*.txt` moves every match.

### destination_path

Where to move to. Project macro paths are resolved.

- A path with no file extension is treated as a folder, and each source keeps its own name inside it.
- A path with an extension is treated as the new file name. This only works with a single source.

### overwrite

Off by default. When on, anything already at the destination is deleted before the move.

### moved_paths

The destination path of each item moved.

> [!NOTE]
> The node also shows a `copied_paths` output inherited from [CopyFiles](CopyFiles.md). It is never filled. Use `moved_paths`.

## Results

The node fails if no source was valid or nothing could be moved. If at least one item was moved, it succeeds even when others failed, and the status details list each failure.
