---
description: Copies files and folders to a destination folder, with support for multiple sources and patterns.
when_to_use: Use it when you need a copy of files or folders in another location and want the originals left alone.
---

## When to use it

You want to back up outputs, gather files into one folder or duplicate a folder. Folders are copied with everything inside them. To remove the originals afterwards, use [MoveFiles](MoveFiles.md) instead.

## Parameters

### source_paths

The files, folders or patterns to copy. Add one item per path or connect a list. A pattern such as `/path/*.txt` copies every match. Duplicates are removed.

### destination_path

Where to copy to. Project macro paths such as `{outputs}/archive` are resolved.

- A path with no file extension is treated as a folder, and each source keeps its own name inside it.
- A path with an extension is treated as the new file name. This only works with a single source. Copying several items to one file path fails.

### overwrite

Off by default. When on, existing files at the destination are replaced. For folders, it lets the copy merge into a folder that already exists.

### copied_paths

The destination path of each item copied.

## Results

The node fails if no source was valid or nothing could be copied. If at least one item was copied, it succeeds even when others failed, and the status details list each failure. A progress bar shows how many items are done.
