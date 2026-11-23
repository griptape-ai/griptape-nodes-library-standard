---
description: Deletes files or folders, including everything inside a folder.
when_to_use: Use it when a workflow needs to clean up files it created, such as temporary renders or old outputs.
---

## When to use it

You want a workflow to remove files or folders when it finishes with them. On engine 0.114 and later, items go to the trash or recycle bin when the system has one, and are deleted permanently only when that fails. Don't count on getting them back. The warning on the node says deletes are permanent, and older engines may delete permanently without trying the trash.

> [!WARNING]
> Deleting a folder deletes everything inside it. Check the preview on the node before you run it.

## Parameters

### file_paths

The files, folders or patterns to delete. Add one item per path or connect a list. Duplicates are removed. Project macro paths such as `{outputs}/old` are resolved.

A pattern like `/path/to/*.txt` deletes every match in that folder. Wildcards are only supported in the last part of the path. A wildcard in a parent folder is reported as invalid.

### Warning panel

The panel on the node lists what will be deleted, grouped by folder, and updates when you change `file_paths`. For a folder, it lists the contents too, up to 50 items. Click **Refresh List** to scan again if the files have changed. After a run, it shows the result instead.

### deleted_paths

Every path that was deleted. Paths that were invalid or failed are not listed.

## Results

An empty list of paths counts as a success and deletes nothing. The node fails if no path could be deleted. If at least one was deleted, the node succeeds even when others failed, and the status details list each failure and invalid path.
