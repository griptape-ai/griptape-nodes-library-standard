---
description: Creates a folder, optionally with any missing parent folders.
when_to_use: Use it when a workflow needs a folder to exist before it writes files into it.
---

## When to use it

You want to be sure a folder is there before saving into it. By default the node is safe to run again, because a folder that already exists counts as success.

## Parameters

### folder_path

The folder to create. If a file already exists at that path, the node fails.

### create_parents

On by default, and creates any missing parent folders. When off, the node fails if the parent folder doesn't exist.

### fail_if_already_exists

Off by default. Turn it on to make the node fail when the folder is already there.

### created_path

The folder path, whether it was just created or already existed.

### already_existed

True when the folder was there before the node ran.
