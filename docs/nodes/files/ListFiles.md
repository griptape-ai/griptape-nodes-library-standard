---
description: Lists the files and folders in a directory, with optional pattern matching, recursion and sorting.
when_to_use: Use it when you want to process every file in a folder, or every file of one type, such as all the .png files in a render directory.
---

## When to use it

You have a folder and want its contents as a list. Use the lists to loop over images, pick the newest render or check what a job produced.

## Example

To get every PNG in any subfolder of a render directory, newest first:

| Parameter | Value |
| --- | --- |
| `directory_path` | `/projects/renders` |
| `match_pattern` | `**/*.png` |
| `sort_by` | Date Modified |
| `sort_order` | Descending |

## Parameters

### directory_path

The folder to list. If empty, the workspace root is listed. It is required when `match_pattern` contains `/` or `**`.

### match_pattern

Without a slash, the pattern is matched against each name, so `*.jpg` finds JPEGs in the listed folder. With a `/` or `**`, it is matched against the path relative to `directory_path`, and the search goes into subfolders on its own. `?` matches one character, and `[abc]` or `[!abc]` match a set of characters. Backslashes are read as `/`.

If the pattern isn't valid, the node fails.

### recursive

Searches subfolders at every depth. It is switched on automatically for a path pattern. Symlink loops are not followed twice.

### list_options

Choose files and folders, files only or folders only. The pattern is applied after this filter.

### use_absolute_paths

Off by default, which returns paths as the system gives them and these may be relative. Turn it on for full paths.

### sort_by

With Unsorted, the order is whatever the filesystem returns. Date Modified uses the last modified time.

### file_paths

The matched entries, files and folders as set by `list_options`.

### file_names

The names only, in the same order as `file_paths`.

### file_count

The number of entries returned. Its tooltip says "files", but folders are counted too when `list_options` includes them.
