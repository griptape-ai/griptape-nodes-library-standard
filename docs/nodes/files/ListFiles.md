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

The folder to list. If empty, the workspace root is listed. It is required when `match_pattern` contains `/` or `**`, and the node fails without it. Use a full path. A relative path such as `tree/other` fails.

### match_pattern

Leading and trailing spaces are removed. Without a slash, the pattern is matched against each name, so `*.jpg` finds JPEGs in the listed folder. With a `/` or `**`, it is matched against the path relative to `directory_path`, and the search goes into subfolders on its own. `*` stays within one folder level, and `**/` matches zero or more folders, so `**/*.png` also finds PNGs directly in `directory_path`. `?` matches one character, and `[abc]` or `[!abc]` match a set of characters. Backslashes are read as `/`, so they cannot be used to escape characters.

The pattern is applied to folders as well as files, so `s?b` with `list_options` set to folders only finds a folder named `sub`. Matching is case-sensitive unless `match_pattern_case_sensitive` is turned off.

The node fails only when a path pattern cannot be compiled, such as a bad character range like `**/[z-a]`. An unclosed `[` is treated as a literal character, and a name pattern such as `[z-a]` simply matches nothing.

### match_pattern_case_sensitive

On by default, so `*.png` does not match `B.PNG`. Turn it off to match any case. It has no effect when `match_pattern` is empty.

### show_hidden

Off by default. Names that start with a dot, such as `.hidden.png` and the contents of a `.hid` folder, are left out. Turn it on to include them, including during recursion.

### recursive

Searches subfolders at every depth. With a pattern that has no slash, the name is matched at every depth. It is switched on automatically for a path pattern. A folder reached through a symlink is skipped if its real path was already listed, so symlinks to a folder already visited are listed as entries but their contents are not repeated.

### list_options

Choose `List files and folders` (the default), `List files only` or `List folders only`. The pattern is applied after this filter, so a `*.png` pattern with folders only returns nothing.

### use_absolute_paths

Off by default. When off, the node returns the paths as the engine gives them. When on, it returns each entry's absolute path.

### sort_by

With Unsorted, the order is whatever the engine returns, which is not alphabetical, and `sort_order` is ignored. Alphabetical sorts by name, ignoring case, so `B.PNG` sorts between `a.png` and `c.jpg`. In a recursive listing it sorts by name only, not by folder, so `sub/d.png` comes after `c.jpg`. Date Modified uses the last modified time.

### sort_order

Ascending (the default) puts A to Z or oldest first. Descending reverses that. It is ignored when `sort_by` is Unsorted.

### file_paths

The matched entries, files and folders as set by `list_options`.

### file_names

The names only, in the same order as `file_paths`.

### was_successful and result_details

The node also has a `failure` control output, `was_successful` and `result_details`. On success, `result_details` is `Success`. If the folder cannot be listed (for example it does not exist, or `directory_path` is a file) or a path pattern is invalid, `was_successful` is false and `result_details` starts with `Failure:`. If the `failure` output is not connected, the node raises the error and stops the workflow. The file outputs are not cleared on failure, so they can hold the list from an earlier run.

### file_count

The number of entries returned. Its tooltip says "files", but folders are counted too when `list_options` includes them.
