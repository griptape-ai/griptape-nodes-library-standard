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

The folder to list. If empty, the workspace root is listed. It is required when `match_pattern` contains `/` or `**`. Use a full path, because a relative path such as `tree/other` fails.

### match_pattern

Leading and trailing spaces are removed. Without a slash, the pattern is matched against each name, so `*.jpg` finds JPEGs in the listed folder. With a `/` or `**`, it is matched against the path relative to `directory_path`, and the search goes into subfolders on its own.

- `*` stays within one folder level. `**/` matches zero or more folders, so `**/*.png` also finds PNGs directly in `directory_path`.
- `?` matches one character. `[abc]` and `[!abc]` match a set of characters.
- The pattern applies to folders as well as files. Matching is case-sensitive unless `match_pattern_case_sensitive` is turned off.
- Backslashes are read as `/`, so they cannot escape characters.

The node fails when a path pattern cannot be compiled, such as a bad character range like `**/[z-a]`.

### match_pattern_case_sensitive

On by default, so `*.png` does not match `B.PNG`. Turn it off to match any case. It has no effect when `match_pattern` is empty.

### show_hidden

Off by default. Names that start with a dot, such as `.hidden.png` and the contents of a `.hid` folder, are left out. Turn it on to include them.

### recursive

Searches subfolders at every depth. It is switched on automatically for a path pattern. A symlinked folder whose real path was already listed is shown as an entry but its contents are not repeated.

### list_options

Choose `List files and folders` (the default), `List files only` or `List folders only`. The pattern is applied after this filter, so a `*.png` pattern with folders only returns nothing.

### use_absolute_paths

Off by default, so paths are returned as the engine gives them. Turn it on for each entry's absolute path.

### sort_by

With Unsorted, the order is whatever the engine returns, which is not alphabetical, and `sort_order` is ignored. Alphabetical sorts by name, ignoring case, so `B.PNG` sorts between `a.png` and `c.jpg`. In a recursive listing it sorts by name only, not by folder. Date Modified uses the last modified time.

### sort_order

Ascending (the default) puts A to Z or oldest first. Descending reverses that.

### file_paths

The matched entries, files and folders as set by `list_options`.

### file_names

The names only, in the same order as `file_paths`.

### was_successful and result_details

The node also has a `failure` control output.

On success, `result_details` is `Success`. If the folder cannot be listed, for example it does not exist or `directory_path` is a file, or a path pattern is invalid, `was_successful` is false and `result_details` starts with `Failure:`. If the `failure` output is not connected, the node raises the error and stops the workflow.

### file_count

The number of entries returned. Its tooltip says "files", but folders are counted too when `list_options` includes them.

## Edge cases

- The file outputs are not cleared on failure, so they can hold the list from an earlier run.
- An unclosed `[` is treated as a literal character, and a name pattern such as `[z-a]` matches nothing.
