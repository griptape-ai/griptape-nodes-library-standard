---
description: Checks a list of paths and returns the ones that exist, are files, end in .py and can be read.
when_to_use: Use it when you have a list of Python file paths and want to keep only the usable ones and see why the others were rejected.
---

## When to use it

You have a list of file paths and want to be sure each one points to a readable Python file before using it. The node sorts the list into valid paths and a text list of problems.

> [!NOTE]
> Despite the general name, this node only accepts Python files. A path to any other file type, such as a `.txt` or `.png`, is reported as `Not a Python file`.

## Parameters

### file_paths

The paths to check. A single value that isn't a list is treated as a list of one.

Each path is checked in this order, and the first failed check is reported: it exists, it is a file and not a folder, it ends in `.py` (any capitalization), and it can be read.

### valid_paths

The paths that passed, as absolute paths.

### validation_errors

One line per rejected path, such as `File does not exist: <path>`. The text is empty when every path passed.
