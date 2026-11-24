---
description: Sets where a node saves a folder of output, using a situation template, a folder name and rules for existing folders.
when_to_use: Use it when a node writes many files into a folder and you want to control the folder name, location and what happens if it already exists.
---

## When to use it

Nodes that save a set of files, such as [ExtractPdfImages](../files/ExtractPdfImages.md), have a `directory` setting. Connect this node to it, using the gear button, to control the folder in one place. The receiving node reads the settings directly. The `directory_destination` output only previews the path.

## How dirname is read

| What you enter | What happens |
| --- | --- |
| A plain name such as `renders` | The situation's macro places and versions the folder. A name like `a/renders` passes `a` as `sub_dirs`. |
| An absolute path, a `~` path or a `file://` URI | Used exactly as written. |
| A name containing `{...}` macros | Used as written, with the macros filled in. |

## Parameters

### situation

Loads a macro, existing-folder rule and folder creation setting into the fields below. Changing it replaces your edits to those fields. The default is the save output directory situation.

### macro

The path template used for plain names. It receives `dir_name` and `sub_dirs`.

### if_dir_exists

What to do when the folder exists: **Increment Version**, **Overwrite Existing** or **Abort / Error**.

This applies to plain names. For an explicit path or a name with macros, it only applies when the path contains a `{###}` version slot. Without one, the folder is used as it is.

### auto_create_path

On by default. Creates missing parent folders.

### directory_destination

A preview of the resolved folder. An unfilled `{###}` slot shows as `###`. It is empty when `dirname` is empty.
