---
description: Turns a project macro path such as {inputs}/file.txt into a full path on disk.
when_to_use: Use it when a node needs a real filesystem path and you only have a project macro path.
---

## When to use it

Project paths often use macros like `{inputs}` or `{outputs}` so a workflow works on any machine. Some nodes and tools need the full path. This node expands the macro for the current project.

For several paths at once, use [ResolveMacroPaths](ResolveMacroPaths.md).

## Parameters

### path

The macro path to resolve, for example `{inputs}/my_file.png`. It accepts a text path or a file artifact such as an image or video. An empty path gives an empty result.

A plain path with no macro is anchored to the workspace folder, and a URL comes back unchanged. Braces that don't name a project variable, as in `notes {draft}.txt`, are treated as part of the filename and the path comes back unchanged.

If a project macro can't be resolved, the node fails and the reason is shown in the status details.

### resolved_path

The full absolute path, or the input unchanged for a URL or a filename with braces. The node does not check that the file exists.
