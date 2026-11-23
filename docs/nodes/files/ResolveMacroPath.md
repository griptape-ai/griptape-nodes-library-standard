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

If the macro can't be resolved, the node fails and the reason is shown in the status details.

### resolved_path

The full absolute path. The node does not check that the file exists.
