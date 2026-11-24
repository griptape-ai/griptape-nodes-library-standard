---
description: Turns a list of project macro paths into full paths on disk.
when_to_use: Use it when you have several project macro paths, such as {inputs}/a.png, and need the real filesystem paths for all of them.
---

## When to use it

This is the list version of [ResolveMacroPath](ResolveMacroPath.md). Add one item per path, or connect a list. Each item can be text or a file artifact.

## Parameters

### paths

The macro paths to resolve. Empty items are skipped, so the output can have fewer entries than the input.

A plain path with no macro is anchored to the workspace folder, and a URL comes back unchanged. Braces that don't name a project variable, as in `notes {draft}.txt`, are treated as part of the filename and the path comes back unchanged.

If any project macro can't be resolved, the node fails and no paths are returned.

### resolved_paths

The full absolute paths, in the same order as the inputs. URLs and filenames with braces come back unchanged. The node does not check that the files exist.
