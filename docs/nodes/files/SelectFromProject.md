---
description: Picks a file or folder and returns it as a project macro path, such as {inputs}/image.png, when it is inside the project.
when_to_use: Use it when you want to choose a file once and keep a path that still works when the project moves to another machine.
---

## When to use it

A normal file picker stores a full path like `/Users/me/project/inputs/image.png`, which breaks on another computer. This node swaps the start of the path for a project macro, such as `{inputs}`, when the file is inside one of the project's folders.

## Parameters

### selected_path

The file or folder to pick. After you choose, the field is rewritten to the macro form when the path is inside the project, so you can see what is stored. A path outside the project is left as it is.

### project_path

The path as a URL artifact. It holds the macro path when `in_project` is True, and the path you picked otherwise. It is empty when nothing is selected.

### in_project

True when the path is inside the project and was converted to a macro path. Use it to warn or branch when a file is not portable.
