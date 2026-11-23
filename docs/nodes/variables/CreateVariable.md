---
description: Creates a workflow variable with a name, type and value, or updates it if it already exists.
when_to_use: Use it when you want to store a value once and read it from other parts of the workflow.
---

## When to use it

Variables hold a value under a name so that other nodes, such as [Get Variable](GetVariable.md), can read it without a wire. Use this node to define one at the start of a flow. If a variable with that name already exists in this flow, its type and value are updated instead.

The variable always belongs to the flow this node is in. There is no scope setting.

## Parameters

### variable_name

The name to store the value under. Wire in a string or type one.

### auto_name

Fills in `variable_name` for you when a value is connected to `value`. The name comes from the first of these that applies: the file name of an image, audio, video or other file value, the file name in a path string, or the name of the node that feeds `value`. Version and frame suffixes such as `_v001` or `_001` are removed.

### auto_name_case

Only shown when `auto_name` is on. Sets the style of the generated name: UPPER CASE, UPPER_SNAKE_CASE, Title Case, snake_case, kebab-case, PascalCase, camelCase or as is. The default is snake_case.

### variable_type

The type of the variable, such as `str`, `int` or `JSON`. When a node is connected to `value`, the type is taken from it and this field is locked. Clearing the type resets `value` to accept anything. Changing it removes connections to `value` that the new type does not accept.

### value

The initial value. When the variable already exists, this replaces its value.

> [!NOTE]
> The node runs on every workflow run so the variable is always written.
