---
description: Runs a block of Python code and returns the value it assigns to `result`.
when_to_use: Use it when you need a small custom calculation or transformation that no existing node covers.
---

## When to use it

You need a bit of logic, such as reshaping a dictionary or doing custom math, and there is no node for it. Write the code, pass in the values it needs, and read the answer from `result`.

## Example

Set `input_variables` to `{"width": 1920, "height": 1080}` and `python_code` to:

```python
result = width / height
```

`result` outputs `1.7777777777777777`.

## Parameters

### python_code

The code to run. It must assign the answer to a variable named `result`. If the code never assigns `result`, the node fails.

### input_variables

A dictionary of names and values. Each pair is turned into a variable at the top of your code, so `{"width": 1920}` lets you write `width` directly. Values are written into the code as Python literals, so plain data such as numbers, strings, lists and dictionaries work. Objects that can't be written as literals do not.

### result

The value of `result` after the code runs. On any failure it is empty, except when no code is given, where it holds a short message.

### Failure handling

If the code raises an error, the node fails and the error output is shown in the status details. When the Failed output is wired, the workflow continues down it. When it isn't, the error is raised and stops the run.

> [!WARNING]
> The code runs inside the engine process with full Python access. Only run code you trust.
