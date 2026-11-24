---
description: Turns inline variable substitution on or off for the workflow.
when_to_use: Use it when you want to write {VARIABLE_NAME} inside a parameter value and have it replaced with the variable's value.
---

## When to use it

With substitution enabled, a token such as `{shot}` inside a parameter value is replaced with the value of the variable `shot` when the workflow runs. Run this node before the nodes that rely on it.

## Parameters

### enabled

On by default. The setting applies to the current workflow only and is saved with it. Running the node with it off disables substitution again.
