---
description: Deletes one or more workflow variables by name.
when_to_use: Use it when you want to clear stored values, for example between runs or at the end of a stage.
---

## When to use it

Give it a name or a list of names and the variables are removed. Names that are not present are skipped unless you ask otherwise.

## Parameters

### variable_names

A single name or a list of names. Spaces around names are trimmed and blank and repeated names are ignored. You can type names that are not in the list.

### fail_if_missing

Off by default. When on, a name that does not exist fails the node. Other names are still deleted.

### deleted_names

The names that were deleted.

### Failure handling

If any deletion fails, such as a read-only variable, or a missing name with `fail_if_missing` on, the node fails after trying every name. The error says how many were deleted. The node reruns if a named variable exists again.

> [!NOTE]
> A variable that is only mentioned by this node is not saved with the workflow.

### scope

In the collapsed Advanced group. It controls where the node looks for the variable.

| Choice | Looks in |
| --- | --- |
| `hierarchical` | This flow, then parent flows, the project, then globals. The default. |
| `current_flow_only` | Only the flow this node is in. |
| `global_only` | Only global variables. |
| `all` | Every layer, without shadowing. Meant for listing. |
