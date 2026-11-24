---
description: Checks whether a workflow variable exists and outputs true or false.
when_to_use: Use it when you want to branch on whether a variable has been created yet.
---

## When to use it

Choose a name and read `exists`. Connect it to an [IfElse](../execution/IfElse.md) to run different steps depending on the answer.

## Parameters

### variable_name

The name to check. You can type a name that is not in the list.

### exists

True if the variable is visible at the chosen scope.

### scope

In the collapsed Advanced group. It controls where the node looks for the variable.

| Choice | Looks in |
| --- | --- |
| `hierarchical` | This flow, then parent flows, the project, then globals. The default. |
| `current_flow_only` | Only the flow this node is in. |
| `global_only` | Only global variables. |
| `all` | Every layer, without shadowing. Meant for listing. |
