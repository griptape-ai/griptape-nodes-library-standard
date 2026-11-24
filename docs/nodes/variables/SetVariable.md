---
description: Sets the value of a workflow variable and creates the variable if it does not exist.
when_to_use: Use it when a later step needs to change a stored value, or when you want to create a variable from a connected value.
---

## When to use it

Pick an existing variable or choose **Create new variable** and type a name. When the node runs, the variable gets the new `value`. Nodes like [Get Variable](GetVariable.md) then see it.

## Parameters

### variable_name

The variable to set. Choose **Create new variable** to type a name in `new_variable_name`. The variable is created as soon as you pick a name, so it shows up in other nodes before the workflow runs. Changing the name points the node at a different variable. It does not rename the old one.

### new_variable_name

Only shown with **Create new variable**. The name for the new variable.

### value

The new value. When a node is connected here, the variable's type is set from that node's output. Disconnecting does not reset the variable's type.

### scope

In the collapsed Advanced group. It controls where the node looks for the variable.

| Choice | Looks in |
| --- | --- |
| `hierarchical` | This flow, then parent flows, the project, then globals. The default. |
| `current_flow_only` | Only the flow this node is in. |
| `global_only` | Only global variables. |
| `all` | Every layer, without shadowing. Meant for listing. |
