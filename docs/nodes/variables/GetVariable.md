---
description: Reads the value of an existing workflow variable.
when_to_use: Use it when you want to use a stored value somewhere without wiring it from where it was set.
---

## When to use it

Pick a variable from the list and `value` outputs what it holds. The node reruns if the variable changes, so downstream nodes see the current value.

## Parameters

### variable_name

The variable to read. Use the refresh button next to it to reload the list. The node fails if the variable does not exist.

### value

The variable's current value. Its type matches the variable.

### scope

In the collapsed Advanced group. It controls where the node looks for the variable.

| Choice | Looks in |
| --- | --- |
| `hierarchical` | This flow, then parent flows, the project, then globals. The default. |
| `current_flow_only` | Only the flow this node is in. |
| `global_only` | Only global variables. |
| `all` | Every layer, without shadowing. Meant for listing. |
