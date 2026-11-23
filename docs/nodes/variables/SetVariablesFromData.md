---
description: Creates or updates many workflow variables at once from a dictionary, JSON, YAML or a list of key and value pairs.
when_to_use: Use it when you have a set of named values, such as shot settings, and want each one as a variable.
---

## When to use it

Instead of one [Create Variable](CreateVariable.md) node per value, give this node a block of data. Each key becomes a variable and the type is guessed from the value.

## Example

Set `data` to:

```yaml
shot: banana
seq: "01"
```

This creates `shot` and `seq`, and `variable_names` outputs `["shot", "seq"]`.

## Parameters

### data

Accepts:

- A dictionary.
- A string. The node tries JSON first, then YAML, then plain lines of `key: value` or `key=value`. Values from the plain-line form are always strings.
- A list of pairs, either `[["NAME", "Jason"]]` or `[{"key": "NAME", "value": "Jason"}]`. A single-entry dictionary per item also works.

If a key repeats, the last value wins. Empty data fails the node.

Types are guessed as bool, int, float, str, list, json for dictionaries, and any for the rest.

### sanitize_names

On by default. Keys that are not valid names, such as `Full Name`, are turned into safe ones such as `Full_Name`, and a leading digit gets an underscore. Keys are merged after cleaning, so `Full Name` and `Full_Name` count as one. When off, an invalid key fails the node.

### collision_behavior

What to do when a variable already exists:

- `Overwrite existing` replaces its value and type.
- `Preserve existing` leaves it alone and does not list it in the output.
- `Error on collision` fails the node.

### variable_names

The names that were written, in order.

### scope

In the collapsed Advanced group. It controls where the node looks for the variable.

| Choice | Looks in |
| --- | --- |
| `hierarchical` | This flow, then parent flows, the project, then globals. The default. |
| `current_flow_only` | Only the flow this node is in. |
| `global_only` | Only global variables. |
| `all` | Every layer, without shadowing. Meant for listing. |
