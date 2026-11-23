---
description: Builds a dictionary from a list of keys and a list of values.
when_to_use: Use it when you have separate lists of names and values and want one dictionary to pass to other nodes.
---

## When to use it

You have a list of keys and a matching list of values and want them as one dictionary with named values. Use it to build structured data or configuration settings for other nodes, or data that is easy to read by key.

To build a dictionary from one pair at a time, use [Key Value Pair](KeyValuePair.md) with [Merge Key Value Pairs](MergeKeyValuePairs.md).

## Example

Create a dictionary for a user profile:

1. Add a Dictionary node.
1. Set `keys` to:
    ```
    ["name", "age", "premium_member", "interests"]
    ```
1. Set `values` to:
    ```
    ["Jane Smith", 32, true, ["hiking", "photography", "coding"]]
    ```
1. The `dict` output is:
    ```
    {"name": "Jane Smith", "age": 32, "premium_member": true, "interests": ["hiking", "photography", "coding"]}
    ```
1. Connect it to nodes that need the profile.

## Parameters

### keys

Keys are converted to strings, which can give unexpected results if you use complex objects as keys. Empty or `None` keys are skipped, unless it is the only key and it has a value. You can give a single value instead of a list and it becomes a one-item list.

### values

Strings, numbers, booleans and nested lists all work. A single value is converted to a one-item list.

If the lists differ in length, keys with no matching value get `None`, and extra values may be ignored.
