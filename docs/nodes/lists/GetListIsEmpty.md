---
description: Outputs true if a list has no items and false if it has any.
when_to_use: Use it when a workflow should branch or wait on whether a list has content, such as checking for pending tasks.
---

## When to use it

You want to know whether a list has any items before you process it, for example to check that data is available or that a task list still has pending tasks. The node works with lists of any item type and outputs a boolean.

## Example

To check whether a task list has pending tasks, connect the list to `items`. `is_empty` is true if there are no tasks and false if there are.

## Parameters

### items

Accepts list connections only. If no list is connected, the list counts as empty.

### is_empty

True if the list is empty or nothing is connected, false if it contains items.
