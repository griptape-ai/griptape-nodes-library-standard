---
description: Outputs the number of items in a list.
when_to_use: Use it when you need to know how many items a list holds, to check a size or to drive other logic.
---

## When to use it

You want the size of a list, for example to check that it has a specific number of items or to validate its length before processing. The node works with lists of any item type.

## Example

To count the tasks in a task list, connect the list to `items`. `length` shows the number of tasks.

## Parameters

### items

Accepts list connections only. If the list is empty or nothing is connected, the length is 0.

### length

A non-negative integer.
