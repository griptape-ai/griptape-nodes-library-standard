---
description: Holds a whole number that you set by hand and passes it to other nodes.
when_to_use: Use it when you want one editable integer in a workflow, such as a frame count, a shot number or a limit.
---

## When to use it

You need a whole number without decimals that you can change in one place and feed to other nodes. Typical uses are counters, indices, quantities, limits and thresholds. Integers are values like 1, 42, -7 or 0, and the default is 0.

## Example

To cap an AI response at 500 tokens:

1. Add an **IntegerInput** node and set `integer` to 500.
1. Connect its output to a model's `max_tokens` parameter.
1. The model now limits its responses to 500 tokens.

The same node can hold a frame count for a render or the number of shots to process in a loop.
