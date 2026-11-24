---
description: Holds a whole number that you set by hand and passes it to other nodes.
when_to_use: Use it when you want one editable integer in a workflow, such as a frame count, a shot number or a limit.
---

## When to use it

You need a whole number that you can change in one place and feed to other nodes. Typical uses are counters, indices, quantities, limits and thresholds. Integers are values like 1, 42, -7 or 0, and the default is 0.

The `integer` parameter is a property and an output only. You can type a value into it, but you cannot connect another node's output to it. The node does not round or check the value itself: setting `3.7` or `abc` directly stores it and passes it on unchanged, so enter whole numbers.

## Example

To cap an AI response at 500 tokens:

1. Add an **IntegerInput** node and set `integer` to 500.
1. Connect its output to the `max_tokens` parameter of a prompt model configuration node.
1. Responses from that configuration are now limited to 500 tokens. A value of 0 or less means the default limit is used.

The same node can hold a frame count for a render or the number of shots to process in a loop.
