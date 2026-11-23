---
description: Shows a float value on the node and passes it through unchanged.
when_to_use: Use it when you want to inspect a decimal value in your graph, for example while debugging.
---

## When to use it

You want to see a floating-point value in your graph, such as a temperature setting. The node does not modify the value. It shows it and passes it through, so it can sit in the middle of a chain.

## Example

1. Add a **Display Float** node.
1. Connect the `temperature` output from another node to its input.
1. The node shows the current temperature value.

## Parameters

### float

The default is 0.0. The same value is passed through as the output, also named `float`.
