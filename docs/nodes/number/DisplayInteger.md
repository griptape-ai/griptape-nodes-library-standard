---
description: Shows an integer value on the node and passes it through unchanged.
when_to_use: Use it when you want to inspect a whole-number value in your graph, for example while debugging.
---

## When to use it

You want to see an integer in your graph, such as a token count. The node does not modify the value. It shows it and passes it through, so it can sit in the middle of a chain.

## Example

1. Add a **Display Integer** node.
1. Connect an integer output to its input.
1. The node shows the current value.

## Parameters

### integer

The default is 0. The same value is passed through as the output, also named `integer`. The node does not convert or check the value: a value set directly is passed on in whatever type it has, so `2.7` stays `2.7`. A connection must come from an integer output. A float output is not accepted.
