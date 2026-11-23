---
description: Compares two numbers and sends the workflow down a True or False path.
when_to_use: Use it when the next step depends on how two numbers relate, such as a count being above a limit.
---

## When to use it

Pick a comparison, give it two numbers, and the flow continues from **True** or **False**. The boolean answer is also stored in `result`, which is hidden on the node.

## Parameters

### evaluate

The comparison to make: `A == B`, `A != B`, `A < B`, `A > B`, `A <= B` or `A >= B`.

> [!WARNING]
> The default value in the source is `==`, which is not one of the choices. Until you pick a choice, the comparison returns False.

### A and B

The two numbers. Integers and decimals are accepted. Both default to 0.
