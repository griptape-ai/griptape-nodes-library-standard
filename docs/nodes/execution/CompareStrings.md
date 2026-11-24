---
description: Compares two strings and sends the workflow down a True or False path.
when_to_use: Use it when the next step depends on matching or containing text, such as checking a keyword.
---

## When to use it

Pick a comparison, give it two strings, and the flow continues from **True** or **False**. The boolean answer is also stored in `result`, which is hidden on the node.

## Parameters

### evaluate

`A == B`, `A != B`, `A < B`, `A > B`, `A <= B` and `A >= B` compare the strings in alphabetical order. `A in B` is true when A appears anywhere inside B. `A not in B` is the opposite.

### case_sensitive

On by default. Turn it off to compare in lowercase, so `Cat` matches `cat`.

### A and B

The two strings. Both default to empty.
