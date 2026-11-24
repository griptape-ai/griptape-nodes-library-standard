---
description: Shows a dictionary as text on the node and passes it through unchanged.
when_to_use: Use it when you want to see what a dictionary holds at a point in the workflow.
---

## When to use it

Connect a dictionary to check its contents while building a workflow. The same dictionary is available on the `dictionary` output, so the node can sit in the middle of a chain.

The display shows Python form, with single quotes, `True` and `None`. It is not JSON. For JSON, use [DisplayJson](../json/DisplayJson.md).
