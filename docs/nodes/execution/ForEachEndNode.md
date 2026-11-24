---
description: Ends a ForEach loop body, collects each iteration's result, and sends the loop back for the next item.
when_to_use: Use it when you have a ForEach Start and need to close the loop and gather the results.
---

## When to use it

Place it at the bottom of the loop body that begins with a [ForEach Start](ForEachStartNode.md). Each time an iteration reaches it, the item is stored and the loop moves on. When all items are done, the flow continues from **On Loop Complete** and `results` holds the list.

## Parameters

### new_item_to_add

The value to collect from this iteration. It is added only when the iteration arrives through **Add Item to Output**. The value is not computed when the iteration arrives through Skip or Break.

### Control inputs

**Add Item to Output** collects the item and continues. **Skip to Next Iteration** continues without collecting. **Break Out of Loop** stops the loop immediately. Skip and break only work when the start node has `run_in_order` on.

### results

A list of every collected item, in order.
