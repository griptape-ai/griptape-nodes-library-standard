---
description: Ends a For Loop body, collects each iteration's result, and sends the loop back for the next value.
when_to_use: Use it when you have a For Loop Start and need to close the loop and gather the results.
---

## When to use it

Place it at the bottom of the loop body that begins with a [For Loop Start](ForLoopStartNode.md). When all iterations are done, the flow continues from **On Loop Complete** and `results` holds the collected list.

## Parameters

### new_item_to_add

The value to collect from this iteration. It is added when the iteration arrives through **Add Item to Output**.

### Control inputs

**Add Item to Output** collects the item and continues. **Skip to Next Iteration** continues without collecting. **Break Out of Loop** stops the loop immediately. Skip and break are unavailable if the start node has `run_in_order` off.

### results

A list of every collected item, in order.
