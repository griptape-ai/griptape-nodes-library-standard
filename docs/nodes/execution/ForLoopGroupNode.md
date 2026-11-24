---
description: A group that runs the nodes inside it once for each number in a range.
when_to_use: Use it when you want a counting loop without wiring separate start and end nodes.
---

## When to use it

Drop the nodes to repeat into the group and set the range. Use `index` inside the group for the current number. Send what you want to keep into `new_item_to_add`. The group does the work of [For Loop Start](ForLoopStartNode.md) and [For Loop End](ForLoopEndNode.md).

## Parameters

### start and end

The first and last values. If `start` is greater than `end`, the loop counts down.

### end_inclusive

On by default. When off, the loop stops before `end`. It sits in the group settings.

### step

How much to move each time, from 1 to 1000. It is always positive and is subtracted when counting down. A value below 1 fails validation.

### index

The current number. Connect it to nodes inside the group.

### execution_mode

`Run Group Items One at a Time` or `Run Group Items All at Once`. Skip and break are only available when running one at a time.

### execution_environment

Where the group runs. The choices depend on the libraries and publishers installed.

### results

The collected items from each iteration, in order.
