---
description: Starts a loop that counts from a start number to an end number by a step and runs the connected nodes each time.
when_to_use: Use it when you want to repeat steps a set number of times, or step through a numeric range.
---

## When to use it

Set a range, build the steps after **On Each**, and close the loop with a [For Loop End](ForLoopEndNode.md). The current number is available on the `index` output. To skip wiring a start and an end, use [For Loop Group](ForLoopGroupNode.md).

## Example

`start` 5, `end` 10 and `step` 2 run for 5, 7 and 9. With `end` 9 the loop also runs for 9 when `end_inclusive` is on, and stops at 7 when it is off.

## Parameters

### start and end

The first and last values. If `start` is greater than `end`, the loop counts down.

### end_inclusive

On by default. When off, the loop stops before reaching `end`. With `start` equal to `end`, one iteration runs when on and none when off.

### step

How much to move each time, from 1 to 1000. It is always positive. When counting down it is subtracted. A value below 1 fails validation.

### run_in_order

On by default. Turn it off to run iterations at the same time. Skip and break are not available in parallel mode, and the progress display is hidden.
