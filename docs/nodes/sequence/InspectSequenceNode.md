---
description: Reports a Sequence's range, gaps, item count and path pattern, and passes the Sequence through unchanged.
when_to_use: Use it when you want to check a Sequence for missing items or see what a scan found.
---

## When to use it

You have a Sequence from [Scan Sequence](ScanSequenceNode.md) or [Scan Split Sequence](ScanSplitSequenceNode.md) and want to know if it is complete. The node reads the Sequence and fills its outputs with the answer.

The `sequence` parameter is both an input and an output, and the value passes through unchanged. You can put the node in the middle of a chain without breaking it.

The outputs update as soon as you connect or change a Sequence, without running the node. If no Sequence is connected, the status says so, and running the node raises an error.

## Parameters

### summary

Multi-line text that lists:

- the active range and the range found on disk
- the gap policy
- the digit width of the item number
- the pattern and the directory
- the item count and the count present on disk
- the missing item numbers, with the middle shortened if there are more than 10
- the number of negative item numbers dropped, if any
- the first and last item paths

Connect it to a Display or Note node to read it.

### existing_entries

Only the items that exist on disk. When a Sequence was made with *Fill gaps with nearest*, the filled slots are left out.

### gaps

The item numbers missing from the active range, in ascending order.

### is_complete

True when there are no gaps. It is false when no Sequence is connected.

### entry_count

How many items the Sequence carries. For a filled Sequence this includes the filled slots, so it can be larger than the number of files on disk.
