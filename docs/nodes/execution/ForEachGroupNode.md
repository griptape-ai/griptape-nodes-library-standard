---
description: A group that runs the nodes inside it once for each item in a list or dictionary.
when_to_use: Use it when you want a ForEach loop without wiring separate start and end nodes.
---

## When to use it

Drop the nodes you want repeated into the group. Connect a list or dictionary to `items`, and use `current_item` and `index` inside the group. Send what you want to keep into `new_item_to_add`. The group does the work of [ForEach Start](ForEachStartNode.md) and [ForEach End](ForEachEndNode.md).

## Parameters

### items

A list or dictionary. For a dictionary, each iteration receives `{"key": ..., "value": ...}`. An empty list or no input skips the loop. Any other type fails with an error.

### current_item and index

The item and its position for this iteration. Connect them to nodes inside the group.

### execution_mode

`Run Group Items One at a Time` or `Run Group Items All at Once`. Skip and break are only available when running one at a time.

### execution_environment

Where the group runs. The choices depend on the libraries and publishers installed.

### testing_mode

Runs one iteration using the item at `test_item_index`. It hides `execution_mode`.

### test_item_index

The zero-based position of the item to test. A value past the end uses the last item.

### results

The collected items from each iteration, in order.
