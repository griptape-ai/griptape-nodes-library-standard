---
description: Starts a loop that runs the connected nodes once for each item in a list or dictionary.
when_to_use: Use it when you want to repeat the same steps for every image, prompt, or other item in a collection.
---

## When to use it

Connect a list or dictionary to `items`, build the steps you want repeated after **On Each Item**, and finish with a [ForEach End](ForEachEndNode.md). The steps run once per item. To avoid wiring a start and an end, use [ForEach Group](ForEachGroupNode.md).

## Parameters

### items

A list or dictionary. For a dictionary, each iteration receives `{"key": ..., "value": ...}`. An empty list or no input skips the loop without an error. Any other type fails with a message asking for a list or dictionary.

### run_in_order

On by default, so items run one after another. Turn it off to run them at the same time. Skip and break are only available in order, and the progress display is hidden in parallel mode.

### testing_mode

Runs a single iteration with the item at `test_item_index`. Useful for trying out the loop body on one item before running all of them. Turning it on hides `run_in_order` and the progress display.

### test_item_index

The zero-based position of the item to test. An index past the end uses the last item.

### current_item

The item for the current iteration. Connect it to the first node in the loop body.
