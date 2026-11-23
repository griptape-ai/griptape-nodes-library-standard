---
description: Scans a numbered file path such as render.####.png into one Sequence and a flat list of file paths.
when_to_use: Use it when you have a numbered image or file sequence on disk and want to loop over it or pass it to other nodes.
---

## When to use it

You have files like `render.0001.png`, `render.0002.png` and so on. Give the node a pattern and it finds every item that exists and returns them in item-number order. Use the `paths` output for anything that takes a list of file paths. Use the `sequence` output for [Inspect Sequence](InspectSequenceNode.md) or any node that takes a Sequence.

If the folder has gaps and you want each unbroken run as its own Sequence, use [Scan Split Sequence](ScanSplitSequenceNode.md).

## Parameters

### template

The path with a token where the item number goes: `####`, `%04d` or `@@@`. `render.####.png` matches `render.0001.png`, `render.0002.png` and so on. A path with no token is treated as a sequence of one by default (see `no_token_behavior`).

Project macros such as `{inputs}/render.####.png` are accepted. The returned paths keep the macro form, so the workflow still works on another machine.

### missing_item_policy

What to do when items are missing inside the range. With items 1, 2, 4, 6 and 7 on disk:

| Choice | Result |
| --- | --- |
| Skip over gaps | `1, 2, 4, 6, 7` |
| Fill gaps with nearest | `1, 2, 2, 4, 4, 6, 7`. Slot 3 reuses item 2 and slot 5 reuses item 4. |
| Abort on sequence gaps | The node fails at the first gap. |

### fail_on_empty_result

On by default. When the scan finds nothing, the node fails, which usually means the path, range or token is wrong. Turn it off to get an empty Sequence and a success status instead. The status message says whether no files matched the name, the digit count did not match the token, or the item range excluded everything.

### Advanced Sequence Control

A collapsed group that limits the scan. Its parameters are `start_mode`, `start_number`, `start_offset`, `end_mode`, `end_number`, `end_offset` and `no_token_behavior`.

- `start_mode` and `end_mode` choose the active range. Pick the first or last item on disk, a specific item number, or an offset from the first or last item. The number or offset field only shows when its mode is selected.
- Numbers and offsets must be 0 or greater.
- `no_token_behavior` controls a path with no token, such as `render.0002.png`. *Treat as a single file* returns only that file. *Treat as part of a sequence* reads the digits as an item number and collects the neighbouring files. *Fail unless a token is present* fails with an error.

### paths

The file paths in item-number order.

### sequence

The Sequence object, with items, range and metadata.

> [!NOTE]
> When the node fails, both outputs are empty and the error goes down the failure path.
