---
description: Scans a numbered file path into one Sequence for each unbroken run of items on disk.
when_to_use: Use it when a numbered sequence has gaps and you want to handle each continuous run separately.
---

## When to use it

You have files like `render.0001.png` through `render.0005.png`, then `render.0008.png` through `render.0012.png`, then `render.0015.png`. This node returns three Sequences, one per run. The node always splits at gaps and has no gap policy to choose.

Connect the `sequences` list to a list node, or pick one Sequence out of it for [Inspect Sequence](InspectSequenceNode.md). If you want a single Sequence and a choice of how to treat gaps, use [Scan Sequence](ScanSequenceNode.md).

## Parameters

### template

The path with a token where the item number goes: `####`, `%04d` or `@@@`. `render.####.png` matches `render.0001.png`, `render.0002.png` and so on. A path with no token is treated as a sequence of one by default.

Project macros such as `{inputs}/render.####.png` are accepted. The returned paths keep the macro form.

### fail_on_empty_result

On by default. When the scan finds nothing, the node fails. Turn it off to get an empty list and a success status. The status message says whether no files matched the name, the digit count did not match the token, or the item range excluded everything.

### Advanced Sequence Control

A collapsed group that limits the scan. It works the same way as in [Scan Sequence](ScanSequenceNode.md#advanced-sequence-control). It holds the active range (`start_mode`, `end_mode` and their number or offset fields) and `no_token_behavior`.

### sequences

One Sequence per run. The status message lists the runs, for example `items 1–2, item 4, items 6–7`.

> [!NOTE]
> When the node fails, `sequences` is an empty list and the error goes down the failure path.
