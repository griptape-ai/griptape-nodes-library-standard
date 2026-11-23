---
description: Writes a message to the engine log at a chosen level and shows it on the node.
when_to_use: Use it when you want to record what is happening at a point in your workflow, for example while debugging.
---

## When to use it

Drop a Logger into a workflow to write a message to the console. The same text shows on the node, so you can read it without opening the log. An empty message is allowed if you only want a marker or timestamp from the log.

## Parameters

### log_level

One of DEBUG, INFO, WARNING, ERROR or CRITICAL. The default is INFO.

### include_node_name

Adds `[node name]` in front of the message. On by default. When the message has several lines, it starts on a new line after the name.

### output

The message as it will appear, with the level padded in front, in a code block. It updates as you edit the other parameters, before the node runs.

### passthrough

Optional. Logger has no control connections of its own, so it only runs when something downstream needs it. Connect `passthrough` to a downstream node's input to force it to run.
