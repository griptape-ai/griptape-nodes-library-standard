---
description: Holds a block of text that you type and passes it to other nodes, optionally also as a list.
when_to_use: Use it to write prompts, notes or lists once and feed them to other nodes.
---

## When to use it

You want to type text, such as a prompt, a list of shot names, or a set of instructions, and connect it to other nodes. Line breaks are kept. Connect `text` to an [Agent](../agents/Agent.md) prompt or any other text input.

The text is typed in the node. It cannot receive a connection.

## Example

1. Add **Text Input** and enter one shot code per line, such as `SH010`, `SH020` and `SH030`.
1. Turn on `split_text` and use `output_split` to get a list of the three codes.
1. Connect `output_split` to a loop to process each shot.

## Parameters

### text

The text to pass on.

### split_text

Off by default. When on, the text is also split into a list that comes out of `output_split`. The split options appear in a collapsed group.

### split_text_options

Splitting works the same as the [Split Text](../lists/SplitText.md) node: `split_mode`, `delimiter_type`, `include_delimiter`, `trim_whitespace` and `remove_empty`. In this node `remove_empty` is on by default and the options do not take connections. If the text cannot be split, `output_split` is an empty list.

### output_split

The list produced by the split. It is only visible when `split_text` is on.
