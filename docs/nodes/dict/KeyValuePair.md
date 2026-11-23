---
description: Makes a dictionary with one key and one value.
when_to_use: Use it when you need a one-entry dictionary, for example to give an Agent extra context or to build a dictionary piece by piece.
---

## When to use it

You have a key and a value and need them as a dictionary, for nodes that take dictionary inputs. It is also a way to build configuration parameters dynamically or to assemble a larger structure piece by piece.

To build a dictionary with several pairs, combine several Key Value Pair nodes with [Merge Key Value Pairs](MergeKeyValuePairs.md), or use [Create Dictionary](Dictionary.md) with lists of keys and values.

## Example

Set a configuration option:

1. Add a Key Value Pair node.
1. Set `key` to `max_tokens` and `value` to `1024`.
1. Connect the `dictionary` output to a node that takes configuration parameters.

The output is `{"max_tokens": "1024"}`.

## Parameters

### key

Text. The node does not convert other types, and the `key` input accepts text only.

### value

A value typed on the node is text, so `1024` becomes `"1024"`. Connect a number, boolean, list or other value to keep its type.

### dictionary

The single-pair dictionary `{key: value}`. It updates as soon as `key` or `value` is edited, and again when the node runs. With nothing set it is `{"": ""}`. It is shown as "Key/Value Pair" in the editor.
