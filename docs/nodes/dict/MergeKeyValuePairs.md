---
description: Combines several dictionaries into one.
when_to_use: Use it when you have key-value pairs from several nodes and need them in a single dictionary.
---

## When to use it

Connect dictionaries from [KeyValuePair](KeyValuePair.md), [Dictionary](Dictionary.md) or any other node. They are merged in order. If two share a key, the later one wins. Only top-level keys are merged. A nested dictionary is replaced, not combined. Inputs that aren't dictionaries are ignored.

## Example

Merging `{"a": 1, "b": 2}` and `{"b": 3}` gives `{"a": 1, "b": 3}`.
