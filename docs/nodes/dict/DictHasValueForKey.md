---
description: Checks whether a dictionary contains a key.
when_to_use: Use it when a branch or condition depends on whether a field exists in a dictionary.
---

## When to use it

Connect a dictionary and a key. `has_key` is true when the key is in the dictionary. It checks that the key exists, not that its value is filled in, so a key holding an empty value still counts. The key must match exactly and only top-level keys are checked.

An empty key, or an input that isn't a dictionary, gives false.
