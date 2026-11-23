---
description: Joins two lists into one, with all items of List A followed by all items of List B.
when_to_use: Use it when you need a single list from two, such as shot lists from two sequences.
---

## When to use it

You have two lists and want one list that holds the items of both, in order. The operation is plain concatenation (`list_a + list_b`). The original lists are not modified.

## Example

```python
# List A: [1, 2]
# List B: [3, 4]
# Output: [1, 2, 3, 4]

# List A: ["a", "b"]
# List B: ["c", "d"]
# Output: ["a", "b", "c", "d"]

# List A: [1, 2]
# List B: []
# Output: [1, 2]
```

## Parameters

### list_a

If the input is not a list, it is treated as an empty list.

### list_b

If the input is not a list, it is treated as an empty list.

### output

A new list. Nested lists inside either input are not flattened. Use [Flatten List](FlattenList.md) if you need that.
