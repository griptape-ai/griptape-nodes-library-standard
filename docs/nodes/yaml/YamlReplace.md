---
description: Replaces the value at a dot-notation path in YAML and outputs the modified YAML.
when_to_use: Use it when you need to change one field in a YAML document, such as a name or a setting, without retyping the rest.
---

## When to use it

You have YAML and want to swap one value. Set `path` to the field, put the new value in `replacement_value`, and read the changed YAML from `output`. The input is not modified. The node works on a copy.

## Parameters

### path

Dot notation with optional list indexes, such as `user.name` or `items[0].title`.

- Missing mapping keys along the path are created.
- An index past the end of a list extends the list. The new slots in between are `null`.
- If `path` is empty, the whole document is replaced by `replacement_value`.
- If the path runs into something that is not a mapping or list, such as a plain string, the YAML is returned unchanged with no error.

### replacement_value

The new value, written as YAML. It can be a single value, a mapping or a list. For example, `Jane` sets a string and a block of indented keys sets a mapping. If it is empty, the target is set to `null`.

### output

The full YAML with the replacement applied. It updates when `yaml`, `path` or `replacement_value` changes.
