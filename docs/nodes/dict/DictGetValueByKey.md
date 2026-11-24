---
description: Looks up a value in a dictionary by key, with an optional fallback when the key is missing.
when_to_use: Use it when you need one field out of a dictionary, such as a prompt or a file name.
---

## When to use it

Connect a dictionary and type the key. The `value` output is what is stored under that key. The key must match exactly and only top-level keys are searched.

## Parameters

### key

If the key is empty, the output is nothing and there is no error.

### supply_default_if_not_found

On by default. When on, a missing key gives `default_value_if_not_found`. When off, a missing key stops the node with an error. A `dict` input that isn't a dictionary is handled the same way.

### default_value_if_not_found

The fallback. Shown only when `supply_default_if_not_found` is on.

### value

The output takes the type of what was found, such as text, number or list.

> [!WARNING]
> When the type of the value changes, connections from `value` to inputs that can't accept the new type are removed.
