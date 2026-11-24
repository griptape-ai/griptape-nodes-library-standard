---
description: Builds a dictionary whose keys come from a list, with optional values matched by position.
when_to_use: Use it when you have a list of names, such as template fields, and want a dictionary with one editable entry for each.
---

## When to use it

Feed it a list like `["NAME", "CITY"]` and you get `{"NAME": "", "CITY": ""}`. Type values into the `KeyValuePairs` property, or supply them in `optional_values`. It suits fill-in-the-blank templates and forms.

## Example

| keys | optional_values | KeyValuePairs |
| ---- | --------------- | ------------- |
| `["one", "two", "three"]` | `["happy", "sad"]` | `{"one": "happy", "two": "sad", "three": ""}` |
| `["ADJ", "NOUN", "ADJ"]` | | `{"ADJ": "", "NOUN": "", "ADJ_2": ""}` |

## Parameters

### keys

Each item is converted to text and trimmed. Blank items are skipped. An empty list clears the output.

### optional_values

Matched to keys by position, not by name. Values are converted to text. Keys without a value get empty text.

### maintain_duplicates

On by default. A repeated key is numbered `_2`, `_3` and so on. When off, repeats are merged into one key.

### KeyValuePairs

The dictionary. It is also an editable property. When the keys change, values you typed are kept for keys that still exist, unless `optional_values` supplies a value for that key.
