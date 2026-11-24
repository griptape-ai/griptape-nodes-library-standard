---
description: Groups a list of dictionaries into a dictionary of lists.
when_to_use: Use it when you have records from a loop and want them collected by a shared field, or all values collected by key.
---

## When to use it

A loop over shots, files or prompts often gives you a list of dictionaries. This node gathers them into one dictionary. Groups keep the order in which their keys first appear.

## Example

With `group_by` empty, every key and value is collected under its key:

```json
[{"a": 1}, {"b": 2}, {"a": 3}]
```

gives `{"a": [1, 3], "b": [2]}`.

With `group_by` set to `shot`, each whole dictionary is filed under its `shot` value:

```json
[{"shot": "010", "file": "a.png"}, {"shot": "020", "file": "b.png"}, {"shot": "010", "file": "c.png"}]
```

gives `{"010": [{"shot": "010", "file": "a.png"}, {"shot": "010", "file": "c.png"}], "020": [{"shot": "020", "file": "b.png"}]}`.

## Parameters

### items

A list of dictionaries. A single dictionary counts as a list of one. Items given as JSON text are read as JSON. An item that isn't a dictionary stops the node with an error.

### group_by

The field name to group by. Group names are converted to text. If an item has no such field, the node stops with an error that names the item.
