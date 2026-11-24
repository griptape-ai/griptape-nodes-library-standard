---
description: Writes a dictionary to a text file.
when_to_use: Use it when you need to keep a dictionary on disk or hand it to another tool as a file.
---

## When to use it

Connect a dictionary and set the file name. The file is replaced if it already exists. An empty path stops the node with an error.

> [!WARNING]
> The file holds the dictionary in Python form, with single quotes, `True` and `None`. It is not valid JSON. For a JSON file, convert with [ToJson](../convert/ToJson.md) and save with [SaveText](../text/SaveText.md). [LoadDictionary](LoadDictionary.md) can read the Python form back.

## Parameters

### output_path

Defaults to `griptape_output.txt`. The path is used as given.
