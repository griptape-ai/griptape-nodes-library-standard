---
description: Reads a text file and converts its contents to a dictionary.
when_to_use: Use it when a dictionary is stored in a file such as JSON or a key: value text file.
---

## When to use it

Pick a file, or connect a path, and the node outputs its contents as a dictionary. The file is read as text and converted with the same rules as [ToDictionary](../convert/ToDictionary.md). JSON files and files of `key: value` lines both work. Text that can't be read as a dictionary comes out as `{"value": "<text>"}`.

The file picker lists text formats such as `.txt`, `.json`, `.yaml`, `.csv`, `.md` and `.xml`.

## Parameters

### file_path

The file to read. Connect a path or choose one with the picker.

### output

The dictionary built from the file.
