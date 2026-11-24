---
description: Generates a unique temporary filename or path, with optional folder, prefix, name, suffix and extension.
when_to_use: Use it when you need a file name that won't clash with an existing file, such as for a scratch output.
---

## When to use it

You want a name no other file is using. The node picks a random name and builds it from the parts you give. No file is left on disk. Only the name is returned.

With `prefix` set to `render`, `name` set to `shot`, `suffix` empty and `extension` set to `png`, the result looks like `render_shot_k3j9x2ab.png`.

## Parameters

### directory

The folder to put the name in. If empty, the system temp folder is used. The folder must already exist, otherwise `filename` contains an error message.

### name

Text placed between the prefix and the random part.

### separator

Goes between prefix, name and the random part. If you clear it, `_` is used.

### suffix

Added after the random part and before the extension, so `suffix` `_backup` with `extension` `txt` gives `..._backup.txt`.

### return_absolute_path

On by default and returns the full path. Turn it off for the filename only.

### filename

The generated name. A new random name is generated each time the node runs or an input changes, so save the value if you need it to stay the same.

> [!WARNING]
> If the name can't be generated, the error text is put in `filename` as `Error: ...` and the node does not fail. Check for this before using the value as a path.
