---
description: Sets where and how a save node writes its file, using a situation template, a filename and rules for existing files.
when_to_use: Use it when you want one place to control the output folder, name and overwrite behavior, and to reuse it across save nodes.
---

## When to use it

Save nodes normally pick a folder from the project's situation. Connect this node to a save node's `output_file` input, such as on [SaveToProject](../files/SaveToProject.md), to take over those choices. The save node reads the settings directly. The `file_destination` output only previews the resulting path.

## How filename is read

| What you enter | What happens |
| --- | --- |
| A name such as `shot.png` | The situation's macro decides the folder and final name. |
| A name with a folder such as `a/shot.png` | Same, and the folder part is passed to the macro as `sub_dirs`. |
| An absolute path inside the project | It is stored as a macro path such as `{outputs}/shot.png`. |
| An absolute path outside the project | It is used exactly as written. A warning says this hurts portability. |

The macro receives the variables `file_name_base`, `file_extension` and `node_name`. `node_name` is the name of the save node this one is connected to, or this node's own name if none is.

## Parameters

### situation

Loads a situation's macro, existing-file rule and folder creation setting into the fields below. Changing the situation replaces any edits to those fields.

### macro

The path template used for plain names. Edit it to change the folder layout. The **Reset to Default** button restores the situation's value for `macro`, `if_file_exists` and `auto_create_path`.

### if_file_exists

What to do when the file is there already.

- **Increment Version** saves under the next available name, so the final name is not guaranteed.
- **Overwrite Existing** replaces the file.
- **Abort / Error** fails.

### auto_create_path

On by default. Creates missing folders when saving.

### coerce_extension_to_match_bytes

On by default. If the data doesn't match the extension, the extension is corrected, so JPEG data saved as `image.png` becomes `image.jpeg`. Turn it off to get an error instead.

### filename

The file name, with macros such as `{workflow_name}_output.png`.

### file_destination

A preview of the resolved path. An unfilled `{###}` version slot shows as `###`.
