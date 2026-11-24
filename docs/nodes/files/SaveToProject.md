---
description: Copies an existing file into the project, using a save situation to decide the folder and file name.
when_to_use: Use it when you have a file somewhere else, such as a download or a temp file, and want it stored in the project with the same naming rules as other saves.
---

## When to use it

A node gave you a path to a file outside your project outputs. This node reads that file and writes a copy using the project's save rules, so it ends up in the right folder with a safe name. The original is not touched.

## Parameters

### source

The file to save, as a path or a file artifact. The node fails if it is empty or can't be read.

### situation

The save situation, which sets the output folder and naming rules for the project. Use the refresh button to reload the list. The dropdown is hidden when a [FileOutputSettings](../filesystem/FileOutputSettings.md) node is connected to `output_file`, because that node sets its own rules.

### output_file

The file name to save as. It starts as `output`. When you set `source`, the extension is filled in from the source file, for example `output.png`. This only happens while the name is still `output`, so a name you typed is never changed.

Connect a FileOutputSettings node to control the folder, naming and what happens when the file already exists.

### saved_url

The saved file as a URL artifact. It holds the project macro path when the file landed inside the project, and the full path otherwise.
