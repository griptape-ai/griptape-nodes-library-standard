---
description: Saves audio from your workflow to a file in your project.
when_to_use: Use it when you want to keep generated or processed audio as a file.
---

## When to use it

Connect audio from a generator, trimmer or mixer to save it to disk. The audio also passes through the `audio` output, so you can keep building after the save.

If the audio input is empty, validation fails with "Audio parameter is required" and the run is cancelled before this node starts. If audio is connected but yields no data, the node finishes with a warning and creates no file.

## Parameters

### situation

Picks where the file is saved, using the situations defined in your project. Use the refresh button to reload the list. The dropdown hides when a file-destination node is connected to `output_file`, because that node decides the location.

### output_file

The file name to save as. The default is `griptape_nodes.mp3`. The node writes the audio bytes as they are and does not convert the format, so the extension should match the audio you connect.
