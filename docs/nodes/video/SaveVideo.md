---
description: Saves a video from your workflow to a file in your project.
when_to_use: Use it when you want to keep a render, a generated clip or a processed video as a file.
---

## When to use it

Connect a video from a generator or an editing node to save it to disk. For example, save the output of a [Resize Video](ResizeVideo.md) node as a proxy for review. The video also passes through the `video` output, so you can keep building after the save.

If the video input is empty at run time, the node finishes with a warning and creates no file.

## Parameters

### situation

Picks where the file is saved, using the situations defined in your project. Use the refresh button to reload the list. The dropdown hides when a file-destination node is connected to `output_file`, because that node decides the location.

### output_file

The file name to save as. The default is `griptape_nodes.mp4`. The node writes the video bytes as they are and does not convert the format, so the extension should match the video you connect.

Before reading a video from a URL or path, the node asks the engine whether the codec may be read. If the engine denies it, the save fails with that reason and the details appear in `result_details`.
