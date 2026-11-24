---
description: Loads a video file from disk or a URL and outputs it, with a player on the node.
when_to_use: Use it when you want to bring a plate, render or reference clip into a workflow.
---

## When to use it

You have a video and want to use it in a workflow. Pick a file with the file browser in the `video` field, or type a path or URL in `path`. The two fields stay in sync. You can also connect a video from another node to `video`. For example, load a plate and send it to [Trim Video](TrimVideo.md) to cut a shot.

The file picker accepts `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`, `.flv`, `.wmv` and `.m4v`.

When the file is local, the node reads the video's metadata with `ffprobe` and attaches it to the output for the player. If it can't resolve the file, it logs a warning and outputs the video without metadata.

## Parameters

### video

Shown in the editor as "Video or Path to Video". The loaded video. It also accepts a string path.

### path

Shown as "File Path or URL". A local file or a URL. It is also an output.

### Copy to Project

If the file is outside the project folder, the node shows a warning that it may not be available when the workflow is shared or run headlessly, and a **Copy to Project** button. The button copies the file into the project and points the node at the copy. It is hidden while a connection feeds `video`, because the connected value would replace the copy on the next run.
