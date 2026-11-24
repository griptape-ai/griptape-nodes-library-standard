---
description: Loads an audio file from disk or a URL and outputs it, with a player on the node.
when_to_use: Use it when you want to bring a recording, such as a review session or a temp dialogue track, into a workflow.
---

## When to use it

You have an audio file and want to use it in a workflow. Pick a file with the file browser in the `audio` field, or type a path or URL in `path`. The two fields stay in sync. You can also connect audio from another node to `audio`. For example, load a temp dialogue track and send it to [Transcribe Audio](TranscribeAudio.md).

The file picker accepts `.mp3`, `.wav`, `.ogg`, `.flac`, `.m4a`, `.aac`, `.wma`, `.opus` and `.webm`.

## Parameters

### audio

Shown in the editor as "Audio or Path to Audio". The loaded audio, with a player.

### path

Shown as "File Path or URL". A local file or a URL. It is also an output, so the resolved path can be passed on.

### Copy to Project

If the file is outside the project folder, the node shows a warning that it may not be available when the workflow is shared or run headlessly, and a **Copy to Project** button. The button copies the file into the project and points the node at the copy. It is hidden while a connection feeds `audio`, because the connected value would replace the copy on the next run.
