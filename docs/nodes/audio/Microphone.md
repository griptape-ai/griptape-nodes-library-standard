---
description: Records audio from your microphone in the editor and outputs it as an audio file.
when_to_use: Use it when you want to record a voice note or narration and feed it into a workflow.
---

## When to use it

You want to record audio instead of loading a file. Use the capture control on the node to record. The recording is the node's `audio` output. For example, record a quick voice note about a shot and connect it to [Transcribe Audio](TranscribeAudio.md) to get written notes.

The `audio` parameter has no input connection. It is set by the recording only. If you already have an audio file, use [Load Audio](LoadAudio.md).
