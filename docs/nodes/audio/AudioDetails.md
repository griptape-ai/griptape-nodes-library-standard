---
description: Reads technical information from an audio file, such as duration, sample rate and codec.
when_to_use: Use it when you need to know how long a clip is or how it was encoded before you process it.
---

## When to use it

Connect audio to read its properties. The values update as soon as audio is set. You can feed them to other nodes, for example to check a clip's length.

The node needs `ffprobe`, which comes with FFmpeg. If it can't be found or can't read the audio, every output falls back to its default (0, or `UNKNOWN` for text) and the node reports no error. Check the engine log for the reason.

## Parameters

### duration

Length in seconds.

### format

The container name reported by ffprobe. It can list several names, such as `mov,mp4,m4a,3gp,3g2,mj2` for an MP4 or M4A file.

### file_size

Size in bytes.

### sample_rate / channels / bitrate / codec

Taken from the first audio stream in the file. The bitrate is for the whole file, in bits per second.
