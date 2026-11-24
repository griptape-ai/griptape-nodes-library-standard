---
description: Reads technical information from an audio file, such as duration, sample rate and codec.
when_to_use: Use it when you need to know how long a clip is or how it was encoded before you process it.
---

## When to use it

Connect audio to read its properties. The values update as soon as audio is set. You can feed them to other nodes, for example to check a clip's length.

If the audio can't be read, the node fails with the reason when it runs.

## Parameters

### duration

Length in seconds.

### format

The container name reported by ffprobe. It can list several names, such as `mov,mp4,m4a,3gp,3g2,mj2` for an MP4 or M4A file.

### file_size

Size in bytes.

### sample_rate / channels / bitrate / codec

Taken from the first audio stream in the file. The bitrate is for the whole file, in bits per second.
