---
description: Pulls the audio track out of a video and saves it as an audio file in the format and quality you choose.
when_to_use: Use it when you need the sound from a clip on its own, for example to transcribe a take or send a dialogue track to an audio node.
---

## When to use it

You have a video and want only its sound. For example, extract the audio from a recorded review session and pass it to [Transcribe Audio](../audio/TranscribeAudio.md) to get written notes.

If the video has no audio stream, the node fails with "No audio stream found in the input video". The ffmpeg run times out after 300 seconds.

This node has no `output_frame_rate` or `processing_speed` setting. Those are hidden because it does not encode video.

## Parameters

### audio_format

`mp3` (default), `wav`, `aac`, `flac`, `ogg` or `m4a`. Changing it also changes the extension of `output_file`.

### audio_quality

`high` (128k, default), `medium` (96k), `low` (64k) or `copy`. The bitrate applies to `mp3`, `aac`, `m4a` and `ogg`. `wav` (uncompressed) and `flac` (lossless) ignore it unless you pick `copy`.

`copy` takes the original audio stream without re-encoding, which is fastest. The stream is written into a file of the chosen format, so the original codec has to be one that format can hold. If it is not, ffmpeg fails and the error is in the logs.

### extracted_audio

The audio file. It is saved to the project using the `output_file` setting, whose default name is `output.mp3`.
