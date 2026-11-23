---
description: Generates music from a text description with Eleven Labs.
when_to_use: Use it when you want a music track, such as a soundtrack or background bed, from a written brief.
---

## When to use it

Describe the music, choose a length and run. The result is saved as an audio file and returned as `audio_url`.
The node runs through Griptape Cloud and uses your Griptape key. Turn on the API Key Provider option to use your own Eleven Labs key instead. It waits for the result for up to `timeout` seconds (600 by default). If a run times out, the Refresh button in the Status group fetches the result later.

## Parameters

### music_duration_seconds

From 10 to 300 seconds. The default is 30.

### output_format

The format names use the pattern `codec_samplerate_bitrate`, for example `mp3_44100_128` is MP3 at 44.1 kHz and 128 kbps. The choices are MP3, PCM, mu-law, A-law and Opus. The default is `mp3_44100_128`.

### force_instrumental

Asks for music with no vocals.
