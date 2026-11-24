---
description: Generates music from a text description with Eleven Labs.
when_to_use: Use it when you want a music track, such as a soundtrack or background bed, from a written brief.
---

## When to use it

Describe the music, choose a length and run. The result is saved as an audio file and returned as `audio_url`.

## What it needs

- A Griptape Nodes license or a Griptape Cloud API key (`GT_CLOUD_API_KEY`). To use your own ElevenLabs key (`ELEVENLABS_API_KEY`) instead, set **API Key Provider** to **Customer**.
- The node waits for the result. `timeout` sets how long, 600 seconds by default, or 0 for no limit. If a run times out, the **Refresh** button on the node fetches the result later.

## Parameters

### music_duration_seconds

From 10 to 300 seconds. The default is 30.

### output_format

The format names use the pattern `codec_samplerate_bitrate`, for example `mp3_44100_128` is MP3 at 44.1 kHz and 128 kbps. The choices are MP3, PCM, mu-law, A-law and Opus. The default is `mp3_44100_128`.

### force_instrumental

Asks for music with no vocals.
