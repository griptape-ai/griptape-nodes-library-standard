---
description: Generates a video driven by an audio clip and a text prompt using LTX models, with an optional image.
when_to_use: Use it when you have a voice, music or sound clip and want a video that goes with it.
---

## When to use it

You have audio and want visuals for it. Connect the audio to `audio`, describe the video in `prompt`, and run the node. Add an image to `image` to guide how the video looks.

The node runs through the Griptape model proxy and needs a Griptape Cloud API key. Switch `api_key_provider` to **Customer** to use your own LTX Studio key instead, stored in the `LTX_API_KEY` secret. `timeout` sets how long the node waits, in seconds, and defaults to 1200. 0 means no limit. If the wait ends first, the Refresh button in the Status group retrieves a finished result.

## Parameters

### model

`ltx-2-pro`, `ltx-2-3-pro`, `ltx-2-5-pro` or `ltx-2-5-fast`. The default is `ltx-2-5-fast`.

### audio

Required. The node converts the audio to a format LTX accepts. WAV files are re-encoded to MP3 first.

> [!NOTE]
> If the audio cannot be loaded, the node reports "Connect audio to 'Input Audio'", the same message as for a missing connection.

### image

Optional image to guide the video.

### resolution

`1920x1080` or `1080x1920`. This node offers no other sizes and has no duration or frame rate setting.

### guidance_scale

Optional, from 1 to 50. Higher values follow the prompt more closely. Leave it empty to use the LTX default.

### video_url

The generated video. It is also saved to the project at `output_file`.
